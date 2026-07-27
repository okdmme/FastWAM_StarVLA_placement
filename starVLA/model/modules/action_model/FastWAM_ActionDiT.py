import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Any, Dict, Mapping, Optional

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def gradient_checkpoint_forward(model, use_gradient_checkpointing, *args, **kwargs):
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            lambda *inputs: model(*inputs, **kwargs),
            *args,
            use_reentrant=False,
        )
    return model(*args, **kwargs)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None):
    bsz, q_len, _ = q.shape
    k_len = k.shape[1]
    head_dim = q.shape[-1] // num_heads
    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return x.transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x, freqs, num_heads):
    bsz, seq_len, _ = x.shape
    head_dim = x.shape[-1] // num_heads
    x = x.view(bsz, seq_len, num_heads, head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(bsz, seq_len, num_heads, -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    return torch.view_as_real(x_out * freqs).flatten(2).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, attn_hidden_dim)
        self.o = nn.Linear(attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
        k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
        v = self.v(x)
        return self.o(flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, attn_hidden_dim)
        self.o = nn.Linear(attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        return self.o(flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask))


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.ffn(input_x)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)


class FastWAMActionDiTHead(nn.Module):
    """StarVLA-compatible action head wrapper around FastWAM ActionDiT."""

    def __init__(self, global_config: Any):
        super().__init__()
        action_model_cfg = global_config.framework.action_model
        action_dit_config = _build_action_dit_config(action_model_cfg)
        pretrained_path = _get_cfg_value(action_model_cfg, "action_dit_pretrained_path", None)
        skip_pretrained = bool(_get_cfg_value(action_model_cfg, "skip_dit_load_from_pretrain", False))

        if pretrained_path or skip_pretrained:
            device = _get_cfg_value(action_model_cfg, "device", "cpu")
            torch_dtype = _parse_torch_dtype(_get_cfg_value(action_model_cfg, "torch_dtype", torch.float32))
            self.model = ActionDiT.from_pretrained(
                action_dit_config=action_dit_config,
                action_dit_pretrained_path=pretrained_path,
                skip_dit_load_from_pretrain=skip_pretrained,
                device=device,
                torch_dtype=torch_dtype,
            )
        else:
            self.model = ActionDiT(**action_dit_config)

        self.action_dim = int(action_dit_config["action_dim"])
        self.action_horizon = int(
            _get_cfg_value(
                action_model_cfg,
                "action_horizon",
                int(_get_cfg_value(action_model_cfg, "future_action_window_size", 7)) + 1,
            )
        )
        self.num_inference_timesteps = int(_get_cfg_value(action_model_cfg, "num_inference_timesteps", 4))
        self.num_timestep_buckets = int(_get_cfg_value(action_model_cfg, "num_timestep_buckets", 1000))
        self.noise_s = float(_get_cfg_value(action_model_cfg, "noise_s", 0.999))
        self.beta_dist = Beta(
            float(_get_cfg_value(action_model_cfg, "noise_beta_alpha", 1.5)),
            float(_get_cfg_value(action_model_cfg, "noise_beta_beta", 1.0)),
        )

        state_dim = int(_get_cfg_value(action_model_cfg, "state_dim", 0) or 0)
        self.state_encoder = nn.Linear(state_dim, int(action_dit_config["text_dim"])) if state_dim > 0 else None

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _prepare_context(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if vl_embs.ndim != 3:
            raise ValueError(f"`vl_embs` must be 3D [B, L, D], got shape {tuple(vl_embs.shape)}")

        context = vl_embs
        if encoder_attention_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]), device=context.device, dtype=torch.bool
            )
        else:
            context_mask = encoder_attention_mask.to(device=context.device, dtype=torch.bool)

        if self.state_encoder is not None:
            if state is None:
                raise ValueError("`state` is required when `state_dim` is configured for FastWAMActionDiTHead.")
            if state.ndim == 3:
                state = state[:, 0, :]
            if state.ndim != 2:
                raise ValueError(f"`state` must be 2D or 3D, got shape {tuple(state.shape)}")
            state_token = self.state_encoder(state.to(device=context.device, dtype=context.dtype)).unsqueeze(1)
            context = torch.cat([context, state_token], dim=1)
            state_mask = torch.ones((context_mask.shape[0], 1), device=context_mask.device, dtype=torch.bool)
            context_mask = torch.cat([context_mask, state_mask], dim=1)

        return context, context_mask

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context, context_mask = self._prepare_context(vl_embs, state, encoder_attention_mask)
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t_bc = t[:, None, None]
        noisy_actions = (1 - t_bc) * noise + t_bc * actions
        velocity = actions - noise
        t_discretized = (t * self.num_timestep_buckets).long()
        pred_velocity = self.model(
            action_tokens=noisy_actions,
            timestep=t_discretized,
            context=context,
            context_mask=context_mask,
        )
        return ((pred_velocity.float() - velocity.float()) ** 2).mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context, context_mask = self._prepare_context(vl_embs, state, encoder_attention_mask)
        actions = torch.randn(
            (context.shape[0], self.action_horizon, self.action_dim),
            device=context.device,
            dtype=context.dtype,
        )
        dt = 1.0 / float(self.num_inference_timesteps)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            timestep = torch.full(
                (context.shape[0],),
                int(t_cont * self.num_timestep_buckets),
                device=context.device,
                dtype=torch.long,
            )
            pred_velocity = self.model(
                action_tokens=actions,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def _maybe_to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping-like config, got {type(value)}")


def _get_cfg_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _parse_torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.removeprefix("torch.")
        if hasattr(torch, normalized):
            dtype = getattr(torch, normalized)
            if isinstance(dtype, torch.dtype):
                return dtype
    raise ValueError(f"Unsupported torch dtype value: {value!r}")


def _build_action_dit_config(action_model_cfg: Any) -> Dict[str, Any]:
    explicit_cfg = _get_cfg_value(action_model_cfg, "action_dit_config", None)
    if explicit_cfg is not None:
        cfg = _maybe_to_plain_dict(explicit_cfg)
    else:
        hidden_dim = int(_get_cfg_value(action_model_cfg, "action_hidden_dim"))
        num_heads = int(_get_cfg_value(action_model_cfg, "num_heads", 8))
        cfg = {
            "hidden_dim": hidden_dim,
            "action_dim": int(_get_cfg_value(action_model_cfg, "action_dim")),
            "ffn_dim": int(_get_cfg_value(action_model_cfg, "ffn_dim", hidden_dim * 4)),
            "text_dim": int(_get_cfg_value(action_model_cfg, "text_dim", hidden_dim)),
            "freq_dim": int(_get_cfg_value(action_model_cfg, "freq_dim", 256)),
            "eps": float(_get_cfg_value(action_model_cfg, "eps", 1e-6)),
            "num_heads": num_heads,
            "attn_head_dim": int(_get_cfg_value(action_model_cfg, "attn_head_dim", hidden_dim // num_heads)),
            "num_layers": int(_get_cfg_value(action_model_cfg, "num_layers", 12)),
            "use_gradient_checkpointing": bool(
                _get_cfg_value(action_model_cfg, "use_gradient_checkpointing", False)
            ),
        }

    required_keys = {
        "hidden_dim",
        "action_dim",
        "ffn_dim",
        "text_dim",
        "freq_dim",
        "eps",
        "num_heads",
        "attn_head_dim",
        "num_layers",
    }
    missing_keys = sorted(required_keys - set(cfg))
    if missing_keys:
        raise ValueError(f"Missing ActionDiT config keys: {missing_keys}")
    return cfg


def get_action_model(config=None):
    """
    Factory: build FastWAM ActionDiT action head from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FastWAMActionDiTHead: StarVLA-compatible FastWAM action diffusion head.
    """
    if config is None:
        raise ValueError("FastWAM_ActionDiT.get_action_model requires a global config.")

    return FastWAMActionDiTHead(global_config=config)
