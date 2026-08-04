# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
FastWAM Framework.

This is the StarVLA framework entry for the full FastWAM architecture:

    FastWAM_WanVideoDiT + FastWAM_ActionDiT + inline MoT

The file is intentionally separate from WanFastWAM.py. WanFastWAM.py remains the
earlier thin Wan2-hidden-states-to-ActionDiT prototype, while this module is the
landing point for the paper-style FastWAM joint video/action training path.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.FastWAM_ActionDiT import ActionDiT
from starVLA.model.modules.world_model.FastWAM_WanVideoDiT import WanVideoDiT, flash_attention, modulate, rope_apply
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all

class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler used by FastWAM."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta


@dataclass
class FastWAMDefaultConfig:
    """Default parameters for the full FastWAM framework entry."""

    name: str = "FastWAM"

    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "video_dit_config": {
                "hidden_dim": 3072,
                "in_dim": 48,
                "ffn_dim": 14336,
                "out_dim": 48,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1.0e-6,
                "patch_size": (1, 2, 2),
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
                "has_image_input": False,
                "seperated_timestep": True,
                "fuse_vae_embedding_in_latents": True,
                "action_conditioned": True,
                "action_dim": 7,
                "video_attention_mask_mode": "bidirectional",
                "use_gradient_checkpointing": False,
            },
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 7,
            "state_dim": 7,
            "proprio_dim": None,
            "action_horizon": 8,
            "action_dit_config": {
                "action_dim": 7,
                "hidden_dim": 1024,
                "ffn_dim": 4096,
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1.0e-6,
                "use_gradient_checkpointing": False,
            },
        }
    )

    mot: dict = field(
        default_factory=lambda: {
            "mot_checkpoint_mixed_attn": True,
            "attention_mode": "fastwam",
        }
    )

    scheduler: dict = field(
        default_factory=lambda: {
            "video_train_shift": 5.0,
            "video_infer_shift": 5.0,
            "video_num_train_timesteps": 1000,
            "action_train_shift": 5.0,
            "action_infer_shift": 5.0,
            "action_num_train_timesteps": 1000,
        }
    )

    loss: dict = field(
        default_factory=lambda: {
            "lambda_video": 1.0,
            "lambda_action": 1.0,
        }
    )

    encoder: dict = field(
        default_factory=lambda: {
            "load_wan2_encoders": False,
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "text_max_length": 512,
            "height": 480,
            "width": 832,
            "num_frames": None,
        }
    )

    checkpoint: dict = field(
        default_factory=lambda: {
            "path": None,
            "strict": False,
        }
    )


@FRAMEWORK_REGISTRY.register("FastWAM")
class FastWAMFramework(baseframework):
    """Full FastWAM framework scaffold for StarVLA."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(FastWAMDefaultConfig, config)

        self.video_expert: WanVideoDiT | None = None
        self.action_expert: ActionDiT | None = None
        self.mot: MoT | None = None
        self.train_video_scheduler: WanContinuousFlowMatchScheduler | None = None
        self.infer_video_scheduler: WanContinuousFlowMatchScheduler | None = None
        self.train_action_scheduler: WanContinuousFlowMatchScheduler | None = None
        self.infer_action_scheduler: WanContinuousFlowMatchScheduler | None = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.video_processor = None
        self.loaded_checkpoint = None
        self.proprio_dim = None
        self.proprio_encoder = None

        self._build_experts()
        self._build_schedulers()
        checkpoint_cfg = self.config.framework.checkpoint
        checkpoint_path = checkpoint_cfg.get("path", None)
        if checkpoint_path:
            self.load_checkpoint(
                checkpoint_path,
                strict=bool(checkpoint_cfg.get("strict", False)),
            )
        encoder_cfg = self.config.framework.encoder
        if bool(encoder_cfg.get("load_wan2_encoders", False)):
            self._ensure_wan2_encoders()

    def _build_experts(self) -> None:
        """Build FastWAM experts from StarVLA config.

        VAE/text encoder/checkpoint loading and StarVLA dataloader adaptation are
        intentionally left for the next integration step. This method wires the
        three core trainable FastWAM modules into the intended StarVLA locations.
        """

        wm_cfg = self.config.framework.world_model
        action_cfg = self.config.framework.action_model
        mot_cfg = self.config.framework.mot

        video_dit_config = dict(wm_cfg.video_dit_config)
        action_dit_config = dict(action_cfg.action_dit_config)
        action_dit_config["action_dim"] = int(action_cfg.action_dim)

        self.video_expert = WanVideoDiT(**video_dit_config)
        self.action_expert = ActionDiT(**action_dit_config)
        self._build_proprio_encoder(action_cfg.get("proprio_dim", None))

        if int(self.action_expert.num_heads) != int(self.video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for FastWAM MoT.")
        if int(self.action_expert.attn_head_dim) != int(self.video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for FastWAM MoT.")
        if int(len(self.action_expert.blocks)) != int(len(self.video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert for FastWAM MoT.")

        self.mot = MoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            mot_checkpoint_mixed_attn=bool(mot_cfg.get("mot_checkpoint_mixed_attn", True)),
        )
        self.dit = self.mot

    def _build_proprio_encoder(self, proprio_dim: Optional[int]) -> None:
        if proprio_dim is None:
            self.proprio_dim = None
            self.proprio_encoder = None
            return
        proprio_dim = int(proprio_dim)
        if proprio_dim <= 0:
            raise ValueError(f"`proprio_dim` must be positive or None, got {proprio_dim}.")

        text_dim = int(self.config.framework.world_model.video_dit_config.text_dim)
        device, dtype = self._model_device_dtype()
        self.proprio_dim = proprio_dim
        self.proprio_encoder = nn.Linear(proprio_dim, text_dim).to(device=device, dtype=dtype)

    def _build_schedulers(self) -> None:
        """Build FastWAM flow schedulers without adding a separate scheduler module."""

        scheduler_cfg = self.config.framework.scheduler
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_cfg.get("video_num_train_timesteps", 1000)),
            shift=float(scheduler_cfg.get("video_train_shift", 5.0)),
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_cfg.get("video_num_train_timesteps", 1000)),
            shift=float(scheduler_cfg.get("video_infer_shift", 5.0)),
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_cfg.get("action_num_train_timesteps", 1000)),
            shift=float(scheduler_cfg.get("action_train_shift", 5.0)),
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_cfg.get("action_num_train_timesteps", 1000)),
            shift=float(scheduler_cfg.get("action_infer_shift", 5.0)),
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

    def load_checkpoint(self, path: str, optimizer=None, strict: bool = False) -> dict[str, Any]:
        """Load official FastWAM-style checkpoints into the StarVLA FastWAM modules.

        Official FastWAM saves the trainable joint model under `payload["mot"]`.
        The `mot` state_dict contains both video and action expert parameters
        because MoT owns them in its `mixtures` ModuleDict. Legacy Wan-only
        checkpoints may contain `payload["dit"]`; those are loaded into the
        video expert only and do not make action inference meaningful.
        """

        payload = torch.load(path, map_location="cpu", mmap=True)
        if not isinstance(payload, dict):
            raise TypeError(f"FastWAM checkpoint must be a dict, got {type(payload)!r}: {path}")

        load_report: dict[str, Any] = {
            "path": str(path),
            "loaded": [],
            "missing_keys": [],
            "unexpected_keys": [],
            "step": payload.get("step", None),
        }

        if "proprio_encoder" in payload:
            proprio_state = payload["proprio_encoder"]
            if not isinstance(proprio_state, dict) or "weight" not in proprio_state:
                raise ValueError("Checkpoint `proprio_encoder` must be a state_dict with a `weight` tensor.")
            proprio_weight = proprio_state["weight"]
            if proprio_weight.ndim != 2:
                raise ValueError(
                    "Checkpoint `proprio_encoder.weight` must be 2D "
                    f"[text_dim, proprio_dim], got {tuple(proprio_weight.shape)}."
                )
            checkpoint_text_dim, checkpoint_proprio_dim = int(proprio_weight.shape[0]), int(proprio_weight.shape[1])
            config_text_dim = int(self.config.framework.world_model.video_dit_config.text_dim)
            if checkpoint_text_dim != config_text_dim:
                raise ValueError(
                    "Checkpoint `proprio_encoder` output dim must match FastWAM text_dim: "
                    f"checkpoint={checkpoint_text_dim}, config={config_text_dim}."
                )
            if self.proprio_encoder is None:
                self._build_proprio_encoder(checkpoint_proprio_dim)
            elif int(self.proprio_dim) != checkpoint_proprio_dim:
                raise ValueError(
                    "Checkpoint `proprio_encoder` input dim does not match configured proprio_dim: "
                    f"checkpoint={checkpoint_proprio_dim}, config={self.proprio_dim}."
                )

        if "mot" in payload:
            incompatible = self.mot.load_state_dict(payload["mot"], strict=bool(strict))
            load_report["loaded"].append("mot")
            load_report["missing_keys"] = list(incompatible.missing_keys)
            load_report["unexpected_keys"] = list(incompatible.unexpected_keys)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into FastWAM video expert only.")
            incompatible = self.video_expert.load_state_dict(payload["dit"], strict=bool(strict))
            load_report["loaded"].append("video_expert")
            load_report["missing_keys"] = list(incompatible.missing_keys)
            load_report["unexpected_keys"] = list(incompatible.unexpected_keys)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        if "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            load_report["loaded"].append("proprio_encoder")
            load_report["proprio_dim"] = int(self.proprio_dim)

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            load_report["loaded"].append("optimizer")

        self.loaded_checkpoint = load_report
        if load_report["missing_keys"] or load_report["unexpected_keys"]:
            logger.warning(
                "Loaded FastWAM checkpoint %s with missing_keys=%d unexpected_keys=%d",
                path,
                len(load_report["missing_keys"]),
                len(load_report["unexpected_keys"]),
            )
        else:
            logger.info("Loaded FastWAM checkpoint %s", path)
        return load_report

    @staticmethod
    def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B,D], got shape {tuple(proprio.shape)}.")
        if int(proprio.shape[1]) != int(self.proprio_dim):
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}.")
        if int(proprio.shape[0]) != int(context.shape[0]):
            raise ValueError(
                f"`proprio` batch size must match context batch size, got {proprio.shape[0]} vs {context.shape[0]}."
            )

        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    def _normalize_proprio_tensor(
        self,
        proprio: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_sequence: bool = True,
    ) -> torch.Tensor:
        if self.proprio_encoder is None:
            raise ValueError("`proprio` was provided but `framework.action_model.proprio_dim` is None.")
        proprio = self._as_tensor(proprio, device=device, dtype=dtype)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 3 and allow_sequence:
            if proprio.shape[1] < 1:
                raise ValueError("`proprio` sequence must contain at least one timestep.")
            proprio = proprio[:, 0, :]
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be [B,D], [D], or [B,T,D], got shape {tuple(proprio.shape)}.")
        if int(proprio.shape[0]) != int(batch_size):
            raise ValueError(f"`proprio` batch size must be {batch_size}, got {proprio.shape[0]}.")
        if int(proprio.shape[1]) != int(self.proprio_dim):
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}.")
        return proprio

    def _examples_to_sample(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("FastWAM forward received an empty examples list.")
        if "input_latents" in examples[0] or "latents" in examples[0]:
            latents_key = "input_latents" if "input_latents" in examples[0] else "latents"
            sample = {
                "input_latents": torch.stack([torch.as_tensor(example[latents_key]) for example in examples], dim=0),
                "context": torch.stack([torch.as_tensor(example["context"]) for example in examples], dim=0),
                "context_mask": torch.stack([torch.as_tensor(example["context_mask"]) for example in examples], dim=0),
                "action": torch.stack([torch.as_tensor(example["action"]) for example in examples], dim=0),
            }
        elif "image" in examples[0] or "video" in examples[0]:
            image_key = "video" if "video" in examples[0] else "image"
            sample = {
                "images": [example[image_key] for example in examples],
                "prompt": [example.get("lang", example.get("prompt", "")) for example in examples],
                "action": torch.stack([torch.as_tensor(example["action"]) for example in examples], dim=0),
            }
        else:
            raise ValueError(
                "FastWAM forward requires either precomputed `input_latents/context/context_mask` "
                "or raw `image`/`video` plus `lang`/`prompt` examples."
            )

        for optional_key in ("first_frame_latents", "action_is_pad", "image_is_pad", "proprio", "state"):
            if optional_key in examples[0] and examples[0][optional_key] is not None:
                sample[optional_key] = torch.stack(
                    [torch.as_tensor(example[optional_key]) for example in examples],
                    dim=0,
                )
        return sample

    def _ensure_wan2_encoders(self) -> None:
        if self.vae is not None and self.text_encoder is not None and self.tokenizer is not None:
            return

        encoder_cfg = self.config.framework.encoder
        if not bool(encoder_cfg.get("load_wan2_encoders", False)):
            raise ValueError(
                "Raw image/video FastWAM training requires Wan2 encoder loading. "
                "Set `framework.encoder.load_wan2_encoders=true`, or provide precomputed "
                "`input_latents/context/context_mask`."
            )

        model_name = encoder_cfg.get("base_wm", None) or self.config.framework.world_model.get("base_wm")
        device, dtype = self._model_device_dtype()

        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info("Loading Wan2 VAE/text encoders for FastWAM adapter from %s", model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=dtype,
        ).to(device=device)
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device=device)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)

    @torch.no_grad()
    def _encode_text_context(self, prompt: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_wan2_encoders()
        device, dtype = self._model_device_dtype()
        if isinstance(prompt, str):
            prompt = [prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=int(self.config.framework.encoder.get("text_max_length", 512)),
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)
        text_embeds = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        ).last_hidden_state
        seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1).long()
        for i, seq_len in enumerate(seq_lens):
            text_embeds[i, seq_len:] = 0
        context_mask = torch.ones_like(text_inputs.attention_mask, dtype=torch.bool)
        return text_embeds.to(device=device, dtype=dtype), context_mask.to(device=device)

    @torch.no_grad()
    def _encode_images_to_latents(self, images: list[Any]) -> torch.Tensor:
        self._ensure_wan2_encoders()
        device, dtype = self._model_device_dtype()
        encoder_cfg = self.config.framework.encoder
        height = int(encoder_cfg.get("height", 480))
        width = int(encoder_cfg.get("width", 832))
        num_frames = encoder_cfg.get("num_frames", None)
        target_frames = None if num_frames is None else int(num_frames)

        preprocessed = []
        frame_counts = []
        for sample_images in images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            video_tensor = self.video_processor.preprocess_video(sample_images, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)
            preprocessed.append(video_tensor)
            frame_counts.append(video_tensor.shape[2])

        target_frames = target_frames if target_frames is not None else max(frame_counts)
        batch_videos = []
        for video_tensor in preprocessed:
            frame_count = video_tensor.shape[2]
            if frame_count > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
            elif frame_count < target_frames:
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - frame_count, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))

        video = torch.stack(batch_videos, dim=0)
        latents = self.vae.encode(video).latent_dist.sample()
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        return (latents - latents_mean) * latents_std

    def _build_raw_video_inputs(self, sample: dict[str, Any]) -> dict[str, Any]:
        images = sample.get("images", sample.get("video", sample.get("image")))
        prompt = sample.get("prompt", sample.get("lang", ""))
        if images is None:
            raise ValueError("Raw FastWAM adapter requires `images`, `video`, or `image`.")
        input_latents = self._encode_images_to_latents(images)
        context, context_mask = self._encode_text_context(prompt)

        converted = dict(sample)
        converted.pop("images", None)
        converted.pop("video", None)
        converted.pop("image", None)
        converted.pop("prompt", None)
        converted.pop("lang", None)
        converted["input_latents"] = input_latents
        converted["context"] = context
        converted["context_mask"] = context_mask
        return converted

    def build_inputs(self, sample: dict[str, Any] | list[dict], tiled: bool = False) -> dict[str, Any]:
        del tiled
        if isinstance(sample, list):
            sample = self._examples_to_sample(sample)
        if not isinstance(sample, dict):
            raise TypeError(f"FastWAM forward expects a dict or list[dict], got {type(sample)!r}.")

        if "input_latents" not in sample:
            if any(key in sample for key in ("images", "video", "image")):
                sample = self._build_raw_video_inputs(sample)
            else:
                raise ValueError("FastWAM training requires `input_latents`.")
        for key in ("context", "context_mask", "action"):
            if key not in sample:
                raise ValueError(f"FastWAM training requires `{key}`.")

        device, dtype = self._model_device_dtype()
        input_latents = self._as_tensor(sample["input_latents"], device=device, dtype=dtype)
        context = self._as_tensor(sample["context"], device=device, dtype=dtype)
        context_mask = self._as_tensor(sample["context_mask"], device=device, dtype=torch.bool)
        action = self._as_tensor(sample["action"], device=device, dtype=dtype)

        if input_latents.ndim != 5:
            raise ValueError(
                f"`input_latents` must be 5D [B, C, T, H, W], got shape {tuple(input_latents.shape)}"
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if action.ndim != 3:
            raise ValueError(f"`action` must be 3D [B, T, action_dim], got shape {tuple(action.shape)}")

        batch_size = input_latents.shape[0]
        if context.shape[0] != batch_size or context_mask.shape[0] != batch_size or action.shape[0] != batch_size:
            raise ValueError(
                "`input_latents`, `context`, `context_mask`, and `action` batch dimensions must match."
            )
        if context.shape[1] != context_mask.shape[1]:
            raise ValueError(
                f"`context_mask` length must match context length, got {context_mask.shape[1]} vs {context.shape[1]}"
            )

        proprio = sample.get("proprio", sample.get("state", None))
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("FastWAM training requires `proprio` or `state` when `proprio_dim` is enabled.")
            proprio = self._normalize_proprio_tensor(
                proprio,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                allow_sequence=True,
            )
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        elif proprio is not None:
            raise ValueError("`proprio`/`state` was provided but `framework.action_model.proprio_dim` is None.")

        if getattr(self.video_expert, "action_conditioned", False):
            num_latent_frames = int(input_latents.shape[2])
            if num_latent_frames <= 1:
                raise ValueError(f"FastWAM action-conditioned training needs >1 latent frame, got {num_latent_frames}.")
            if action.shape[1] % (num_latent_frames - 1) != 0:
                raise ValueError(
                    "`action` horizon must be divisible by latent transitions "
                    f"({num_latent_frames - 1}), got {action.shape[1]}."
                )

        first_frame_latents = sample.get("first_frame_latents", None)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        if first_frame_latents is None and fuse_flag:
            first_frame_latents = input_latents[:, :, 0:1]
        elif first_frame_latents is not None:
            first_frame_latents = self._as_tensor(first_frame_latents, device=device, dtype=dtype)

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = self._as_tensor(action_is_pad, device=device, dtype=torch.bool)
            if action_is_pad.shape != action.shape[:2]:
                raise ValueError(
                    f"`action_is_pad` must match action [B,T], got {tuple(action_is_pad.shape)} vs {tuple(action.shape[:2])}"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            image_is_pad = self._as_tensor(image_is_pad, device=device, dtype=torch.bool)
            if image_is_pad.shape[0] != batch_size:
                raise ValueError(f"`image_is_pad` batch size must be {batch_size}, got {image_is_pad.shape[0]}")

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        if image_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Precomputed-latent `image_is_pad` must already be aligned to latent steps; "
                f"got {tuple(image_is_pad.shape)} vs loss steps {video_loss_token.shape[1]}."
            )
        valid = (~image_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample: dict[str, Any] | list[dict], tiled: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=input_latents.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=action.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
            image_is_pad = None
            if inputs["image_is_pad"] is not None:
                image_is_pad = inputs["image_is_pad"][:, 1:]
        else:
            image_is_pad = inputs["image_is_pad"]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if inputs["action_is_pad"] is not None:
            valid = (~inputs["action_is_pad"]).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_cfg = self.config.framework.loss
        loss_total = float(loss_cfg.get("lambda_video", 1.0)) * loss_video + float(
            loss_cfg.get("lambda_action", 1.0)
        ) * loss_action
        return loss_total, {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
        }

    def forward(self, examples=None, **kwargs):
        loss, loss_dict = self.training_loss(examples, tiled=bool(kwargs.get("tiled", False)))
        return {"action_loss": loss, **loss_dict}

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _examples_to_prediction_sample(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("FastWAM predict_action received an empty examples list.")
        if "first_frame_latents" in examples[0]:
            return {
                "first_frame_latents": torch.stack(
                    [torch.as_tensor(example["first_frame_latents"]) for example in examples],
                    dim=0,
                ),
                "context": torch.stack([torch.as_tensor(example["context"]) for example in examples], dim=0),
                "context_mask": torch.stack([torch.as_tensor(example["context_mask"]) for example in examples], dim=0),
                **(
                    {
                        "proprio": torch.stack(
                            [torch.as_tensor(example.get("proprio", example.get("state"))) for example in examples],
                            dim=0,
                        )
                    }
                    if "proprio" in examples[0] or "state" in examples[0]
                    else {}
                ),
            }
        if "input_latents" in examples[0] or "latents" in examples[0]:
            latents_key = "input_latents" if "input_latents" in examples[0] else "latents"
            return {
                "first_frame_latents": torch.stack(
                    [torch.as_tensor(example[latents_key])[:, 0:1] for example in examples],
                    dim=0,
                ),
                "context": torch.stack([torch.as_tensor(example["context"]) for example in examples], dim=0),
                "context_mask": torch.stack([torch.as_tensor(example["context_mask"]) for example in examples], dim=0),
                **(
                    {
                        "proprio": torch.stack(
                            [torch.as_tensor(example.get("proprio", example.get("state"))) for example in examples],
                            dim=0,
                        )
                    }
                    if "proprio" in examples[0] or "state" in examples[0]
                    else {}
                ),
            }
        if "image" in examples[0] or "video" in examples[0]:
            image_key = "video" if "video" in examples[0] else "image"
            raw_images = []
            for example in examples:
                sample_images = example[image_key]
                if isinstance(sample_images, (list, tuple)):
                    raw_images.append([sample_images[0]])
                else:
                    raw_images.append([sample_images])
            first_frame_latents = self._encode_images_to_latents(raw_images)
            context, context_mask = self._encode_text_context(
                [example.get("lang", example.get("prompt", "")) for example in examples]
            )
            return {
                "first_frame_latents": first_frame_latents,
                "context": context,
                "context_mask": context_mask,
                **(
                    {
                        "proprio": torch.stack(
                            [torch.as_tensor(example.get("proprio", example.get("state"))) for example in examples],
                            dim=0,
                        )
                    }
                    if "proprio" in examples[0] or "state" in examples[0]
                    else {}
                ),
            }
        raise ValueError(
            "FastWAM predict_action requires `first_frame_latents/context/context_mask`, "
            "`input_latents/context/context_mask`, or raw `image`/`video` plus `lang`/`prompt`."
        )

    def _build_predict_inputs(self, examples) -> dict[str, torch.Tensor]:
        if isinstance(examples, list):
            sample = self._examples_to_prediction_sample(examples)
        elif isinstance(examples, dict):
            sample = dict(examples)
            if "first_frame_latents" not in sample:
                if "input_latents" in sample:
                    sample["first_frame_latents"] = sample["input_latents"][:, :, 0:1]
                elif any(key in sample for key in ("images", "image", "video")):
                    raw_inputs = self._build_raw_video_inputs(sample)
                    sample["first_frame_latents"] = raw_inputs["input_latents"][:, :, 0:1]
                    sample["context"] = raw_inputs["context"]
                    sample["context_mask"] = raw_inputs["context_mask"]
                else:
                    raise ValueError("FastWAM predict_action requires `first_frame_latents` or raw image/video input.")
        else:
            raise TypeError(f"FastWAM predict_action expects dict or list[dict], got {type(examples)!r}.")

        for key in ("first_frame_latents", "context", "context_mask"):
            if key not in sample:
                raise ValueError(f"FastWAM predict_action requires `{key}`.")

        device, dtype = self._model_device_dtype()
        first_frame_latents = self._as_tensor(sample["first_frame_latents"], device=device, dtype=dtype)
        context = self._as_tensor(sample["context"], device=device, dtype=dtype)
        context_mask = self._as_tensor(sample["context_mask"], device=device, dtype=torch.bool)
        if first_frame_latents.ndim != 5:
            raise ValueError(
                "`first_frame_latents` must be 5D [B,C,1,H,W] or [B,C,T,H,W], "
                f"got {tuple(first_frame_latents.shape)}."
            )
        first_frame_latents = first_frame_latents[:, :, 0:1]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != first_frame_latents.shape[0] or context_mask.shape[0] != first_frame_latents.shape[0]:
            raise ValueError("Predict input batch dimensions must match.")
        if context.shape[1] != context_mask.shape[1]:
            raise ValueError(
                f"`context_mask` length must match context length, got {context_mask.shape[1]} vs {context.shape[1]}"
            )
        proprio = sample.get("proprio", sample.get("state", None))
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("FastWAM predict_action requires `proprio` or `state` when `proprio_dim` is enabled.")
            proprio = self._normalize_proprio_tensor(
                proprio,
                batch_size=first_frame_latents.shape[0],
                device=device,
                dtype=dtype,
                allow_sequence=True,
            )
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        elif proprio is not None:
            raise ValueError("`proprio`/`state` was provided but `framework.action_model.proprio_dim` is None.")
        return {
            "first_frame_latents": first_frame_latents,
            "context": context,
            "context_mask": context_mask,
        }

    @torch.no_grad()
    def infer_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> torch.Tensor:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("FastWAM action inference requires `video_attention_mask_mode='first_frame_causal'`.")
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")

        batch_size = int(first_frame_latents.shape[0])
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=first_frame_latents.device, dtype=first_frame_latents.dtype)

        timestep_video = torch.zeros(
            (batch_size,),
            dtype=first_frame_latents.dtype,
            device=first_frame_latents.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=first_frame_latents.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype,
                device=latents_action.device,
            )
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        return latents_action.detach().to(device="cpu", dtype=torch.float32)

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        predict_inputs = self._build_predict_inputs(examples)
        action_horizon = int(
            kwargs.get(
                "action_horizon",
                self.config.framework.action_model.get("action_horizon", 1),
            )
        )
        actions = self.infer_action(
            first_frame_latents=predict_inputs["first_frame_latents"],
            context=predict_inputs["context"],
            context_mask=predict_inputs["context_mask"],
            action_horizon=action_horizon,
            num_inference_steps=int(kwargs.get("num_inference_steps", kwargs.get("num_ddim_steps", 20))),
            sigma_shift=kwargs.get("sigma_shift", None),
            seed=kwargs.get("seed", None),
            rand_device=kwargs.get("rand_device", "cpu"),
        )
        return {"normalized_actions": np.asarray(actions)}
