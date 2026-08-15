from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .tensor_compare import TensorRecord, save_tensor_record


class TraceWriter:
    def __init__(self, output_dir: str | Path, prefix: str):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.records: list[TensorRecord] = []

    def tensor(self, stage: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        self.records.append(save_tensor_record(f"{self.prefix}/{stage}", tensor, self.output_dir))


def build_shared_action_inputs(
    *,
    action_shape: tuple[int, int, int],
    scheduler: Any,
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    rand_device: str = "cpu",
    sigma_shift: float | None = None,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=rand_device).manual_seed(seed)
    initial_action_noise = torch.randn(
        action_shape,
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=int(num_inference_steps),
        device=device,
        dtype=dtype,
        shift_override=sigma_shift,
    )
    return {
        "initial_action_noise": initial_action_noise.detach().cpu(),
        "infer_timesteps_action": timesteps.detach().cpu(),
        "infer_deltas_action": deltas.detach().cpu(),
    }


@torch.no_grad()
def prefill_video_cache_with_trace(
    model: Any,
    *,
    video_pre: dict[str, Any],
    attention_mask: torch.Tensor,
    video_seq_len: int,
    trace: TraceWriter,
) -> list[dict[str, torch.Tensor]]:
    mot = model.mot
    expert = mot.mixtures["video"]
    x = video_pre["tokens"]
    kv_cache: list[dict[str, torch.Tensor]] = []
    video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
    for layer_idx in range(mot.num_layers):
        block = expert.blocks[layer_idx]
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = mot._build_expert_attention_io(
            expert=expert,
            block=block,
            x=x,
            freqs=video_pre["freqs"],
            t_mod=video_pre["t_mod"],
        )
        mixed = mot._mixed_attention(q_cat=q, k_cat=k, v_cat=v, attention_mask=video_attention_mask)
        x = mot._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gc,
            mixed_slice=mixed,
            context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
        )
        trace.tensor(f"07_video_dit_block_output/layer_{layer_idx:02d}", x)
        trace.tensor(f"08_kv_cache/layer_{layer_idx:02d}/k", k)
        trace.tensor(f"08_kv_cache/layer_{layer_idx:02d}/v", v)
        kv_cache.append({"k": k, "v": v})
    return kv_cache


@torch.no_grad()
def action_with_cache_with_trace(
    model: Any,
    *,
    latents_action: torch.Tensor,
    timestep_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_kv_cache: list[dict[str, torch.Tensor]],
    attention_mask: torch.Tensor,
    video_seq_len: int,
    step_idx: int,
    trace: TraceWriter,
) -> torch.Tensor:
    action_pre = model.action_expert.pre_dit(
        action_tokens=latents_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    trace.tensor(f"06_action_encoder_output/step_{step_idx:02d}", action_pre["tokens"])

    mot = model.mot
    expert = mot.mixtures["action"]
    x = action_pre["tokens"]
    total_seq_len = int(video_seq_len) + int(x.shape[1])
    action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
    for layer_idx in range(mot.num_layers):
        block = expert.blocks[layer_idx]
        q_action, k_action, v_action, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = (
            mot._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
            )
        )
        layer_cache = video_kv_cache[layer_idx]
        k_cat = torch.cat([layer_cache["k"], k_action], dim=1)
        v_cat = torch.cat([layer_cache["v"], v_action], dim=1)
        mixed = mot._mixed_attention(q_cat=q_action, k_cat=k_cat, v_cat=v_cat, attention_mask=action_attention_mask)
        x = mot._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gc,
            mixed_slice=mixed,
            context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
        )
        trace.tensor(f"09_action_mot_block_output/step_{step_idx:02d}/layer_{layer_idx:02d}", x)
    return model.action_expert.post_dit(x, action_pre)


@torch.no_grad()
def trace_action_inference(
    model: Any,
    *,
    first_frame_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    initial_action_noise: torch.Tensor,
    infer_timesteps_action: torch.Tensor,
    infer_deltas_action: torch.Tensor,
    trace: TraceWriter,
) -> torch.Tensor:
    model.eval()
    device = first_frame_latents.device
    dtype = first_frame_latents.dtype
    latents_action = initial_action_noise.to(device=device, dtype=dtype).clone()
    trace.tensor("06_initial_action_noise", latents_action)

    timestep_video = torch.zeros((first_frame_latents.shape[0],), dtype=dtype, device=device)
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_horizon,
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_kv_cache = prefill_video_cache_with_trace(
        model,
        video_pre=video_pre,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
        trace=trace,
    )

    for step_idx, (step_t_action, step_delta_action) in enumerate(
        zip(infer_timesteps_action, infer_deltas_action)
    ):
        timestep_action = step_t_action.reshape(1).expand(first_frame_latents.shape[0]).to(
            device=device,
            dtype=latents_action.dtype,
        )
        pred_action = action_with_cache_with_trace(
            model,
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            step_idx=step_idx,
            trace=trace,
        )
        trace.tensor(f"10_predicted_velocity/step_{step_idx:02d}", pred_action)
        latents_action = model.infer_action_scheduler.step(
            pred_action,
            step_delta_action.to(device=device, dtype=latents_action.dtype),
            latents_action,
        )
        trace.tensor(f"11_scheduler_output/step_{step_idx:02d}", latents_action)

    normalized = latents_action.detach().to(device="cpu", dtype=torch.float32)
    if normalized.shape[0] == 1:
        trace.tensor("12_final_normalized_actions", normalized[0])
        return normalized[0]
    trace.tensor("12_final_normalized_actions", normalized)
    return normalized

