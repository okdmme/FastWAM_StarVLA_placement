# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
FastWAM Framework.

This is the StarVLA framework entry for the full FastWAM architecture:

    FastWAM_WanVideoDiT + FastWAM_ActionDiT + FastWAM_MoT

The file is intentionally separate from WanFastWAM.py. WanFastWAM.py remains the
earlier thin Wan2-hidden-states-to-ActionDiT prototype, while this module is the
landing point for the paper-style FastWAM joint video/action training path.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from starVLA.model.framework.WM4A.FastWAM_MoT import MoT
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.FastWAM_ActionDiT import ActionDiT
from starVLA.model.modules.world_model.FastWAM_WanVideoDiT import WanVideoDiT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


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

        self._build_experts()
        self._build_schedulers()

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

    def _examples_to_precomputed_sample(self, examples: list[dict]) -> dict[str, Any]:
        if not examples:
            raise ValueError("FastWAM forward received an empty examples list.")
        if "input_latents" not in examples[0] and "latents" not in examples[0]:
            raise ValueError(
                "FastWAM full training currently requires precomputed `input_latents` in examples. "
                "Image/video-to-latent adapter is the next integration step."
            )

        latents_key = "input_latents" if "input_latents" in examples[0] else "latents"
        sample = {
            "input_latents": torch.stack([torch.as_tensor(example[latents_key]) for example in examples], dim=0),
            "context": torch.stack([torch.as_tensor(example["context"]) for example in examples], dim=0),
            "context_mask": torch.stack([torch.as_tensor(example["context_mask"]) for example in examples], dim=0),
            "action": torch.stack([torch.as_tensor(example["action"]) for example in examples], dim=0),
        }
        for optional_key in ("first_frame_latents", "action_is_pad", "image_is_pad"):
            if optional_key in examples[0] and examples[0][optional_key] is not None:
                sample[optional_key] = torch.stack(
                    [torch.as_tensor(example[optional_key]) for example in examples],
                    dim=0,
                )
        return sample

    def build_inputs(self, sample: dict[str, Any] | list[dict], tiled: bool = False) -> dict[str, Any]:
        del tiled
        if isinstance(sample, list):
            sample = self._examples_to_precomputed_sample(sample)
        if not isinstance(sample, dict):
            raise TypeError(f"FastWAM forward expects a dict or list[dict], got {type(sample)!r}.")

        if "input_latents" not in sample:
            if "video" in sample:
                raise NotImplementedError(
                    "FastWAM image/video-to-latent training adapter is not ported yet. "
                    "Provide precomputed `input_latents`, `context`, `context_mask`, and `action`."
                )
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

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        raise NotImplementedError(
            "FastWAM predict_action is not ported yet. Next step: port FastWAM.infer_action()."
        )
