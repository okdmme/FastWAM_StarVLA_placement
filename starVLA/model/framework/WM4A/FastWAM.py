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
from typing import Optional

import torch

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

    def forward(self, examples=None, **kwargs):
        raise NotImplementedError(
            "FastWAM framework components are placed, but training_loss/sample adaptation "
            "has not been ported yet. Next step: port FastWAM.training_loss() into this forward()."
        )

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        raise NotImplementedError(
            "FastWAM predict_action is not ported yet. Next step: port FastWAM.infer_action()."
        )
