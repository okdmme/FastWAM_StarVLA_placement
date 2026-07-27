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

        self._build_experts()

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
