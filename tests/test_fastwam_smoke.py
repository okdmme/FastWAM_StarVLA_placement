import unittest

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.WM4A.FastWAM import WanContinuousFlowMatchScheduler
from starVLA.model.framework.base_framework import build_framework


def _tiny_fastwam_cfg():
    hidden_dim = 8
    text_dim = 6
    action_dim = 3
    return OmegaConf.create(
        {
            "framework": {
                "name": "FastWAM",
                "world_model": {
                    "video_dit_config": {
                        "hidden_dim": hidden_dim,
                        "in_dim": 4,
                        "ffn_dim": 16,
                        "out_dim": 4,
                        "text_dim": text_dim,
                        "freq_dim": 8,
                        "eps": 1.0e-6,
                        "patch_size": [1, 1, 1],
                        "num_heads": 2,
                        "attn_head_dim": 4,
                        "num_layers": 1,
                        "has_image_input": False,
                        "seperated_timestep": True,
                        "fuse_vae_embedding_in_latents": True,
                        "action_conditioned": True,
                        "action_dim": action_dim,
                        "video_attention_mask_mode": "bidirectional",
                        "use_gradient_checkpointing": False,
                    },
                },
                "action_model": {
                    "action_dim": action_dim,
                    "state_dim": action_dim,
                    "action_horizon": 1,
                    "action_dit_config": {
                        "action_dim": action_dim,
                        "hidden_dim": hidden_dim,
                        "ffn_dim": 16,
                        "num_heads": 2,
                        "attn_head_dim": 4,
                        "num_layers": 1,
                        "text_dim": text_dim,
                        "freq_dim": 8,
                        "eps": 1.0e-6,
                        "use_gradient_checkpointing": False,
                    },
                },
                "mot": {
                    "mot_checkpoint_mixed_attn": False,
                },
                "scheduler": {
                    "video_train_shift": 5.0,
                    "video_infer_shift": 5.0,
                    "video_num_train_timesteps": 1000,
                    "action_train_shift": 5.0,
                    "action_infer_shift": 5.0,
                    "action_num_train_timesteps": 1000,
                },
            }
        }
    )


class FastWAMSmokeTest(unittest.TestCase):
    def test_scheduler_shapes(self):
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        timestep = scheduler.sample_training_t(2, torch.device("cpu"), torch.float32)
        x = torch.zeros(2, 4, 2, 2, 2)
        noise = torch.ones_like(x)

        noisy = scheduler.add_noise(x, noise, timestep)
        target = scheduler.training_target(x, noise, timestep)
        weight = scheduler.training_weight(timestep)

        self.assertEqual(tuple(timestep.shape), (2,))
        self.assertEqual(tuple(noisy.shape), tuple(x.shape))
        self.assertEqual(tuple(target.shape), tuple(x.shape))
        self.assertEqual(tuple(weight.shape), (2,))

    def test_build_framework_uses_tiny_fastwam_config(self):
        model = build_framework(_tiny_fastwam_cfg())

        self.assertEqual(model.__class__.__name__, "FastWAMFramework")
        self.assertEqual(len(model.video_expert.blocks), 1)
        self.assertEqual(len(model.action_expert.blocks), 1)
        self.assertIs(model.dit, model.mot)
        self.assertIs(model.train_scheduler, model.train_video_scheduler)

    def test_tiny_experts_and_mot_forward(self):
        torch.manual_seed(0)
        model = build_framework(_tiny_fastwam_cfg())
        model.eval()

        batch_size = 2
        context = torch.randn(batch_size, 5, 6)
        context_mask = torch.ones(batch_size, 5, dtype=torch.bool)
        timestep_video = torch.full((batch_size,), 250.0)
        timestep_action = torch.full((batch_size,), 500.0)
        latents = torch.randn(batch_size, 4, 2, 2, 2)
        actions = torch.randn(batch_size, 1, 3)

        video_state = model.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=actions,
            fuse_vae_embedding_in_latents=True,
        )
        action_state = model.action_expert.pre_dit(
            action_tokens=actions,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        seq_len = video_state["tokens"].shape[1] + action_state["tokens"].shape[1]
        outputs = model.mot(
            embeds_all={
                "video": video_state["tokens"],
                "action": action_state["tokens"],
            },
            attention_mask=torch.ones(seq_len, seq_len, dtype=torch.bool),
            freqs_all={
                "video": video_state["freqs"],
                "action": action_state["freqs"],
            },
            context_all={
                "video": {
                    "context": video_state["context"],
                    "mask": video_state["context_mask"],
                },
                "action": {
                    "context": action_state["context"],
                    "mask": action_state["context_mask"],
                },
            },
            t_mod_all={
                "video": video_state["t_mod"],
                "action": action_state["t_mod"],
            },
        )

        self.assertEqual(tuple(outputs["video"].shape), tuple(video_state["tokens"].shape))
        self.assertEqual(tuple(outputs["action"].shape), tuple(action_state["tokens"].shape))

    def test_forward_returns_action_loss_for_precomputed_latents(self):
        torch.manual_seed(0)
        model = build_framework(_tiny_fastwam_cfg())
        model.train()

        batch = {
            "input_latents": torch.randn(2, 4, 2, 2, 2),
            "context": torch.randn(2, 5, 6),
            "context_mask": torch.ones(2, 5, dtype=torch.bool),
            "action": torch.randn(2, 1, 3),
            "image_is_pad": torch.zeros(2, 2, dtype=torch.bool),
            "action_is_pad": torch.zeros(2, 1, dtype=torch.bool),
        }

        out = model(batch)

        self.assertIn("action_loss", out)
        self.assertIn("loss_video", out)
        self.assertIn("loss_action", out)
        self.assertEqual(out["action_loss"].ndim, 0)
        self.assertTrue(torch.isfinite(out["action_loss"]))

        out["action_loss"].backward()
        grad_params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(grad_params)

    def test_compute_loss_routes_vla_batch(self):
        torch.manual_seed(0)
        model = build_framework(_tiny_fastwam_cfg())
        batch = {
            "input_latents": torch.randn(2, 4, 2, 2, 2),
            "context": torch.randn(2, 5, 6),
            "context_mask": torch.ones(2, 5, dtype=torch.bool),
            "action": torch.randn(2, 1, 3),
        }

        out = model.compute_loss("vla", batch)

        self.assertIn("action_loss", out)
        self.assertEqual(out["action_loss"].ndim, 0)
        self.assertTrue(torch.isfinite(out["action_loss"]))

    def test_raw_examples_route_through_encoder_adapter(self):
        torch.manual_seed(0)
        model = build_framework(_tiny_fastwam_cfg())
        model.train()

        model._encode_images_to_latents = lambda images: torch.randn(2, 4, 2, 2, 2)
        model._encode_text_context = lambda prompt: (
            torch.randn(2, 5, 6),
            torch.ones(2, 5, dtype=torch.bool),
        )
        examples = [
            {"image": ["frame0", "frame1"], "lang": "pick", "action": torch.randn(1, 3)},
            {"image": ["frame0", "frame1"], "lang": "place", "action": torch.randn(1, 3)},
        ]

        out = model(examples)

        self.assertIn("action_loss", out)
        self.assertEqual(out["action_loss"].ndim, 0)
        self.assertTrue(torch.isfinite(out["action_loss"]))

    def test_raw_examples_require_encoder_loading_or_precomputed_latents(self):
        model = build_framework(_tiny_fastwam_cfg())

        with self.assertRaisesRegex(ValueError, "load_wan2_encoders=true"):
            model(
                [
                    {"image": ["frame0", "frame1"], "lang": "pick", "action": torch.randn(1, 3)},
                    {"image": ["frame0", "frame1"], "lang": "place", "action": torch.randn(1, 3)},
                ]
            )


if __name__ == "__main__":
    unittest.main()
