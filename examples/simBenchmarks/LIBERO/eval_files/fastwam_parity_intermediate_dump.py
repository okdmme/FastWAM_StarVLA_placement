#!/usr/bin/env python
"""ABCI smoke runner for StarVLA FastWAM official checkpoints.

This script intentionally separates model construction, dtype/device placement,
checkpoint loading, optional Wan2 encoder loading, and action prediction. The
StarVLA FastWAM framework can auto-load these from config, but doing it here
keeps ABCI memory behavior explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.framework.base_framework import build_framework


def _resolve_repo_path(path: str | None, root: Path) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((root / path_obj).resolve())


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _parse_state(text: str | None, dim: int) -> torch.Tensor:
    if text is None or text.strip() == "":
        return torch.zeros(1, dim, dtype=torch.float32)
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != dim:
        raise ValueError(f"--state must contain {dim} comma-separated values, got {len(values)}.")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def _load_image(path: str | None, height: int, width: int) -> Image.Image:
    if path is None:
        return Image.new("RGB", (width, height), color=(127, 127, 127))
    return Image.open(path).convert("RGB")


def _config_for_manual_load(cfg: Any, root: Path, mode: str) -> tuple[Any, str]:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    checkpoint_path = _resolve_repo_path(cfg.framework.checkpoint.path, root)
    if checkpoint_path is None:
        raise ValueError("Config must set framework.checkpoint.path.")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    cfg.framework.checkpoint.path = None
    cfg.framework.encoder.load_wan2_encoders = mode == "raw"
    cfg.framework.encoder.base_wm = _resolve_repo_path(cfg.framework.encoder.base_wm, root)
    cfg.framework.world_model.base_wm = _resolve_repo_path(cfg.framework.world_model.base_wm, root)
    return cfg, checkpoint_path


def _checkpoint_load_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "loaded": report.get("loaded", []),
        "step": report.get("step"),
        "proprio_dim": report.get("proprio_dim"),
        "missing_keys_count": len(report.get("missing_keys", [])),
        "unexpected_keys_count": len(report.get("unexpected_keys", [])),
        "missing_keys_sample": report.get("missing_keys", [])[:20],
        "unexpected_keys_sample": report.get("unexpected_keys", [])[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="StarVLA FastWAM official inference config.")
    parser.add_argument("--mode", choices=("load", "latent", "raw"), default="load")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--latent-spatial-downsample", type=int, default=16)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image", default=None, help="Optional raw RGB image path for --mode raw.")
    parser.add_argument("--state", default=None, help="Comma-separated proprio/state values. Defaults to zeros.")
    parser.add_argument("--save-actions", default=None)
    args = parser.parse_args()

    root = _repo_root()
    cfg = OmegaConf.load(_resolve_repo_path(args.config, root))
    cfg, checkpoint_path = _config_for_manual_load(cfg, root, args.mode)

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)

    print(json.dumps({"repo_root": str(root), "mode": args.mode, "device": str(device), "dtype": args.dtype}, indent=2))
    model = build_framework(cfg)
    model.eval()
    model.to(device=device, dtype=dtype)

    report = model.load_checkpoint(checkpoint_path, strict=bool(cfg.framework.checkpoint.get("strict", False)))
    print(json.dumps({"checkpoint": checkpoint_path, "load_report": _checkpoint_load_report(report)}, indent=2))

    if args.mode == "load":
        return

    action_horizon = int(cfg.framework.action_model.action_horizon)
    action_dim = int(cfg.framework.action_model.action_dim)
    proprio_dim = int(model.proprio_dim)
    state = _parse_state(args.state, proprio_dim).repeat(args.batch_size, 1)

    if args.mode == "latent":
        height = int(cfg.framework.encoder.height)
        width = int(cfg.framework.encoder.width)
        latent_h = height // int(args.latent_spatial_downsample)
        latent_w = width // int(args.latent_spatial_downsample)
        first_frame_latents = torch.randn(args.batch_size, 48, 1, latent_h, latent_w, dtype=torch.float32)
        context = torch.randn(args.batch_size, args.context_length, 4096, dtype=torch.float32)
        context_mask = torch.ones(args.batch_size, args.context_length, dtype=torch.bool)
        examples: dict[str, Any] = {
            "first_frame_latents": first_frame_latents,
            "context": context,
            "context_mask": context_mask,
            "proprio": state,
        }
    else:
        model._ensure_wan2_encoders()
        image = _load_image(args.image, int(cfg.framework.encoder.height), int(cfg.framework.encoder.width))
        examples = {
            "image": [image for _ in range(args.batch_size)],
            "prompt": [args.prompt for _ in range(args.batch_size)],
            "proprio": state,
        }


    print("\n===== DUMP STARVLA INTERMEDIATES =====")

    if args.mode != "raw":
        raise RuntimeError("Intermediate parity dump requires --mode raw.")

    # Exactly one raw VAE/T5 encoding pass.
    raw_inputs = model._build_raw_video_inputs(examples)

    first_frame_latents = raw_inputs["input_latents"][:, :, 0:1]
    text_context = raw_inputs["context"]
    text_context_mask = raw_inputs["context_mask"]

    device, dtype = model._model_device_dtype()

    # `state` is already the official-normalized proprio passed via --state.
    proprio = state.to(device=device, dtype=dtype)

    context_with_proprio, context_mask_with_proprio = (
        model._append_proprio_to_context(
            context=text_context,
            context_mask=text_context_mask,
            proprio=proprio,
        )
    )

    dump_dir = (
        root
        / "playground/fastwam_parity"
        / "libero_goal_task0_episode0_seed7"
        / "starvla_parity_aligned_seed7_steps10"
        / "intermediates"
    )
    dump_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        dump_dir / "first_frame_latents.npy",
        first_frame_latents.detach().cpu().float().numpy(),
    )
    np.save(
        dump_dir / "text_context.npy",
        text_context.detach().cpu().float().numpy(),
    )
    np.save(
        dump_dir / "text_context_mask.npy",
        text_context_mask.detach().cpu().numpy(),
    )
    np.save(
        dump_dir / "context_with_proprio.npy",
        context_with_proprio.detach().cpu().float().numpy(),
    )
    np.save(
        dump_dir / "context_mask_with_proprio.npy",
        context_mask_with_proprio.detach().cpu().numpy(),
    )

    print(
        "first_frame_latents:",
        tuple(first_frame_latents.shape),
        first_frame_latents.dtype,
    )
    print(
        "text_context:",
        tuple(text_context.shape),
        text_context.dtype,
    )
    print(
        "text_context_mask:",
        tuple(text_context_mask.shape),
        text_context_mask.dtype,
    )
    print(
        "context_with_proprio:",
        tuple(context_with_proprio.shape),
        context_with_proprio.dtype,
    )
    print("saved:", dump_dir)

    return

    out = model.predict_action(
        examples,
        action_horizon=action_horizon,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device="cpu",
    )
    actions = np.asarray(out["normalized_actions"])

    if args.save_actions:
        save_path = Path(args.save_actions)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, actions)
        print(f"saved_actions: {save_path}")
    summary = {
        "actions_shape": list(actions.shape),
        "expected_shape": [args.batch_size, action_horizon, action_dim],
        "finite": bool(np.isfinite(actions).all()),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
    }
    print(json.dumps(summary, indent=2))
    if summary["actions_shape"] != summary["expected_shape"]:
        raise RuntimeError(f"Unexpected action shape: {summary}")
    if not summary["finite"]:
        raise RuntimeError("Predicted actions contain non-finite values.")


if __name__ == "__main__":
    main()
