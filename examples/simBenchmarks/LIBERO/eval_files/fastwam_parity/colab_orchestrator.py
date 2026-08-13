from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from .determinism import configure_reference_mode, release_cuda_memory
from .stats import deterministic_anchor_state, load_dataset_stats, normalize_minmax, unnormalize_minmax
from .tensor_compare import compare_tensors, save_tensor_record, write_manifest
from .trace import TraceWriter, build_shared_action_inputs, trace_action_inference


STARVLA_REPO = "https://github.com/okdmme/FastWAM_StarVLA_placement.git"
STARVLA_COMMIT = "374607be1e26d700f36b8a6f3f9fd30c018af79e"
OFFICIAL_REPO = "https://github.com/yuantianyuan01/FastWAM.git"
OFFICIAL_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
OFFICIAL_ASSET_REPO = "SereneC/wan-series-checkpoint"
OFFICIAL_ASSET_REVISION = "fec1e03"
RUNTIME_DIR = Path("/content/fastwam_parity")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def make_anchor_image(height: int, width: int) -> np.ndarray:
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((x + y) % 256).astype(np.uint8)
    image[..., 1] = ((2 * x + y) % 256).astype(np.uint8)
    image[..., 2] = ((x + 2 * y) % 256).astype(np.uint8)
    return image


def _this_starvla_root() -> Path:
    return Path(__file__).resolve().parents[5]


def ensure_repos(runtime_dir: Path, starvla_dir_arg: str | None = None) -> tuple[Path, Path]:
    starvla_dir = Path(starvla_dir_arg).resolve() if starvla_dir_arg else _this_starvla_root()
    official_dir = runtime_dir / "FastWAM_official"
    if not (starvla_dir / ".git").exists():
        raise FileNotFoundError(f"StarVLA source must be a git checkout: {starvla_dir}")

    if not official_dir.exists():
        run(["git", "clone", OFFICIAL_REPO, str(official_dir)])
    run(["git", "fetch", "origin", OFFICIAL_COMMIT], cwd=official_dir)
    run(["git", "checkout", "--detach", OFFICIAL_COMMIT], cwd=official_dir)
    return starvla_dir, official_dir


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_starvla_model(starvla_dir: Path, cfg: Any, checkpoint_path: Path, device: str, dtype: torch.dtype):
    sys.path.insert(0, str(starvla_dir))
    from starVLA.model.framework.base_framework import build_framework

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.framework.checkpoint.path = None
    cfg.framework.encoder.load_wan2_encoders = True
    model = build_framework(cfg)
    model.eval().to(device=torch.device(device), dtype=dtype)
    model.load_checkpoint(str(checkpoint_path), strict=False)
    model._ensure_wan2_encoders()
    return model


def _load_official_model(
    official_dir: Path,
    cfg: Any,
    checkpoint_path: Path,
    official_assets_dir: Path,
    tokenizer_dir: Path,
    official_model_id: str,
    official_tokenizer_model_id: str,
    redirect_common_files: bool,
    device: str,
    dtype: torch.dtype,
):
    sys.path.insert(0, str(official_dir / "src"))
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.helpers import loader as official_loader

    vae_path = official_assets_dir / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    text_path = official_assets_dir / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    if not vae_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(
            "Pinned official encoder assets are missing. Expected:\n"
            f"  {vae_path}\n  {text_path}\n"
            "Run the notebook asset-download cell before the inference cell."
        )
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Official tokenizer directory is missing: {tokenizer_dir}")

    # The pinned official loader hard-codes an obsolete DiffSynth repository.
    # Supply the same converted files from a pinned public mirror as local
    # paths, without changing the official model implementation.
    original_resolve_configs = official_loader._resolve_configs

    def resolve_local_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
        configs = original_resolve_configs(model_id, tokenizer_model_id, redirect_common_files=False)
        dit_config, text_config, vae_config, tokenizer_config = configs
        text_config.path = str(text_path)
        text_config.model_id = None
        vae_config.path = str(vae_path)
        vae_config.model_id = None
        tokenizer_config.path = str(tokenizer_dir)
        tokenizer_config.model_id = None
        return dit_config, text_config, vae_config, tokenizer_config

    official_loader._resolve_configs = resolve_local_configs

    try:
        framework = cfg.framework
        model = FastWAM.from_wan22_pretrained(
            device=device,
            torch_dtype=dtype,
            model_id=official_model_id,
            tokenizer_model_id=official_tokenizer_model_id,
            tokenizer_max_len=int(framework.encoder.text_max_length),
            load_text_encoder=True,
            proprio_dim=int(framework.action_model.proprio_dim),
            redirect_common_files=redirect_common_files,
            video_dit_config=OmegaConf.to_container(framework.world_model.video_dit_config, resolve=True),
            action_dit_config=OmegaConf.to_container(framework.action_model.action_dit_config, resolve=True),
            skip_dit_load_from_pretrain=True,
            mot_checkpoint_mixed_attn=False,
            video_train_shift=float(framework.scheduler.video_train_shift),
            video_infer_shift=float(framework.scheduler.video_infer_shift),
            video_num_train_timesteps=int(framework.scheduler.video_num_train_timesteps),
            action_train_shift=float(framework.scheduler.action_train_shift),
            action_infer_shift=float(framework.scheduler.action_infer_shift),
            action_num_train_timesteps=int(framework.scheduler.action_num_train_timesteps),
        )
    finally:
        official_loader._resolve_configs = original_resolve_configs
    model.load_checkpoint(str(checkpoint_path))
    model.eval().to(device=torch.device(device), dtype=dtype)
    return model


def _official_prepare(model: Any, image: np.ndarray, prompt: str, state: torch.Tensor, trace: TraceWriter):
    x = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 127.5 - 1.0
    # The official VAE is loaded in the reference dtype (bfloat16 in the
    # Colab run). Match its input dtype/device before entering Conv3d.
    x = x.to(device=model.device, dtype=model.torch_dtype)
    trace.tensor("02_preprocessing/official_image_chw_minus1_1", x)
    first_frame_latents = model._encode_input_image_latents_tensor(x)
    context, context_mask = model.encode_prompt(prompt)
    normalized_state = state.to(device=model.device, dtype=model.torch_dtype)
    proprio_token = model.proprio_encoder(normalized_state.unsqueeze(0).to(dtype=context.dtype)).to(dtype=context.dtype)
    context_with_state, mask_with_state = model._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=normalized_state.unsqueeze(0),
    )
    trace.tensor("03_prompt/token_context", context)
    trace.tensor("03_prompt/context_mask", context_mask)
    trace.tensor("04_vae_latent", first_frame_latents)
    trace.tensor("05_normalized_state", normalized_state)
    trace.tensor("05_state_encoder_output", proprio_token)
    return first_frame_latents, context_with_state, mask_with_state


def _starvla_prepare(model: Any, image: np.ndarray, prompt: str, state: torch.Tensor, trace: TraceWriter):
    pil = Image.fromarray(image)
    trace.tensor("02_preprocessing/starvla_image_uint8_hwc", torch.from_numpy(image))
    sample = {"image": [pil], "prompt": [prompt], "proprio": state.unsqueeze(0)}
    raw_inputs = model._build_raw_video_inputs(sample)
    first_frame_latents = raw_inputs["input_latents"][:, :, 0:1]
    context = raw_inputs["context"]
    context_mask = raw_inputs["context_mask"]
    device, dtype = model._model_device_dtype()
    normalized_state = state.to(device=device, dtype=dtype)
    proprio_token = model.proprio_encoder(normalized_state.unsqueeze(0).unsqueeze(1)).to(dtype=context.dtype)
    context_with_state, mask_with_state = model._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=normalized_state.unsqueeze(0),
    )
    trace.tensor("03_prompt/token_context", context)
    trace.tensor("03_prompt/context_mask", context_mask)
    trace.tensor("04_vae_latent", first_frame_latents)
    trace.tensor("05_normalized_state", normalized_state)
    trace.tensor("05_state_encoder_output", proprio_token)
    return first_frame_latents, context_with_state, mask_with_state


def compare_manifests(official_records, starvla_records, output_path: Path) -> list[dict[str, Any]]:
    by_stage = {record.stage.split("/", 1)[1]: record for record in official_records}
    results = []
    for star_record in starvla_records:
        stage = star_record.stage.split("/", 1)[1]
        official_record = by_stage.get(stage)
        if official_record is None:
            continue
        official_tensor = torch.load(official_record.path, map_location="cpu")
        starvla_tensor = torch.load(star_record.path, map_location="cpu")
        row = {"stage": stage, **compare_tensors(official_tensor, starvla_tensor)}
        results.append(row)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", default=str(RUNTIME_DIR))
    parser.add_argument("--starvla-dir", default=None)
    parser.add_argument("--config", default="starVLA/config/training/starvla_fastwam_libero_parity_aligned.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/fastwam_release/libero_uncond_2cam224.pt")
    parser.add_argument("--stats", default="checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
    parser.add_argument("--wan-model", default="playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--official-model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--official-tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--official-assets-dir", default=None)
    parser.add_argument("--official-redirect-common-files", action="store_true", default=True)
    parser.add_argument(
        "--download-source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Hub used by the pinned official loader for missing Wan components.",
    )
    parser.add_argument("--prompt", default="pick up the object and place it into the target area")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # The official loader defaults to ModelScope. Pin the source explicitly so
    # a fresh Colab does not fail before the first trace is written.
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = args.download_source
    metadata = configure_reference_mode(seed=args.seed)
    metadata.update(
        {
            "starvla_start_revision": STARVLA_COMMIT,
            "official_revision": OFFICIAL_COMMIT,
            "official_encoder_asset_repo": OFFICIAL_ASSET_REPO,
            "official_encoder_asset_revision": OFFICIAL_ASSET_REVISION,
            "official_encoder_asset_mode": "pinned_local_mirror",
        }
    )
    starvla_dir, official_dir = ensure_repos(runtime_dir, args.starvla_dir)

    cfg_path = starvla_dir / args.config
    cfg = OmegaConf.load(cfg_path)
    checkpoint_path = starvla_dir / args.checkpoint
    stats_path = starvla_dir / args.stats
    official_assets_dir = Path(args.official_assets_dir) if args.official_assets_dir else runtime_dir / "official_assets"
    tokenizer_dir = starvla_dir / args.wan_model / "tokenizer"
    dtype = _dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_dataset_stats(stats_path)
    height = int(cfg.framework.encoder.height)
    width = int(cfg.framework.encoder.width)
    image = make_anchor_image(height, width)
    raw_state = deterministic_anchor_state(stats)
    normalized_state = normalize_minmax(raw_state, stats, "state")

    anchor_dir = runtime_dir / "anchor"
    anchor_dir.mkdir(exist_ok=True)
    Image.fromarray(image).save(anchor_dir / "anchor.png")
    (anchor_dir / "metadata.json").write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "raw_state": raw_state.tolist(),
                "normalized_state": normalized_state.tolist(),
                "height": height,
                "width": width,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    action_horizon = int(cfg.framework.action_model.action_horizon)
    action_dim = int(cfg.framework.action_model.action_dim)

    official_trace = TraceWriter(runtime_dir / "tensors", "official")
    official_trace.records.append(save_tensor_record("official/01_raw_anchor/rgb_uint8_hwc", torch.from_numpy(image), runtime_dir / "tensors"))
    official_trace.records.append(save_tensor_record("official/01_raw_anchor/normalized_state", normalized_state, runtime_dir / "tensors"))
    official_model = _load_official_model(
        official_dir,
        cfg,
        checkpoint_path,
        official_assets_dir,
        tokenizer_dir,
        args.official_model_id,
        args.official_tokenizer_model_id,
        bool(args.official_redirect_common_files),
        device,
        dtype,
    )
    shared = build_shared_action_inputs(
        action_shape=(1, action_horizon, action_dim),
        scheduler=official_model.infer_action_scheduler,
        num_inference_steps=args.num_inference_steps,
        device=torch.device(device),
        dtype=dtype,
        seed=args.seed,
    )
    first_frame, context, context_mask = _official_prepare(
        official_model,
        image,
        args.prompt,
        normalized_state,
        official_trace,
    )
    official_actions = trace_action_inference(
        official_model,
        first_frame_latents=first_frame,
        context=context,
        context_mask=context_mask,
        action_horizon=action_horizon,
        initial_action_noise=shared["initial_action_noise"],
        infer_timesteps_action=shared["infer_timesteps_action"],
        infer_deltas_action=shared["infer_deltas_action"],
        trace=official_trace,
    )
    official_unnorm = unnormalize_minmax(official_actions, stats, "action")
    official_trace.tensor("13_unnormalized_actions", official_unnorm)
    write_manifest(official_trace.records, runtime_dir / "official_manifest.json", metadata)
    del official_model
    gc.collect()
    release_cuda_memory()

    star_trace = TraceWriter(runtime_dir / "tensors", "starvla")
    star_trace.records.append(save_tensor_record("starvla/01_raw_anchor/rgb_uint8_hwc", torch.from_numpy(image), runtime_dir / "tensors"))
    star_trace.records.append(save_tensor_record("starvla/01_raw_anchor/normalized_state", normalized_state, runtime_dir / "tensors"))
    star_model = _load_starvla_model(starvla_dir, cfg, checkpoint_path, device, dtype)
    first_frame, context, context_mask = _starvla_prepare(star_model, image, args.prompt, normalized_state, star_trace)
    star_actions = trace_action_inference(
        star_model,
        first_frame_latents=first_frame,
        context=context,
        context_mask=context_mask,
        action_horizon=action_horizon,
        initial_action_noise=shared["initial_action_noise"],
        infer_timesteps_action=shared["infer_timesteps_action"],
        infer_deltas_action=shared["infer_deltas_action"],
        trace=star_trace,
    )
    star_unnorm = unnormalize_minmax(star_actions, stats, "action")
    star_trace.tensor("13_unnormalized_actions", star_unnorm)
    write_manifest(star_trace.records, runtime_dir / "starvla_manifest.json", metadata)
    del star_model
    gc.collect()
    release_cuda_memory()

    results = compare_manifests(official_trace.records, star_trace.records, runtime_dir / "comparison.json")
    shutil.make_archive(str(runtime_dir / "fastwam_parity_results"), "zip", runtime_dir)
    print(json.dumps({"runtime_dir": str(runtime_dir), "num_compared_stages": len(results)}, indent=2))


if __name__ == "__main__":
    main()
