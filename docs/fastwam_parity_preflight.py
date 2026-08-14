"""Fail-fast preflight for the ABCI FastWAM exact-core parity job.

Run this from the StarVLA repository root after activating the same virtual
environment that the PBS job will use.  It intentionally performs no model
construction and therefore is safe on a login node.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / "FastWAM"
ASSETS = ROOT.parent / "fastwam_official_assets"
PINNED_FASTWAM_REVISION = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"

REQUIRED_IMPORTS = (
    "torch",
    "numpy",
    "omegaconf",
    "PIL",
    "transformers",
    "diffusers",
    "safetensors",
    "huggingface_hub",
    "fastwam",
)

REQUIRED_PATHS = (
    ROOT / "examples/simBenchmarks/LIBERO/eval_files/fastwam_parity/colab_orchestrator.py",
    ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
    ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json",
    ROOT / "starVLA/config/training/starvla_fastwam_libero_parity_aligned.yaml",
    ASSETS / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
)


def revision(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def main() -> None:
    failures: list[str] = []
    imported: dict[str, str] = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            imported[name] = str(getattr(module, "__version__", "available"))
        except Exception as exc:  # report all missing imports in one run
            failures.append(f"import {name}: {type(exc).__name__}: {exc}")

    for path in REQUIRED_PATHS:
        if not path.is_file():
            failures.append(f"missing file: {path}")

    official_revision = revision(OFFICIAL)
    if official_revision is None:
        failures.append(f"official FastWAM checkout is missing: {OFFICIAL}")
    elif official_revision != PINNED_FASTWAM_REVISION:
        failures.append(
            f"official FastWAM revision mismatch: expected {PINNED_FASTWAM_REVISION}, got {official_revision}"
        )

    try:
        from starVLA.model.framework.base_framework import _auto_import_framework_modules

        _auto_import_framework_modules()
    except Exception as exc:
        failures.append(f"StarVLA framework import: {type(exc).__name__}: {exc}")

    torch_info: dict[str, object] = {}
    if "torch" in imported:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available_on_login_node": torch.cuda.is_available(),
        }

    result = {
        "repo_root": str(ROOT),
        "python": sys.version.split()[0],
        "imports": imported,
        "official_fastwam_revision": official_revision,
        "torch": torch_info,
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
