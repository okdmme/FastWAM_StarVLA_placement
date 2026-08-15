from __future__ import annotations

import os
import platform
from typing import Any

import torch


def configure_reference_mode(seed: int = 7) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    # A warning is not sufficient for a bitwise-parity claim.  Fail instead of
    # silently selecting a known non-deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    return {
        "seed": seed,
        "tf32": False,
        "torch_compile": False,
        "flash_attention": False,
        "deterministic_algorithms_warn_only": False,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_vram_bytes": detect_cuda_vram_bytes(),
    }


def detect_cuda_vram_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).total_memory)


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
