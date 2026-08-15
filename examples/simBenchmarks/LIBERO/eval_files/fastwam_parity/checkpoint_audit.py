"""Exact, per-tensor audit for an official FastWAM checkpoint load.

This intentionally compares the checkpoint tensors with the tensors resident in
the constructed model, rather than relying only on ``load_state_dict``'s
missing/unexpected-key report.  The audit is designed for a large-memory ABCI
node: it transfers at most one target tensor to CPU at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .tensor_compare import tensor_sha256


def _tensor_count(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel()) for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))


def _audit_state_dict(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source_keys = set(source)
    target_keys = set(target)
    common_keys = sorted(source_keys & target_keys)
    mismatches: list[dict[str, Any]] = []
    matched_tensor_count = 0
    matched_parameter_count = 0

    for key in common_keys:
        source_tensor = source[key]
        target_tensor = target[key]
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(target_tensor, torch.Tensor):
            mismatches.append({"key": key, "reason": "non_tensor_value"})
            continue

        target_cpu = target_tensor.detach().cpu()
        same_shape = tuple(source_tensor.shape) == tuple(target_cpu.shape)
        same_dtype = source_tensor.dtype == target_cpu.dtype
        equal = bool(same_shape and same_dtype and torch.equal(source_tensor, target_cpu))
        if equal:
            matched_tensor_count += 1
            matched_parameter_count += int(source_tensor.numel())
            continue
        mismatches.append(
            {
                "key": key,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_cpu.shape),
                "source_dtype": str(source_tensor.dtype),
                "target_dtype": str(target_cpu.dtype),
                "source_sha256": tensor_sha256(source_tensor),
                "target_sha256": tensor_sha256(target_cpu),
            }
        )

    source_tensor_count = sum(isinstance(value, torch.Tensor) for value in source.values())
    source_parameter_count = _tensor_count(source)
    return {
        "source_tensor_count": source_tensor_count,
        "source_parameter_count": source_parameter_count,
        "target_tensor_count": sum(isinstance(value, torch.Tensor) for value in target.values()),
        "matched_tensor_count": matched_tensor_count,
        "matched_parameter_count": matched_parameter_count,
        "parameter_coverage_ratio": (
            float(matched_parameter_count / source_parameter_count) if source_parameter_count else 1.0
        ),
        "missing_target_keys": sorted(source_keys - target_keys),
        "unexpected_target_keys": sorted(target_keys - source_keys),
        "mismatches": mismatches,
        "exact": (
            not (source_keys - target_keys)
            and not (target_keys - source_keys)
            and not mismatches
            and matched_tensor_count == source_tensor_count
        ),
    }


def audit_loaded_checkpoint(checkpoint_path: str | Path, model: Any) -> dict[str, Any]:
    """Return an exact load audit for ``mot`` and ``proprio_encoder``.

    The official checkpoint stores only these trainable components.  It does
    not contain the Wan VAE or text encoder, so they are deliberately outside
    this audit's coverage denominator.
    """

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", mmap=True)
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ValueError(f"Expected an official FastWAM payload with `mot`: {path}")

    report: dict[str, Any] = {
        "checkpoint": str(path),
        "checkpoint_step": payload.get("step"),
        "checkpoint_torch_dtype": str(payload.get("torch_dtype")),
        "mot": _audit_state_dict(payload["mot"], model.mot.state_dict()),
    }
    if "proprio_encoder" in payload:
        if getattr(model, "proprio_encoder", None) is None:
            report["proprio_encoder"] = {"exact": False, "reason": "model_has_no_proprio_encoder"}
        else:
            report["proprio_encoder"] = _audit_state_dict(
                payload["proprio_encoder"], model.proprio_encoder.state_dict()
            )

    component_reports = [value for value in report.values() if isinstance(value, dict) and "exact" in value]
    report["exact"] = bool(component_reports) and all(component["exact"] for component in component_reports)
    return report
