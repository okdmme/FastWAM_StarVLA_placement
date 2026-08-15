from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_dataset_stats(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _field_bounds(stats: dict[str, Any], field: str) -> tuple[torch.Tensor, torch.Tensor]:
    field_stats = stats[field]["default"]
    if "global_min" not in field_stats or "global_max" not in field_stats:
        raise ValueError(f"Dataset stats missing {field}.default.global_min/global_max")
    lo = torch.tensor(field_stats["global_min"], dtype=torch.float32)
    hi = torch.tensor(field_stats["global_max"], dtype=torch.float32)
    return lo, hi


def normalize_minmax(value: torch.Tensor, stats: dict[str, Any], field: str) -> torch.Tensor:
    lo, hi = _field_bounds(stats, field)
    lo = lo.to(device=value.device, dtype=value.dtype)
    hi = hi.to(device=value.device, dtype=value.dtype)
    return 2.0 * (value - lo) / (hi - lo).clamp_min(torch.finfo(value.dtype).eps) - 1.0


def unnormalize_minmax(value: torch.Tensor, stats: dict[str, Any], field: str) -> torch.Tensor:
    lo, hi = _field_bounds(stats, field)
    lo = lo.to(device=value.device, dtype=value.dtype)
    hi = hi.to(device=value.device, dtype=value.dtype)
    return (value + 1.0) * 0.5 * (hi - lo) + lo


def deterministic_anchor_state(stats: dict[str, Any]) -> torch.Tensor:
    lo, hi = _field_bounds(stats, "state")
    return ((lo + hi) * 0.5).to(torch.float32)

