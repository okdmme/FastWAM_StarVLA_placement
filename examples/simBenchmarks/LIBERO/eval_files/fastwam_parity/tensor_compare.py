from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
REQUIRED_TENSOR_RECORD_KEYS = {
    "stage",
    "path",
    "shape",
    "dtype",
    "device",
    "sha256",
}


@dataclass(frozen=True)
class TensorRecord:
    stage: str
    path: str
    shape: list[int]
    dtype: str
    device: str
    sha256: str


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    # NumPy cannot represent torch.bfloat16. Hash the contiguous raw storage
    # through a byte view so the digest preserves the exact tensor dtype/data.
    raw_bytes = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def save_tensor_record(stage: str, tensor: torch.Tensor, output_dir: Path) -> TensorRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = stage.replace("/", "__").replace(" ", "_")
    path = output_dir / f"{safe_name}.pt"
    device = str(tensor.device)
    torch.save(tensor.detach().cpu(), path)
    return TensorRecord(
        stage=stage,
        path=str(path),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=device,
        sha256=tensor_sha256(tensor),
    )


def _first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    if a.shape != b.shape:
        return {"index": None, "official": None, "starvla": None, "reason": "shape_mismatch"}
    neq = a.detach().cpu().ne(b.detach().cpu())
    if not bool(neq.any()):
        return None
    flat_idx = int(neq.reshape(-1).nonzero(as_tuple=False)[0].item())
    idx = list(torch.unravel_index(torch.tensor(flat_idx), a.shape))
    idx = [int(x) for x in idx]
    return {
        "index": idx,
        "official": _scalar_or_list(a.detach().cpu()[tuple(idx)]),
        "starvla": _scalar_or_list(b.detach().cpu()[tuple(idx)]),
    }


def _scalar_or_list(value: torch.Tensor) -> Any:
    if value.numel() == 1:
        return value.item()
    return value.tolist()


def compare_tensors(official: torch.Tensor, starvla: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_equal": bool(torch.equal(official.detach().cpu(), starvla.detach().cpu())),
        "official_shape": list(official.shape),
        "starvla_shape": list(starvla.shape),
        "official_dtype": str(official.dtype),
        "starvla_dtype": str(starvla.dtype),
        "official_sha256": tensor_sha256(official),
        "starvla_sha256": tensor_sha256(starvla),
        "first_mismatch": _first_mismatch(official, starvla),
    }
    if official.shape != starvla.shape:
        result.update(
            {
                "max_abs_error": None,
                "mean_abs_error": None,
                "relative_l2_error": None,
            }
        )
        return result

    a = official.detach().cpu().to(torch.float64)
    b = starvla.detach().cpu().to(torch.float64)
    diff = (a - b).abs()
    denom = torch.linalg.vector_norm(a).clamp_min(torch.finfo(torch.float64).eps)
    result.update(
        {
            "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
            "relative_l2_error": float((torch.linalg.vector_norm(a - b) / denom).item()),
        }
    )
    return result


def write_manifest(records: list[TensorRecord], output_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "records": [asdict(record) for record in records],
    }
    validate_manifest(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest `records` must be a list.")
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record {idx} must be a dict.")
        missing = REQUIRED_TENSOR_RECORD_KEYS.difference(record)
        if missing:
            raise ValueError(f"Manifest record {idx} missing keys: {sorted(missing)}")
        if not isinstance(record["shape"], list):
            raise ValueError(f"Manifest record {idx} `shape` must be a list.")
        if len(str(record["sha256"])) != 64:
            raise ValueError(f"Manifest record {idx} has invalid sha256: {record['sha256']!r}")
