"""Resolve the torch device for embeddings and the cross-encoder."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def resolve_torch_device(preferred: str | None = None) -> str:
    """
    Prefer CUDA when the GPU is visible to PyTorch.

    `preferred` comes from EMBEDDING_DEVICE. Empty/None means auto-detect.
    """
    import torch

    requested = (preferred or "").strip().lower() or None

    if requested == "cpu":
        return "cpu"

    if requested and requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "EMBEDDING_DEVICE=%s but PyTorch has no CUDA build or GPU; using CPU",
            preferred,
        )
        return "cpu"

    if torch.cuda.is_available():
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        return "cuda"

    logger.info("CUDA not available to PyTorch; using CPU")
    return "cpu"


def _blank_gpu(device: str) -> dict[str, Any]:
    return {
        "available": device == "cuda",
        "device": device,
        "name": None,
        "utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
        "temperature_c": None,
    }


def _metric(value: str) -> int | None:
    text = value.strip()
    if not text or text.lower() in {"[n/a]", "n/a"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_gpu_csv(text: str) -> dict[str, Any] | None:
    """Parse one `nvidia-smi --format=csv,noheader,nounits` row."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4 or not parts[3]:
        return None
    return {
        "available": True,
        "device": "cuda",
        "name": parts[3],
        "utilization_percent": _metric(parts[0]),
        "memory_used_mb": _metric(parts[1]),
        "memory_total_mb": _metric(parts[2]),
        "temperature_c": _metric(parts[4]) if len(parts) > 4 else None,
    }


def _nvidia_smi_csv() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,name,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _torch_gpu_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return _blank_gpu("cpu")
    if not torch.cuda.is_available():
        return _blank_gpu("cpu")
    try:
        free, total = torch.cuda.mem_get_info(0)
    except (RuntimeError, AttributeError):
        logger.warning("Could not read CUDA memory", exc_info=True)
        snapshot = _blank_gpu("cuda")
    else:
        snapshot = _blank_gpu("cuda")
        snapshot["memory_total_mb"] = int(total / (1024 * 1024))
        snapshot["memory_used_mb"] = int((total - free) / (1024 * 1024))
    try:
        snapshot["name"] = torch.cuda.get_device_name(0)
    except RuntimeError:
        logger.warning("Could not read CUDA device name", exc_info=True)
    return snapshot


def gpu_snapshot() -> dict[str, Any]:
    """Machine GPU stats for the dashboard. CPU hosts return device=cpu."""
    parsed = parse_gpu_csv(_nvidia_smi_csv() or "")
    if parsed is not None:
        return parsed
    return _torch_gpu_snapshot()
