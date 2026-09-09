"""Resolve the torch device for embeddings and the cross-encoder."""

from __future__ import annotations

import logging

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
