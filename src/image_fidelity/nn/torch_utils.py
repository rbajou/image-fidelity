"""Shared PyTorch checkpoint and image conversion helpers."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_checkpoint(model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load model weights from a checkpoint written by the training task."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """(H, W) or (H, W, C) array -> (C, H, W) float tensor."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Expected a 2D or (H, W, C) image, got shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array))


def from_tensor(tensor: torch.Tensor, like: np.ndarray) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = array[0] if like.ndim == 2 else np.transpose(array, (1, 2, 0))
    return array.astype(like.dtype if np.issubdtype(like.dtype, np.floating) else np.float32, copy=False)
