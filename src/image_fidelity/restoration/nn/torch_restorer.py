"""Adapters exposing trained PyTorch models through the NumPy ``Restorer`` interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..classical.base import Restorer


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


class TorchRestorer(Restorer):
    """Run a trained image-to-image network on a single image."""

    def __init__(self, model: nn.Module, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if checkpoint is not None:
            load_checkpoint(self.model, checkpoint, self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, observed: np.ndarray) -> np.ndarray:
        batch = to_tensor(observed).unsqueeze(0).to(self.device)
        return from_tensor(self.model(batch)[0], np.asarray(observed))

    def __repr__(self) -> str:
        return f"TorchRestorer(model={type(self.model).__name__}, device={self.device})"
