"""Adapters exposing trained PyTorch models through the NumPy ``Restorer`` interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from image_fidelity.nn.torch_utils import from_tensor, load_checkpoint, to_tensor

from ..classical.base import Restorer


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
