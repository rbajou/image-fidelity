"""Binary neural segmentation with connected-component instance labels."""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from image_fidelity.nn.torch_utils import load_checkpoint, to_tensor


class TorchSegmenter:
    """Convert single-channel logits into an integer instance mask."""

    def __init__(
        self, model: nn.Module, checkpoint: str | Path | None = None,
        device: str = "cpu", threshold: float = 0.5, connectivity: int = 2,
    ):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be strictly between 0 and 1")
        if connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 (4-neighbor) or 2 (8-neighbor)")
        if getattr(model, "residual", False):
            raise ValueError("Segmentation requires a model without residual prediction")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if checkpoint is not None:
            load_checkpoint(self.model, checkpoint, self.device)
        self.model.eval()
        self.threshold = threshold
        self.connectivity = connectivity

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        logits = self.model(to_tensor(image).unsqueeze(0).to(self.device))
        if logits.shape != (1, 1, *image.shape[:2]):
            raise ValueError(f"Expected logits of shape (1, 1, H, W), got {tuple(logits.shape)}")
        foreground = (logits.sigmoid()[0, 0] >= self.threshold).cpu().numpy()
        labels, _ = ndimage.label(foreground, structure=ndimage.generate_binary_structure(2, self.connectivity))
        return labels.astype(np.int32, copy=False)
