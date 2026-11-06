"""Inpainting problem: recover a masked-out region of an image from a known (or random) mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from image_fidelity.degradation.base import as_rng

from .base import InverseProblem


@dataclass
class InpaintingProblem(InverseProblem):
    """Observation is the clean image with an unknown region masked out: ``y = mask * x + fill_value``.

    ``mask`` is boolean and marks the *known* (kept) pixels. If not
    provided, a random mask dropping ``missing_fraction`` of pixels (all
    channels together) is drawn on each call.
    """

    mask: Optional[np.ndarray] = None
    missing_fraction: float = 0.2
    fill_value: float = 0.0
    seed: Optional[Union[int, np.random.Generator]] = None

    def _mask_for(self, image: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return self.mask
        rng = as_rng(self.seed)
        return rng.random(image.shape[:2]) >= self.missing_fraction

    def degrade(self, image: np.ndarray) -> np.ndarray:
        mask = self._mask_for(image)
        out = image.copy()
        out[~mask] = self.fill_value
        return out

    def operator_params(self) -> Dict[str, Any]:
        return {"mask": self.mask, "fill_value": self.fill_value}
