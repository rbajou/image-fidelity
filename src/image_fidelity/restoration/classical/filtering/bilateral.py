"""Bilateral filter: edge-preserving smoothing that weights neighbors by spatial and intensity distance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..base import Restorer


@dataclass
class BilateralFilter(Restorer):
    """Edge-preserving denoiser: averages neighbors weighted by spatial closeness and intensity similarity.

    ``sigma_spatial`` controls the size of the neighborhood (pixels),
    ``sigma_range`` controls how quickly the intensity weight falls off with
    pixel value difference (small values preserve edges more strongly).
    """

    sigma_spatial: float = 2.0
    sigma_range: float = 0.1
    radius: Optional[int] = None

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        r = self.radius if self.radius is not None else max(1, int(np.ceil(3 * self.sigma_spatial)))
        image = observed if observed.ndim == 3 else observed[..., None]
        h, w, _ = image.shape

        padded = np.pad(image, ((r, r), (r, r), (0, 0)), mode="reflect").astype(np.float64)
        center = padded[r : r + h, r : r + w, :]
        accum = np.zeros_like(center)
        weight_sum = np.zeros_like(center)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                spatial_weight = np.exp(-(dy**2 + dx**2) / (2 * self.sigma_spatial**2))
                shifted = padded[r + dy : r + dy + h, r + dx : r + dx + w, :]
                range_weight = np.exp(-((shifted - center) ** 2) / (2 * self.sigma_range**2))
                weight = spatial_weight * range_weight
                accum += weight * shifted
                weight_sum += weight

        result = (accum / weight_sum).reshape(observed.shape)
        return result.astype(observed.dtype, copy=False)
