"""Non-local means denoising backed by scikit-image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Restorer


@dataclass
class NonLocalMeans(Restorer):
    patch_size: int = 5
    patch_distance: int = 6
    h: float = 0.1
    fast_mode: bool = True

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        try:
            from skimage.restoration import denoise_nl_means
        except ImportError as exc:
            raise ImportError("Non-local means requires scikit-image; install the project dependencies.") from exc

        result = denoise_nl_means(
            observed,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            h=self.h,
            fast_mode=self.fast_mode,
            preserve_range=True,
            channel_axis=-1 if observed.ndim == 3 else None,
        )
        return np.asarray(result, dtype=observed.dtype)