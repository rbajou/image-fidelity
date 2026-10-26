"""Total variation denoising: promotes piecewise-smooth solutions while preserving edges.

Solves ``min_x 0.5 * ||x - y||^2 + weight * TV(x)`` using Chambolle's dual
projected-gradient algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Restorer


@dataclass
class TotalVariationDenoising(Restorer):
    """Denoises ``observed`` by minimizing ``0.5 * ||x - y||^2 + weight * TV(x)`` (Chambolle's algorithm)."""

    weight: float = 0.1
    n_iter: int = 100
    step_size: float = 0.125

    def __call__(self, observed: np.ndarray) -> np.ndarray:
        if observed.ndim == 2:
            return self._denoise_channel(observed)
        return np.stack(
            [self._denoise_channel(observed[..., c]) for c in range(observed.shape[-1])],
            axis=-1,
        )

    def _denoise_channel(self, channel: np.ndarray) -> np.ndarray:
        original = channel.astype(np.float64)
        p = np.zeros(original.shape + (2,), dtype=np.float64)

        for _ in range(self.n_iter):
            div_p = self._divergence(p)
            grad_step = self._gradient(div_p - original / self.weight)
            norm = np.sqrt((grad_step**2).sum(axis=-1, keepdims=True))
            p = (p + self.step_size * grad_step) / (1.0 + self.step_size * norm)

        restored = original - self.weight * self._divergence(p)
        return restored.astype(channel.dtype, copy=False)

    @staticmethod
    def _gradient(x: np.ndarray) -> np.ndarray:
        grad = np.zeros(x.shape + (2,), dtype=np.float64)
        grad[:-1, :, 0] = x[1:, :] - x[:-1, :]
        grad[:, :-1, 1] = x[:, 1:] - x[:, :-1]
        return grad

    @staticmethod
    def _divergence(p: np.ndarray) -> np.ndarray:
        div = np.zeros(p.shape[:2], dtype=np.float64)
        div[:-1, :] += p[:-1, :, 0]
        div[1:, :] -= p[:-1, :, 0]
        div[:, :-1] += p[:, :-1, 1]
        div[:, 1:] -= p[:, :-1, 1]
        return div
