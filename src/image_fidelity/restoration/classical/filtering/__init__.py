"""Classical pixel-neighborhood filters used as simple denoisers."""

from __future__ import annotations

from .bilateral import BilateralFilter
from .gaussian import GaussianFilter
from .median import MedianFilter
from .non_local_means import NonLocalMeans

__all__ = ["GaussianFilter", "MedianFilter", "BilateralFilter", "NonLocalMeans"]
