"""Variational (energy-minimization) restoration methods."""

from __future__ import annotations

from .tikhonov import TikhonovDeconvolution
from .total_variation import TotalVariationDenoising

__all__ = ["TikhonovDeconvolution", "TotalVariationDenoising"]
