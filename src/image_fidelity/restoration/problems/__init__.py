"""Inverse imaging problems: forward degradation models and the parameters needed to invert them."""

from __future__ import annotations

from .base import InverseProblem
from .deconvolution import DeconvolutionProblem
from .denoising import DenoisingProblem
from .inpainting import InpaintingProblem
from .super_resolution import SuperResolutionProblem

__all__ = [
    "InverseProblem",
    "DenoisingProblem",
    "DeconvolutionProblem",
    "InpaintingProblem",
    "SuperResolutionProblem",
]
