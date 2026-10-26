"""Classical inverse-filtering methods for deconvolution with a known PSF."""

from __future__ import annotations

from .richardson_lucy import RichardsonLucyDeconvolution
from .wiener import WienerDeconvolution

__all__ = ["WienerDeconvolution", "RichardsonLucyDeconvolution"]
