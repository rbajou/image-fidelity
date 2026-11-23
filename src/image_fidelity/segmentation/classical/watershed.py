"""Learning-free baseline: thresholding, morphology, distance transform and watershed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class WatershedConfig:
    threshold: Literal["otsu", "local"] = "otsu"
    local_block_size: int = 51  # forced odd, for local thresholding
    local_offset: float = 0.0
    bright_objects: bool | Literal["auto"] = True  # False: dark nuclei; "auto": minority class
    sigma: float = 1.0  # Gaussian pre-smoothing (0 = none)
    opening_radius: int = 2
    min_size: int = 30  # minimum object / hole size (pixels)
    min_distance: int = 5  # minimum distance between markers
    h_maxima: float | None = None  # if set, markers are h-maxima of the distance map
    smooth_distance_sigma: float = 1.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image (or HxWxC), got {image.shape}")
    return img


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return sizes[labels] >= min_size


def foreground_mask(image: np.ndarray, cfg: WatershedConfig = WatershedConfig()) -> np.ndarray:
    """Threshold (Otsu/local), then clean up morphologically."""
    img = _to_gray(image)
    if cfg.sigma > 0:
        img = filters.gaussian(img, sigma=cfg.sigma)
    if img.max() == img.min():
        return np.zeros(img.shape, dtype=bool)

    if cfg.threshold == "otsu":
        mask = img > filters.threshold_otsu(img)
    elif cfg.threshold == "local":
        block = cfg.local_block_size | 1
        mask = img > filters.threshold_local(img, block, offset=cfg.local_offset)
    else:
        raise ValueError(f"Unknown threshold: {cfg.threshold!r}")
    if cfg.bright_objects == "auto":
        bright = img > filters.threshold_otsu(img)
        bright_objects = bright.mean() <= 0.5
    else:
        bright_objects = bool(cfg.bright_objects)
    if not bright_objects:
        mask = ~mask

    if cfg.opening_radius > 0:
        mask = morphology.opening(mask, morphology.disk(cfg.opening_radius))
    mask = _remove_small(mask, cfg.min_size)
    mask = ~_remove_small(~mask, cfg.min_size)
    return mask


def watershed_segment(image: np.ndarray, cfg: WatershedConfig = WatershedConfig()) -> np.ndarray:
    """Instance segmentation: return an int32 label map (0 = background)."""
    mask = foreground_mask(image, cfg)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    dist = ndi.distance_transform_edt(mask)
    if cfg.smooth_distance_sigma > 0:
        dist = filters.gaussian(dist, sigma=cfg.smooth_distance_sigma)

    if cfg.h_maxima is not None:
        peaks = morphology.h_maxima(dist, cfg.h_maxima).astype(bool) & mask
    else:
        coords = peak_local_max(
            dist, min_distance=cfg.min_distance, labels=measure.label(mask), exclude_border=False
        )
        peaks = np.zeros(mask.shape, dtype=bool)
        peaks[tuple(coords.T)] = True
    markers = measure.label(peaks)

    labels = segmentation.watershed(-dist, markers, mask=mask)
    counts = np.bincount(labels.ravel())
    keep = counts >= cfg.min_size
    keep[0] = False
    lut = np.cumsum(keep) * keep  # sequential labels, 0 = background
    return lut[labels].astype(np.int32)


class WatershedSegmenter:
    """Callable image -> instance map, usable by tasks and metrics."""

    def __init__(self, **parameters):
        self.config = WatershedConfig(**parameters)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return watershed_segment(image, self.config)
