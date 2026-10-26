"""Baseline sans apprentissage : seuillage, morphologie, distance, watershed."""

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
    local_block_size: int = 51  # forcé impair, pour le seuillage local
    local_offset: float = 0.0
    bright_objects: bool | Literal["auto"] = True  # False : noyaux sombres ; "auto" : classe minoritaire
    sigma: float = 1.0  # lissage gaussien préalable (0 = aucun)
    opening_radius: int = 2
    min_size: int = 30  # taille minimale des objets / trous (pixels)
    min_distance: int = 5  # distance minimale entre marqueurs
    h_maxima: float | None = None  # si défini, marqueurs = h-maxima de la distance
    smooth_distance_sigma: float = 1.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    if img.ndim != 2:
        raise ValueError(f"Image 2D attendue (ou HxWxC), reçu {image.shape}")
    return img


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Retire les composantes connexes de moins de `min_size` pixels."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return sizes[labels] >= min_size


def foreground_mask(image: np.ndarray, cfg: WatershedConfig = WatershedConfig()) -> np.ndarray:
    """Seuillage (Otsu/local) puis nettoyage morphologique."""
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
        raise ValueError(f"threshold inconnu : {cfg.threshold!r}")
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
    """Segmentation d'instances : renvoie une carte d'étiquettes int32 (0 = fond)."""
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
    lut = np.cumsum(keep) * keep  # étiquettes séquentielles, 0 = fond
    return lut[labels].astype(np.int32)


class WatershedSegmenter:
    """Callable image -> carte d'instances, utilisable par les tâches et métriques."""

    def __init__(self, **parameters):
        self.config = WatershedConfig(**parameters)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return watershed_segment(image, self.config)
