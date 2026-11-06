"""Visualization helpers for images and restoration results."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from .params import RCPARAMS


def show_before_after(
    originals: Sequence[np.ndarray],
    degraded: Sequence[np.ndarray],
    restored: Optional[Sequence[np.ndarray]] = None,
    *,
    titles: Optional[Sequence[str]] = None,
    show: bool = True,
) -> Tuple[object, np.ndarray]:
    """Plot matching original/degraded(/restored) images as rows in one window and return figure and axes."""
    if not originals:
        raise ValueError("at least one image pair is required")
    if len(originals) != len(degraded):
        raise ValueError("originals and degraded must contain the same number of images")
    if restored is not None and len(restored) != len(originals):
        raise ValueError("restored must contain the same number of images as originals")

    plt = _import_pyplot()
    rows = _rows(originals, degraded, restored)
    figure, axes = plt.subplots(len(rows), len(originals), squeeze=False, figsize=(4 * len(originals), 4 * len(rows)))
    _draw(figure, axes, rows, titles)
    if show:
        plt.show()
    return figure, axes


def save_before_after(
    path: str | Path,
    originals: Sequence[np.ndarray],
    degraded: Sequence[np.ndarray],
    restored: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
) -> Path:
    """Save original/degraded/restored images as three rows in an image file, without opening a window."""
    if not originals:
        raise ValueError("at least one image is required")
    if not len(originals) == len(degraded) == len(restored):
        raise ValueError("originals, degraded and restored must contain the same number of images")
    _import_pyplot()
    from matplotlib.figure import Figure

    rows = _rows(originals, degraded, restored)
    figure = Figure(figsize=(4 * len(originals), 4 * len(rows)))
    axes = figure.subplots(len(rows), len(originals), squeeze=False)
    _draw(figure, axes, rows, titles)
    path = Path(path)
    figure.savefig(path, dpi=100)
    return path


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update(RCPARAMS)
    except ImportError as exc:
        raise ImportError(
            "Image visualization requires Matplotlib; install it with `pip install -e '.[visualization]'`."
        ) from exc
    return plt


def _rows(originals, degraded, restored):
    rows = (("Original", originals), ("Degraded", degraded))
    if restored is not None:
        rows += (("Restored", restored),)
    return rows


def _draw(figure, axes, rows, titles=None) -> None:
    if titles is not None:
        if len(titles) != len(rows[0][1]):
            raise ValueError("titles must contain one entry per image")
        for column, title in enumerate(titles):
            axes[0, column].set_title(str(title)[:10]+"...")
    for row, (_, images) in enumerate(rows):
        for column, image in enumerate(images):
            axes[row, column].imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
            axes[row, column].set_axis_off()
    for row, (label, _) in enumerate(rows):
        y = 1 - (row + 0.5) / len(rows)
        figure.text(0.01, y, label, rotation=90, va="center")
    figure.tight_layout(rect=(0.04, 0, 1, 1))

def save_image_pair(
    path: str | Path,
    original: np.ndarray,
    degraded: np.ndarray,
    title: Optional[str] = None,
) -> Path:
    """Save an original/degraded pair side by side in one image file, without opening a window."""
    _import_pyplot()
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.4 if title else 4))
    axes = figure.subplots(1, 2, squeeze=False)[0]
    for axis, label, image in zip(axes, ("Original", "Degraded"), (original, degraded)):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
        axis.set_title(label)
        axis.set_axis_off()
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=100)
    return path


def save_segmentation_preview(
    path: str | Path,
    original: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    *,
    image_id: str,
    method: str,
    degradation: Mapping[str, object],
    scores: Optional[Mapping[str, float]] = None,
) -> Path:
    """Save an image, its degraded input, and ground-truth and predicted instance overlays."""
    from skimage.segmentation import find_boundaries

    _import_pyplot()
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    figure = Figure(figsize=(16, 5))
    axes = figure.subplots(1, 4)
    for axis, image in zip(axes, (original, observed, observed, observed)):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
        axis.set_axis_off()
    axes[0].set_title("Original")
    axes[1].set_title("Degraded input")
    axes[2].set_title("Ground truth" if ground_truth is not None else "Ground truth unavailable")
    axes[3].set_title("Prediction")

    if ground_truth is not None:
        boundaries = find_boundaries(ground_truth, mode="inner", background=0)
        overlay = np.zeros((*boundaries.shape, 4), dtype=float)
        overlay[boundaries] = (0, 1, 1, 1)
        axes[2].imshow(overlay)

    labels, inverse = np.unique(prediction, return_inverse=True)
    colors = colormaps["hsv"]((np.arange(len(labels)) * 0.618033988749895) % 1)
    colors[:, 3] = np.where(labels == 0, 0, 0.45)
    axes[3].imshow(colors[inverse].reshape((*prediction.shape, 4)))
    boundaries = find_boundaries(prediction, mode="inner", background=0)
    overlay = np.zeros((*boundaries.shape, 4), dtype=float)
    overlay[boundaries] = (1, 1, 1, 0.9)
    axes[3].imshow(overlay)

    degradation_label = ", ".join(f"{key}={value}" for key, value in degradation.items())
    title = f"{image_id}\n{method} | {degradation_label}"
    if scores is not None:
        title += (
            f"\nDice={scores['dice']:.3f} | F1={scores['f1']:.3f}"
            f" | instances: predicted={int(scores['n_pred'])}, ground truth={int(scores['n_gt'])}"
        )
    figure.suptitle(title, fontsize="small")
    figure.tight_layout(rect=(0, 0, 1, 0.85))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    return path
