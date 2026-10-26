"""Visualization helpers for images and restoration results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

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
