"""Generic dataset for arbitrary image collections stored in a (possibly nested) folder."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..io.readers import read_image
from .base import BaseImageDataset, Sample

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".bmp")


class ImageFolderDataset(BaseImageDataset):
    """Loads every image found recursively under ``root_dir`` (any layout, mixed formats).

    Samples are sorted by relative path; ``sample_id`` is the relative path without
    extension (POSIX separators). No target is provided.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable[[Sample], Sample]] = None,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
        grayscale: bool = False,
    ):
        super().__init__(transform=transform)
        self.root_dir = Path(root_dir)
        self.extensions = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions)
        self.grayscale = grayscale
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        self.paths: List[Path] = sorted(
            p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.extensions
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _load_sample(self, index: int) -> Sample:
        path = self.paths[index]
        image = read_image(path)
        if self.grayscale and image.ndim == 3 and image.shape[-1] in (3, 4):
            weights = np.array([0.299, 0.587, 0.114])
            gray = image[..., :3].astype(np.float64) @ weights
            image = gray.astype(image.dtype) if np.issubdtype(image.dtype, np.integer) else gray
        rel = path.relative_to(self.root_dir)
        return Sample(
            image=image,
            target=None,
            metadata={"sample_id": rel.with_suffix("").as_posix(), "image_path": str(path)},
        )


def open_dataset(
    path: str | Path,
    transform: Optional[Callable[[Sample], Sample]] = None,
    grayscale: bool = False,
) -> BaseImageDataset:
    """Open ``path`` with the BBBC038 layout if it matches (``<id>/images/``), else as a generic image folder."""
    from .bbbc038 import BBBC038Dataset

    path = Path(path)
    if path.is_dir() and any((p / "images").is_dir() for p in path.iterdir() if p.is_dir()):
        return BBBC038Dataset(path, transform=transform, grayscale=grayscale)
    return ImageFolderDataset(path, transform=transform, grayscale=grayscale)


def load_experiment_split(
    root_dir: str | Path,
    train_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
    transform: Optional[Callable[[Sample], Sample]] = None,
    grayscale: bool = False,
):
    """Build an `ExperimentSplit` for any dataset.

    With ``train_dir`` and ``test_dir`` (relative to ``root_dir``) the predefined
    partitions are used and validation is carved out of train. Otherwise all images
    under ``root_dir`` are randomly split by ``val_fraction`` / ``test_fraction``.
    """
    from .splits import make_experiment_split, split_dataset

    root = Path(root_dir)
    if train_dir and test_dir:
        return make_experiment_split(
            open_dataset(root / train_dir, transform, grayscale),
            open_dataset(root / test_dir, transform, grayscale),
            validation_ratio=val_fraction,
            seed=seed,
            shuffle=shuffle,
        )
    if train_dir or test_dir:
        raise ValueError("train_dir and test_dir must be both set or both unset")
    return split_dataset(
        open_dataset(root, transform, grayscale),
        ratios=(1.0 - val_fraction - test_fraction, val_fraction, test_fraction),
        seed=seed,
        shuffle=shuffle,
    )
