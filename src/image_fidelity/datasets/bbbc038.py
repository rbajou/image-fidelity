"""Dataset wrapper for BBBC038 (2018 Data Science Bowl nuclei segmentation)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..io.readers import read_image
from .base import BaseImageDataset, Sample


class BBBC038Dataset(BaseImageDataset):
    """Loads BBBC038 samples stored one folder per image::

        <root>/<sample_id>/images/<sample_id>.png
        <root>/<sample_id>/masks/*.png   (one binary mask per instance; absent for unlabeled test data)

    Each item exposes the raw image and, when available, an instance-labeled
    mask (0 = background, 1..N = individual nucleus instances).
    """

    IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable[[Sample], Sample]] = None,
        grayscale: bool = False,
    ):
        super().__init__(transform=transform)
        self.grayscale = grayscale
        self.root_dir = Path(root_dir)
        self.sample_ids: List[str] = self._discover_sample_ids()

    def _discover_sample_ids(self) -> List[str]:
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"BBBC038 root directory not found: {self.root_dir}")
        return sorted(p.name for p in self.root_dir.iterdir() if p.is_dir() and (p / "images").is_dir())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _find_image_path(self, sample_dir: Path) -> Path:
        images_dir = sample_dir / "images"
        candidates = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS)
        if not candidates:
            raise FileNotFoundError(f"No image found in {images_dir}")
        return candidates[0]

    def _load_instance_mask(self, sample_dir: Path, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        masks_dir = sample_dir / "masks"
        if not masks_dir.is_dir():
            return None
        mask_paths = sorted(p for p in masks_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS)
        if not mask_paths:
            return None
        instance_mask = np.zeros(shape, dtype=np.int32)
        for label, mask_path in enumerate(mask_paths, start=1):
            instance_mask[read_image(mask_path) > 0] = label
        return instance_mask

    def _load_sample(self, index: int) -> Sample:
        sample_id = self.sample_ids[index]
        sample_dir = self.root_dir / sample_id
        image_path = self._find_image_path(sample_dir)
        image = read_image(image_path)
        instance_mask = self._load_instance_mask(sample_dir, image.shape[:2])
        if self.grayscale and image.ndim == 3 and image.shape[-1] in (3, 4):
            gray = image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
            image = gray.astype(image.dtype) if np.issubdtype(image.dtype, np.integer) else gray
        return Sample(
            image=image,
            target=instance_mask,
            metadata={
                "sample_id": sample_id,
                "image_path": str(image_path),
                "n_instances": int(instance_mask.max()) if instance_mask is not None else 0,
            },
        )
