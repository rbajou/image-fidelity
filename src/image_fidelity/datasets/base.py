"""Base classes shared by all image datasets (framework-agnostic, no torch dependency)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np


@dataclass
class Sample:
    """A single dataset item: an image, an optional target, and metadata."""

    image: np.ndarray
    target: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseImageDataset(ABC):
    """Minimal map-style dataset interface: implement ``__len__`` and ``_load_sample``."""

    def __init__(self, transform: Optional[Callable[[Sample], Sample]] = None):
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _load_sample(self, index: int) -> Sample:
        """Load the raw sample (image + target + metadata) at the given index."""

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        sample = self._load_sample(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]
