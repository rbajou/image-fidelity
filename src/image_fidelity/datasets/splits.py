"""Train/validation/test splitting utilities for map-style image datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .base import BaseImageDataset, Sample


@dataclass
class ExperimentSplit:
    """Bundles the three dataset partitions used throughout an experiment."""

    train: BaseImageDataset
    validation: BaseImageDataset
    test: BaseImageDataset


class DatasetSubset(BaseImageDataset):
    """A read-only view over a subset of another dataset's indices."""

    def __init__(self, dataset: BaseImageDataset, indices: Sequence[int]):
        super().__init__(transform=None)
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_sample(self, index: int) -> Sample:
        return self.dataset[self.indices[index]]


def split_dataset(
    dataset: BaseImageDataset,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    shuffle: bool = True,
) -> ExperimentSplit:
    """Randomly partition a dataset into train/validation/test subsets.

    ``ratios`` (train, validation, test) must sum to 1.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios} (sum={sum(ratios)})")

    indices: List[int] = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    n_train = int(round(ratios[0] * len(indices)))
    n_val = int(round(ratios[1] * len(indices)))

    return ExperimentSplit(
        train=DatasetSubset(dataset, indices[:n_train]),
        validation=DatasetSubset(dataset, indices[n_train : n_train + n_val]),
        test=DatasetSubset(dataset, indices[n_train + n_val :]),
    )


def make_experiment_split(
    train_dataset: BaseImageDataset,
    test_dataset: BaseImageDataset,
    validation_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> ExperimentSplit:
    """Build an `ExperimentSplit` from separately-loaded train/test datasets.

    Useful for datasets like BBBC038 that ship pre-defined train/test folders:
    a validation subset is carved out of ``train_dataset``, and ``test_dataset``
    is used as-is.
    """
    indices: List[int] = list(range(len(train_dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    n_val = int(round(validation_ratio * len(indices)))

    return ExperimentSplit(
        train=DatasetSubset(train_dataset, indices[n_val:]),
        validation=DatasetSubset(train_dataset, indices[:n_val]),
        test=test_dataset,
    )
