"""Configuration-driven loading and preparation of dataset samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from . import BaseImageDataset, DatasetSubset, Sample, load_experiment_split, open_dataset


def normalize_sample(sample: Sample) -> Sample:
    image = sample.image
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32, copy=False)
    return Sample(image=image, target=sample.target, metadata=sample.metadata)


def load_samples(
    config: DictConfig,
    split: str | None = None,
    max_samples: int | None = None,
    exclusive_train: bool = False,
) -> tuple[DatasetSubset | BaseImageDataset, list[Sample]]:
    """Load and prepare a split; optionally keep training disjoint from validation."""
    data_config = config.data
    task_config = config.task
    split = str(split if split is not None else task_config.split)
    root = Path(data_config.root_dir)
    train_dir = data_config.get("train_dir")
    test_dir = data_config.get("test_dir")
    grayscale = bool(data_config.get("grayscale", False))
    val_fraction = float(data_config.get("val_fraction", 0.15))
    seed = int(data_config.get("seed", config.seed))
    shuffle = bool(data_config.get("shuffle", True))
    limit_keys = {"test": "test", "train": "train", "val": "val", "validation": "val"}
    if split not in limit_keys:
        raise ValueError(f"Unsupported dataset split {split!r}; choose train, validation, or test")
    limit_key = limit_keys[split]

    if train_dir and test_dir and split in {"test", "train"} and not (exclusive_train and split == "train"):
        dataset = open_dataset(root / str(test_dir if split == "test" else train_dir), normalize_sample, grayscale)
    else:
        experiment_split = load_experiment_split(
            root,
            train_dir=train_dir,
            test_dir=test_dir,
            val_fraction=val_fraction,
            test_fraction=float(data_config.get("test_fraction", 0.15)),
            seed=seed,
            shuffle=shuffle,
            transform=normalize_sample,
            grayscale=grayscale,
        )
        dataset = {"test": experiment_split.test, "train": experiment_split.train}.get(
            split, experiment_split.validation
        )

    limits = data_config.get("limit", {})
    data_limit = limits.get(limit_key) if limits else None
    task_limit = max_samples if max_samples is not None else task_config.get("max_samples")
    selected_limits = [int(value) for value in (task_limit, data_limit) if value is not None]
    limit = min(selected_limits) if selected_limits else len(dataset)
    if limit < 1:
        raise ValueError("The effective sample limit must be at least 1")
    dataset = DatasetSubset(dataset, range(min(limit, len(dataset))))
    samples = list(dataset)
    if not samples:
        raise ValueError(f"No samples found in the {split!r} split under {root}")
    return dataset, samples


def seed_component(component: Any, seed: int) -> None:
    if hasattr(component, "seed"):
        component.seed = np.random.default_rng(seed)


def sample_id(sample: Sample, index: int) -> str:
    return str(sample.metadata.get("sample_id", f"sample-{index:06d}"))
