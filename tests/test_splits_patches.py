import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.datasets import (
    BBBC038Dataset,
    ExperimentSplit,
    PatchExtractor,
    make_experiment_split,
    split_dataset,
)

DATA_ROOT = ROOT / "data" / "bbbc038"


def test_split_dataset_ratios_and_no_overlap():
    dataset = BBBC038Dataset(DATA_ROOT / "stage1_train")
    split = split_dataset(dataset, ratios=(0.8, 0.1, 0.1), seed=0)

    assert isinstance(split, ExperimentSplit)
    assert len(split.train) + len(split.validation) + len(split.test) == len(dataset)

    train_ids = {s.metadata["sample_id"] for s in split.train}
    val_ids = {s.metadata["sample_id"] for s in split.validation}
    test_ids = {s.metadata["sample_id"] for s in split.test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_dataset_is_reproducible():
    dataset = BBBC038Dataset(DATA_ROOT / "stage1_train")
    split_a = split_dataset(dataset, seed=123)
    split_b = split_dataset(dataset, seed=123)
    assert split_a.train.indices == split_b.train.indices


def test_make_experiment_split_uses_test_dataset_as_is():
    train_dataset = BBBC038Dataset(DATA_ROOT / "stage1_train")
    test_dataset = BBBC038Dataset(DATA_ROOT / "stage1_test")

    split = make_experiment_split(train_dataset, test_dataset, validation_ratio=0.1, seed=0)

    assert split.test is test_dataset
    assert len(split.train) + len(split.validation) == len(train_dataset)


def test_patch_extractor_no_overlap_exact_grid():
    image = np.arange(1024 * 1024 * 3, dtype=np.uint8).reshape(1024, 1024, 3)
    extractor = PatchExtractor(patch_size=256)

    patches = extractor.extract(image)

    assert patches.shape == (16, 256, 256, 3)
    assert extractor.grid_shape(1024, 1024) == (4, 4)
    np.testing.assert_array_equal(patches[0], image[:256, :256, :])
    np.testing.assert_array_equal(patches[-1], image[768:, 768:, :])


def test_patch_extractor_pads_when_not_evenly_tileable():
    image = np.zeros((300, 300), dtype=np.uint8)
    extractor = PatchExtractor(patch_size=256, pad_mode="reflect")

    patches = extractor.extract(image)

    assert extractor.grid_shape(300, 300) == (2, 2)
    assert patches.shape == (4, 256, 256)


def test_patch_extractor_raises_without_pad_mode():
    image = np.zeros((300, 300), dtype=np.uint8)
    extractor = PatchExtractor(patch_size=256, pad_mode=None)

    try:
        extractor.extract(image)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-tileable array without pad_mode")


def test_patch_extractor_extract_sample_keeps_image_and_target_aligned():
    dataset = BBBC038Dataset(DATA_ROOT / "stage1_train")
    sample = dataset[0]
    extractor = PatchExtractor(patch_size=64)

    patches = extractor.extract_sample(sample)

    assert len(patches) > 0
    for i, patch in enumerate(patches):
        assert patch.image.shape[:2] == (64, 64)
        assert patch.target.shape == (64, 64)
        assert patch.metadata["patch_index"] == i
        assert patch.metadata["sample_id"] == sample.metadata["sample_id"]
