import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.datasets import (
    BBBC038Dataset,
    ExperimentSplit,
    PatchExtractor,
    make_experiment_split,
    split_dataset,
)


def _create_bbbc_split(root, split, sample_count):
    for index in range(sample_count):
        sample_id = f"{split}-{index}"
        sample_dir = root / split / sample_id
        images_dir = sample_dir / "images"
        masks_dir = sample_dir / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir()
        image = np.full((64, 64), index, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        Image.fromarray(image).save(images_dir / f"{sample_id}.png")
        Image.fromarray(mask).save(masks_dir / "instance.png")


def test_split_dataset_ratios_and_no_overlap(tmp_path):
    """The requested split ratios partition every sample without overlap."""
    _create_bbbc_split(tmp_path, "stage1_train", sample_count=10)
    dataset = BBBC038Dataset(tmp_path / "stage1_train")
    split = split_dataset(dataset, ratios=(0.8, 0.1, 0.1), seed=0)

    assert isinstance(split, ExperimentSplit)
    assert len(split.train) + len(split.validation) + len(split.test) == len(dataset)

    train_ids = {s.metadata["sample_id"] for s in split.train}
    val_ids = {s.metadata["sample_id"] for s in split.validation}
    test_ids = {s.metadata["sample_id"] for s in split.test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_dataset_is_reproducible(tmp_path):
    """Using the same seed produces the same training subset."""
    _create_bbbc_split(tmp_path, "stage1_train", sample_count=10)
    dataset = BBBC038Dataset(tmp_path / "stage1_train")
    split_a = split_dataset(dataset, seed=123)
    split_b = split_dataset(dataset, seed=123)
    assert split_a.train.indices == split_b.train.indices


def test_make_experiment_split_uses_test_dataset_as_is(tmp_path):
    """A predefined test dataset stays intact while validation comes from train."""
    _create_bbbc_split(tmp_path, "stage1_train", sample_count=10)
    _create_bbbc_split(tmp_path, "stage1_test", sample_count=2)
    train_dataset = BBBC038Dataset(tmp_path / "stage1_train")
    test_dataset = BBBC038Dataset(tmp_path / "stage1_test")

    split = make_experiment_split(train_dataset, test_dataset, validation_ratio=0.1, seed=0)

    assert split.test is test_dataset
    assert len(split.train) + len(split.validation) == len(train_dataset)


def test_patch_extractor_no_overlap_exact_grid():
    """Exact tiling produces the expected grid and preserves edge patches."""
    image = np.arange(1024 * 1024 * 3, dtype=np.uint8).reshape(1024, 1024, 3)
    extractor = PatchExtractor(patch_size=256)

    patches = extractor.extract(image)

    assert patches.shape == (16, 256, 256, 3)
    assert extractor.grid_shape(1024, 1024) == (4, 4)
    np.testing.assert_array_equal(patches[0], image[:256, :256, :])
    np.testing.assert_array_equal(patches[-1], image[768:, 768:, :])


def test_patch_extractor_pads_when_not_evenly_tileable():
    """Reflect padding makes a non-tileable image fit a complete patch grid."""
    image = np.tile(np.arange(300, dtype=np.uint16), (300, 1))
    extractor = PatchExtractor(patch_size=256, pad_mode="reflect")

    patches = extractor.extract(image)

    assert extractor.grid_shape(300, 300) == (2, 2)
    assert patches.shape == (4, 256, 256)
    assert patches[1, 0, 44] == image[0, 298]
    assert patches[1, 0, 45] == image[0, 297]


def test_patch_extractor_rejects_non_tileable_image_without_padding():
    """A non-tileable image raises a clear error when padding is disabled."""
    image = np.zeros((300, 300), dtype=np.uint8)
    extractor = PatchExtractor(patch_size=256, pad_mode=None)

    with pytest.raises(ValueError, match="pad_mode"):
        extractor.extract(image)


def test_patch_extractor_extract_sample_keeps_image_and_target_aligned(tmp_path):
    """Extracted image and target patches remain spatially aligned."""
    _create_bbbc_split(tmp_path, "stage1_train", sample_count=1)
    dataset = BBBC038Dataset(tmp_path / "stage1_train")
    sample = dataset[0]
    extractor = PatchExtractor(patch_size=64)

    patches = extractor.extract_sample(sample)

    assert len(patches) > 0
    for i, patch in enumerate(patches):
        assert patch.image.shape[:2] == (64, 64)
        assert patch.target.shape == (64, 64)
        assert patch.metadata["patch_index"] == i
        assert patch.metadata["sample_id"] == sample.metadata["sample_id"]
