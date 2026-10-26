import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.io import read_image, write_image, extract_metadata
from image_fidelity.datasets import BBBC038Dataset

DATA_ROOT = ROOT / "data" / "bbbc038"


def test_read_png_metadata_matches_pixels():
    sample_dir = next((DATA_ROOT / "stage1_train").iterdir())
    image_path = next((sample_dir / "images").iterdir())

    array = read_image(image_path)
    meta = extract_metadata(image_path)

    assert array.shape[:2] == meta.shape[:2]
    assert array.dtype.name == meta.dtype


def test_write_read_roundtrip_npy(tmp_path):
    array = np.arange(24, dtype=np.uint16).reshape(4, 6)
    path = tmp_path / "arr.npy"
    write_image(path, array)
    loaded = read_image(path)
    np.testing.assert_array_equal(array, loaded)


def test_write_read_roundtrip_tiff(tmp_path):
    array = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    path = tmp_path / "arr.tiff"
    write_image(path, array)
    loaded = read_image(path)
    np.testing.assert_array_equal(array, loaded)


def test_bbbc038_train_sample_has_instance_mask():
    dataset = BBBC038Dataset(DATA_ROOT / "stage1_train")
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample.image.ndim in (2, 3)
    assert sample.target is not None
    assert sample.target.shape == sample.image.shape[:2]
    assert sample.metadata["n_instances"] == int(sample.target.max())


def test_bbbc038_test_sample_has_no_mask():
    dataset = BBBC038Dataset(DATA_ROOT / "stage1_test")
    sample = dataset[0]
    assert sample.target is None
    assert sample.metadata["n_instances"] == 0
