import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.datasets import BBBC038Dataset
from image_fidelity.io import extract_metadata, read_image, write_image


def _create_bbbc_sample(root, split, sample_id, labeled):
    sample_dir = root / split / sample_id
    image_dir = sample_dir / "images"
    image_dir.mkdir(parents=True)
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    Image.fromarray(image).save(image_dir / f"{sample_id}.png")

    if labeled:
        masks_dir = sample_dir / "masks"
        masks_dir.mkdir()
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(masks_dir / "instance.png")


def test_read_png_metadata_matches_pixels(tmp_path):
    """PNG metadata reports the dimensions and dtype of the decoded pixels."""
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    array = read_image(image_path)
    meta = extract_metadata(image_path)

    assert array.shape[:2] == meta.shape[:2]
    assert array.dtype.name == meta.dtype


def test_write_read_roundtrip_npy(tmp_path):
    """Writing and reading a NumPy image preserves every array value."""
    array = np.arange(24, dtype=np.uint16).reshape(4, 6)
    path = tmp_path / "arr.npy"
    write_image(path, array)
    loaded = read_image(path)
    np.testing.assert_array_equal(array, loaded)


def test_write_read_roundtrip_tiff(tmp_path):
    """Writing and reading a TIFF image preserves every array value."""
    array = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    path = tmp_path / "arr.tiff"
    write_image(path, array)
    loaded = read_image(path)
    np.testing.assert_array_equal(array, loaded)


def test_bbbc038_train_sample_has_instance_mask(tmp_path):
    """Labeled BBBC038 samples expose an instance mask with matching dimensions."""
    _create_bbbc_sample(tmp_path, "stage1_train", "train-1", labeled=True)
    dataset = BBBC038Dataset(tmp_path / "stage1_train")
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample.image.ndim in (2, 3)
    assert sample.target is not None
    assert sample.target.shape == sample.image.shape[:2]
    assert sample.metadata["n_instances"] == int(sample.target.max())


def test_bbbc038_test_sample_has_no_mask(tmp_path):
    """Unlabeled BBBC038 samples expose no target and report zero instances."""
    _create_bbbc_sample(tmp_path, "stage1_test", "test-1", labeled=False)
    dataset = BBBC038Dataset(tmp_path / "stage1_test")
    sample = dataset[0]
    assert sample.target is None
    assert sample.metadata["n_instances"] == 0
