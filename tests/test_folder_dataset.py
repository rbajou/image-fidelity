import numpy as np
from PIL import Image

from image_fidelity.datasets import ImageFolderDataset, load_experiment_split, open_dataset


def _make(root):
    (root / "sub").mkdir(parents=True)
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    for name in ["a.png", "b.gif", "c.jpeg", "sub/d.png", "sub/e.tif"]:
        if name.endswith("tif"):
            Image.fromarray(arr).save(root / name, format="TIFF")
        else:
            Image.fromarray(arr).save(root / name)
    (root / "notes.txt").write_text("ignored")


def test_folder_dataset_finds_mixed_formats_recursively(tmp_path):
    """Folder datasets recursively discover supported image formats only."""
    _make(tmp_path)
    dataset = open_dataset(tmp_path)
    assert isinstance(dataset, ImageFolderDataset)
    assert len(dataset) == 5
    assert {s.metadata["sample_id"] for s in dataset} == {"a", "b", "c", "sub/d", "sub/e"}


def test_load_experiment_split_without_predefined_partitions(tmp_path):
    """A folder without predefined partitions is divided into all three splits."""
    _make(tmp_path)
    split = load_experiment_split(tmp_path, val_fraction=0.2, test_fraction=0.2)
    assert len(split.train) + len(split.validation) + len(split.test) == 5
