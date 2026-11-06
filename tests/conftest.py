from pathlib import Path

import numpy as np
import pytest
from PIL import Image


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def compose_task_config(tmp_path):
    """Compose a Hydra task config backed by a temporary sample image."""
    data_dir = tmp_path / "data"
    image_dir = data_dir / "bbbc038" / "stage1_test" / "sample-1" / "images"
    image_dir.mkdir(parents=True)
    image = np.arange(24 * 24, dtype=np.uint8).reshape(24, 24)
    Image.fromarray(image).save(image_dir / "sample-1.png")

    def compose_config(*overrides: str):
        from hydra import compose, initialize_config_dir

        config_overrides = [
            "data=bbbc038_small",
            f"paths.data_dir={data_dir}",
            f"paths.output_dir={tmp_path / 'outputs'}",
            "task.max_samples=1",
            *overrides,
        ]
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            return compose(config_name="config", overrides=config_overrides)

    return compose_config
