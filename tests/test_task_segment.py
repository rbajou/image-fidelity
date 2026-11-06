from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from PIL import Image

from image_fidelity.tasks import dispatch


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_segment_dispatch_writes_runs_and_summary(tmp_path):
    """The segment task saves instance masks and segmentation metric summaries."""
    data_dir = tmp_path / "data"
    sample = data_dir / "bbbc038" / "stage1_train" / "s1"
    (sample / "images").mkdir(parents=True)
    (sample / "masks").mkdir()
    yy, xx = np.mgrid[:64, :64]
    for k, cx in enumerate((20, 44)):
        disk = ((xx - cx) ** 2 + (yy - 32) ** 2 < 100).astype(np.uint8)
        Image.fromarray(disk * 255).save(sample / "masks" / f"{k}.png")
    image = np.zeros((64, 64), dtype=np.uint8)
    image[(xx - 20) ** 2 + (yy - 32) ** 2 < 100] = 200
    image[(xx - 44) ** 2 + (yy - 32) ** 2 < 100] = 200
    Image.fromarray(image).save(sample / "images" / "s1.png")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="config",
            overrides=[
                "task=segment",
                "metrics=segmentation",
                "task.split=train",
                f"data.root_dir={data_dir / 'bbbc038'}",
                f"paths.output_dir={tmp_path / 'outputs'}",
            ],
        )
    output_dir = dispatch(config)

    runs = pd.read_parquet(output_dir / "results" / "runs.parquet")
    summary = pd.read_parquet(output_dir / "results" / "summary.parquet")
    assert len(runs) == 1 and runs.loc[0, "n_gt"] == 2
    assert "dice_mean" in summary.columns
    assert (output_dir / "masks" / "s1.npy").is_file()
