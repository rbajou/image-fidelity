from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image

from image_fidelity.tasks import dispatch


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _create_test_image(root: Path) -> None:
    image_dir = root / "bbbc038" / "stage1_test" / "sample-1" / "images"
    image_dir.mkdir(parents=True)
    image = np.arange(24 * 24, dtype=np.uint8).reshape(24, 24)
    Image.fromarray(image).save(image_dir / "sample-1.png")


def _compose_config(tmp_path: Path, *overrides: str):
    data_dir = tmp_path / "data"
    _create_test_image(data_dir)
    config_overrides = [
        "data=bbbc038_small",
        f"paths.data_dir={data_dir}",
        f"paths.output_dir={tmp_path / 'outputs'}",
        "task.max_samples=1",
        *overrides,
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=config_overrides)


def test_restore_dispatch_writes_prediction_and_metrics(tmp_path):
    config = _compose_config(
        tmp_path,
        "task=restore",
        "method=gaussian",
        "degradation=gaussian_noise",
        "degradation.sigma=0.05",
    )

    output_dir = dispatch(config)

    prediction = np.load(output_dir / "predictions" / "sample-1.npy")
    runs = pd.read_parquet(output_dir / "results" / "runs.parquet").to_dict("records")
    assert prediction.shape == (24, 24)
    assert runs[0]["image_id"] == "sample-1"
    assert runs[0]["PSNR"] > 0
    assert set(runs[0]) >= {"SSIM", "NRMSE", "grad_error", "bias", "flux_ratio", "runtime"}
    assert (output_dir / "figures" / "metric_distributions.png").is_file()
    assert (output_dir / "figures" / "metric_boxplots.png").is_file()


def test_benchmark_dispatch_runs_configured_methods(tmp_path, monkeypatch):
    from image_fidelity.benchmark import runner as benchmark_runner

    def fake_write_results_tables(records, output_dir, group_by):
        assert len(records) == 4
        runs_path = output_dir / "runs.parquet"
        summary_path = output_dir / "summary.parquet"
        runs_path.write_bytes(b"test parquet")
        summary_path.write_bytes(b"test parquet")
        return runs_path, summary_path

    monkeypatch.setattr(benchmark_runner, "write_results_tables", fake_write_results_tables)
    config = _compose_config(tmp_path, "experiment=benchmark_restore_classical")

    output_dir = dispatch(config)

    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "results" / "runs.parquet").is_file()
    assert not (output_dir / "predictions").exists()


def test_dispatch_rejects_unimplemented_task(tmp_path):
    config = _compose_config(tmp_path)
    config.task.name = "unknown"

    with pytest.raises(ValueError, match="implemented tasks: benchmark, degrade, download, plot, restore, segment, train"):
        dispatch(config)

def test_segment_dispatch_writes_runs_and_summary(tmp_path):
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
    # assert runs.loc[0, "f1"] == 1.0
    assert "dice_mean" in summary.columns
    assert (output_dir / "masks" / "s1.npy").is_file()
