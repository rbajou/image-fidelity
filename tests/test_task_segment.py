from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image

from image_fidelity.datasets import Sample
from image_fidelity.tasks import dispatch
from image_fidelity.tasks import segment
from image_fidelity.visualization import save_segmentation_preview


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
                "method=watershed",
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
    assert (output_dir / "images" / "s1.png").is_file()


def _sample(sample_name, annotated=True):
    yy, xx = np.mgrid[:48, :48]
    disk = (xx - 24) ** 2 + (yy - 24) ** 2 < 100
    image = np.zeros((48, 48), dtype=np.float32)
    image[disk] = 0.8
    return Sample(image=image, target=disk.astype(np.int32) if annotated else None, metadata={"sample_id": sample_name})


def test_segment_previews_and_metrics_with_mixed_annotations(compose_task_config, monkeypatch):
    samples = [_sample("unlabeled", False), _sample("annotated"), _sample("extra")]
    monkeypatch.setattr(segment, "load_samples", lambda config: (None, samples))
    previews = []
    real_save = segment.save_segmentation_preview

    def capture_preview(*args, **kwargs):
        previews.append((args, kwargs))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(segment, "save_segmentation_preview", capture_preview)
    config = compose_task_config(
        "task=segment", "method=watershed", "metrics=segmentation",
        "degradation=gaussian_noise", "degradation.sigma=0.1", "task.image_samples=2",
    )
    output = dispatch(config)

    assert sorted(p.name for p in (output / "images").glob("*.png")) == ["annotated.png", "unlabeled.png"]
    assert len(list((output / "masks").glob("*.npy"))) == 3
    assert [record["image_id"] for record in pd.read_parquet(output / "results" / "runs.parquet").to_dict("records")] == [
        "annotated", "extra"
    ]
    assert pd.read_parquet(output / "results" / "summary.parquet").loc[0, "n"] == 2
    assert previews[0][0][4] is None
    assert previews[0][1]["scores"] is None
    assert previews[1][1]["scores"]["n_gt"] == 1
    assert previews[1][1]["degradation"]["degradation_sigma"] == 0.1
    assert not np.array_equal(previews[0][0][1], previews[0][0][2])
    assert np.array_equal(previews[0][0][3], np.load(output / "masks" / "unlabeled.npy"))


@pytest.mark.parametrize(("overrides", "preview_count"), [
    ((), 1),
    (("task.save.images=false",), 0),
    (("task.image_samples=0",), 0),
])
def test_segment_without_ground_truth(compose_task_config, monkeypatch, caplog, overrides, preview_count):
    monkeypatch.setattr(segment, "load_samples", lambda config: (None, [_sample("unlabeled", False)]))
    config = compose_task_config("task=segment", "method=watershed", "metrics=segmentation", *overrides)

    output = dispatch(config)

    assert (output / "masks" / "unlabeled.npy").is_file()
    assert len(list((output / "images").glob("*.png"))) == preview_count
    assert not (output / "results").exists()
    assert not (output / "figures").exists()
    assert "No samples with ground-truth masks" in caplog.text


@pytest.mark.parametrize("annotated", [True, False])
def test_segmentation_preview_renders_instances_and_optional_ground_truth(tmp_path, monkeypatch, annotated):
    from matplotlib.figure import Figure

    sample = _sample("image", annotated)
    prediction = np.zeros((48, 48), dtype=np.int32)
    prediction[10:20, 10:20] = 42
    prediction[10:20, 20:30] = 99
    saved = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        saved.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    rgb = np.repeat(sample.image[:, :, None], 3, axis=2)
    path = save_segmentation_preview(
        tmp_path / "nested" / "preview.png", rgb, rgb, prediction, sample.target,
        image_id="image", method="WatershedSegmenter",
        degradation={"degradation": "GaussianNoise", "degradation_sigma": 0.1},
        scores={"dice": 0.9, "f1": 0.8, "n_pred": 2, "n_gt": 1} if annotated else None,
    )
    assert path.is_file()
    with Image.open(path) as image:
        assert image.size == (1600, 500)
    figure = saved[0]
    assert len(figure.axes) == 4
    assert figure.axes[2].get_title() == ("Ground truth" if annotated else "Ground truth unavailable")
    assert len(figure.axes[2].images) == (2 if annotated else 1)
    assert "degradation_sigma=0.1" in figure._suptitle.get_text()
    assert ("Dice=0.900" in figure._suptitle.get_text()) == annotated
    instance_overlay = np.asarray(figure.axes[3].images[1].get_array())
    assert instance_overlay[0, 0, 3] == 0
    assert instance_overlay[15, 15, 3] > 0
    assert not np.array_equal(instance_overlay[15, 15], instance_overlay[15, 25])
