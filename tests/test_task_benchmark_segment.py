import numpy as np
import pandas as pd
import pytest
from PIL import Image

import image_fidelity.tasks.benchmark_segment as benchmark_segment
from image_fidelity.tasks import dispatch


def _add_annotation(tmp_path):
    sample_dir = tmp_path / "data" / "bbbc038" / "stage1_test" / "sample-1"
    masks_dir = sample_dir / "masks"
    masks_dir.mkdir()
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[5:16, 7:18] = 255
    Image.fromarray(mask).save(masks_dir / "instance.png")
    return (mask > 0).astype(np.int32)


def test_segmentation_benchmark_scores_and_saves_instance_predictions(
    tmp_path, compose_task_config, monkeypatch
):
    target = _add_annotation(tmp_path)

    monkeypatch.setattr(
        benchmark_segment,
        "instantiate",
        lambda config: lambda image: target.copy(),
    )
    config = compose_task_config(
        "task=benchmark_segment",
        "task.split=test",
        "metrics=segmentation",
        "model=segmentation_unet",
        "+method@task.methods.unet=unet_segmenter",
        "task.methods.unet.checkpoint=null",
        "+method@task.methods.watershed=watershed",
        "task.save.figures=true",
        "task.save.images=false",
    )

    output_dir = dispatch(config)

    runs = pd.read_parquet(output_dir / "results" / "runs.parquet")
    assert len(runs) == 2
    assert set(runs["method"]) == {"unet", "watershed"}
    assert runs.loc[runs["method"] == "unet", "dice"].iloc[0] == 1.0
    assert runs.loc[runs["method"] == "watershed", "iou"].iloc[0] == 1.0
    assert runs.loc[runs["method"] == "unet", "method_unet_threshold"].iloc[0] == 0.5
    assert runs.loc[runs["method"] == "watershed", "method_watershed_threshold"].iloc[0] == "otsu"
    assert "threshold" not in runs.columns
    for method in ("unet", "watershed"):
        saved_prediction = next((output_dir / "predictions" / method).glob("*.npy"))
        assert np.array_equal(np.load(saved_prediction), target)
        assert (output_dir / "figures" / f"metric_distributions_{method}.png").is_file()
    assert not (output_dir / "figures" / "metric_distributions.png").exists()
    assert (output_dir / "figures" / "metric_boxplots_by_method.png").is_file()
    assert not list((output_dir / "images").glob("*.npy"))


def test_segmentation_benchmark_requires_annotations(compose_task_config):
    config = compose_task_config(
        "task=benchmark_segment",
        "task.split=test",
        "metrics=segmentation",
        "+method@task.methods.fixed=watershed",
    )

    with pytest.raises(ValueError, match="require an annotated target mask for every sample"):
        dispatch(config)
