import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn

from image_fidelity.datasets import Sample
from image_fidelity.degradation import IdentityDegradation
from image_fidelity.nn.models import UNet
from image_fidelity.restoration.nn.models import UNet as LegacyUNet
from image_fidelity.segmentation.nn import TorchSegmenter
from image_fidelity.segmentation.training import SegmentationPatchDataset
from image_fidelity.tasks import dispatch
from image_fidelity.cli.cli import translate_arguments


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def _make_dataset(root):
    yy, xx = np.mgrid[:24, :24]
    for split, count in (("stage1_train", 6), ("stage1_test", 2)):
        for i in range(count):
            folder = root / split / f"s{i}"
            (folder / "images").mkdir(parents=True)
            (folder / "masks").mkdir()
            mask = ((xx - 8 - i % 3) ** 2 + (yy - 12) ** 2 < 20).astype(np.uint8)
            Image.fromarray(mask * 255).save(folder / "images" / f"s{i}.png")
            Image.fromarray(mask * 255).save(folder / "masks" / "mask.png")


def test_shared_unet_is_backward_compatible():
    assert LegacyUNet is UNet
    old_model = LegacyUNet(base_channels=4, depth=2, residual=True)
    new_model = UNet(base_channels=4, depth=2, residual=True)
    new_model.load_state_dict(old_model.state_dict())
    assert translate_arguments(["segment", "method=unet_segmenter"]) == [
        "task=segment", "method=unet_segmenter"
    ]
    assert translate_arguments(["train", "trainer.epochs=1"]) == ["task=train_restoration", "trainer.epochs=1"]
    assert translate_arguments(["benchmark_restore", "method=gaussian,nlm"]) == [
        "task=benchmark_restore",
        "+method@task.methods.gaussian=gaussian",
        "+method@task.methods.nlm=nlm",
    ]
    assert translate_arguments(["benchmark", "method=gaussian,nlm"]) == [
        "task=benchmark_restore",
        "+method@task.methods.gaussian=gaussian",
        "+method@task.methods.nlm=nlm",
    ]
    assert translate_arguments(["benchmark_segment"]) == ["task=benchmark_segment"]
    assert translate_arguments(["benchmark_segment", "method=watershed,unet_segmenter"]) == [
        "task=benchmark_segment",
        "+method@task.methods.watershed=watershed",
        "+method@task.methods.unet_segmenter=unet_segmenter",
    ]


def test_patch_dataset_keeps_image_and_mask_aligned():
    image = np.zeros((23, 19), dtype=np.float32)
    image[3:16, 5:12] = 1
    sample = Sample(image, (image * 7).astype(np.int32), {"sample_id": "sample"})
    dataset = SegmentationPatchDataset([sample], IdentityDegradation(), 16, True, 42)
    for seed in range(10):
        x, y = dataset[(0, seed)]
        assert torch.equal(x, y)
    val = SegmentationPatchDataset([sample], IdentityDegradation(), 32, False, 42)
    x, y = val[0]
    assert x.shape == y.shape == (1, 32, 32)
    assert torch.equal(x, y)
    with pytest.raises(ValueError, match="Missing segmentation mask"):
        SegmentationPatchDataset([Sample(image)], IdentityDegradation(), 16, True, 42)
    with pytest.raises(ValueError, match="Invalid segmentation mask"):
        SegmentationPatchDataset([Sample(image, np.zeros((5, 5)))], IdentityDegradation(), 16, True, 42)


def test_segmenter_threshold_and_connectivity():
    image = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
    model = nn.Sequential(nn.Conv2d(1, 1, 1))
    with torch.no_grad():
        model[0].weight.fill_(20)
        model[0].bias.fill_(-10)
    four = TorchSegmenter(model, connectivity=1)(image)
    eight = TorchSegmenter(model, connectivity=2)(image)
    assert four.shape == image.shape and four.dtype == np.int32
    assert four.max() == 3 and eight.max() == 1
    assert np.array_equal(eight > 0, image > 0)
    assert not TorchSegmenter(model, threshold=0.99999)(image).any()
    with pytest.raises(ValueError, match="threshold"):
        TorchSegmenter(model, threshold=1)
    with pytest.raises(ValueError, match="residual"):
        TorchSegmenter(UNet(residual=True))
    with pytest.raises(ValueError, match="logits"):
        TorchSegmenter(nn.Conv2d(1, 2, 1))(image)


def test_segmentation_training_and_inference_end_to_end(tmp_path):
    data = tmp_path / "data"
    _make_dataset(data)
    train_output = tmp_path / "train"
    common = [
        f"data.root_dir={data}", "data.val_fraction=0.34", "project.device=cpu",
        "model.base_channels=4", "model.depth=2", "model.normalization=none",
    ]
    config = _compose(
        "experiment=train_segmentation_unet", *common, "trainer.device=cpu",
        "trainer.epochs=1", "trainer.batch_size=2", "trainer.num_workers=0",
        "task.patch_size=16", "task.save_figures=true", f"paths.output_dir={train_output}",
        f"paths.checkpoints_dir={train_output / 'checkpoints'}",
    )
    dispatch(config)
    checkpoint = train_output / "checkpoints" / "best.pt"
    assert checkpoint.is_file() and (train_output / "checkpoints" / "last.pt").is_file()
    torch.manual_seed(int(config.seed))
    initial = UNet(base_channels=4, depth=2, normalization="none")
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    assert any(not torch.equal(trained[name], value) for name, value in initial.state_dict().items())
    history = json.loads((train_output / "results" / "history.json").read_text())
    assert {"train_loss", "val_loss", "train_dice", "val_iou"} <= history[0].keys()
    assert {path.name for path in (train_output / "figures").glob("*.png")} == {
        "loss_vs_epoch.png", "dice_vs_epoch.png", "iou_vs_epoch.png"
    }
    results = json.loads((train_output / "results" / "test_metrics.json").read_text())
    assert results["summary"]["n_images"] == 2
    assert {"dice", "iou", "n_pred", "n_gt", "ap"} <= results["per_image"][0].keys()

    resumed_output = tmp_path / "resumed"
    resume_config = _compose(
        "experiment=train_segmentation_unet", *common, "trainer.device=cpu",
        "trainer.epochs=1", "trainer.batch_size=2", "trainer.num_workers=0", "optimizer.lr=0",
        "task.patch_size=16", "task.save_figures=false", "task.run_test=false",
        "task.resume=true", f"task.checkpoint={checkpoint}", f"paths.output_dir={resumed_output}",
        f"paths.checkpoints_dir={resumed_output / 'checkpoints'}",
    )
    dispatch(resume_config)
    resumed = torch.load(resumed_output / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(value, resumed[name]) for name, value in trained.items())

    output = tmp_path / "segment"
    config = _compose(
        "experiment=segment_unet", *common, f"method.checkpoint={checkpoint}",
        "task.save.images=false", "task.save.figures=false", f"paths.output_dir={output}",
    )
    dispatch(config)
    assert len(list((output / "masks").glob("*.npy"))) == 2
    assert (output / "results" / "runs.parquet").is_file()
    assert (output / "results" / "summary.parquet").is_file()


def test_segmentation_training_test_split_without_annotations(tmp_path, caplog):
    data = tmp_path / "data"
    _make_dataset(data)
    for mask in (data / "stage1_test").glob("*/masks/mask.png"):
        mask.unlink()
    output = tmp_path / "out"
    config = _compose(
        "experiment=train_segmentation_unet", f"data.root_dir={data}", "data.val_fraction=0.34",
        "project.device=cpu", "trainer.device=cpu", "trainer.epochs=1", "trainer.batch_size=2",
        "trainer.num_workers=0", "task.patch_size=16", "task.save_figures=false",
        "model.base_channels=4", "model.depth=2", "model.normalization=none",
        f"paths.output_dir={output}", f"paths.checkpoints_dir={output / 'checkpoints'}",
    )
    dispatch(config)
    assert (output / "checkpoints" / "best.pt").is_file()
    assert not (output / "results" / "test_metrics.json").exists()
    assert "segmentation test metrics unavailable" in caplog.text


def test_restoration_training_still_uses_shared_engine(tmp_path):
    data = tmp_path / "data"
    _make_dataset(data)
    output = tmp_path / "restore_train"
    config = _compose(
        "experiment=train_restoration_unet", f"data.root_dir={data}", "data.val_fraction=0.34",
        "trainer.device=cpu", "trainer.epochs=1", "trainer.batch_size=2",
        "trainer.num_workers=0", "task.patch_size=16", "task.save_figures=false",
        "model.base_channels=4", "model.depth=2", "model.normalization=none",
        f"paths.output_dir={output}", f"paths.checkpoints_dir={output / 'checkpoints'}",
    )
    dispatch(config)
    assert (output / "checkpoints" / "best.pt").is_file()
    history = json.loads((output / "results" / "history.json").read_text())
    assert {"train_loss", "val_loss", "train_psnr", "val_ssim"} <= history[0].keys()
    results = json.loads((output / "results" / "test_metrics.json").read_text())
    assert results["summary"]["n_images"] == 2
