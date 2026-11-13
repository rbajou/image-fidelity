"""Training and testing pipeline for learned restoration networks."""

from __future__ import annotations

import copy
import json
import logging
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from image_fidelity.metrics import mae, mse, psnr, ssim
from image_fidelity.visualization import plot_test_boxplots, plot_training_curves

from image_fidelity.datasets import load_samples, sample_id, seed_component
from image_fidelity.restoration.nn.torch_restorer import TorchRestorer, to_tensor
from image_fidelity.training.engine import SeededSampler, fit, setup_training

LOGGER = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """Random clean crops degraded on the fly; deterministic center crops when ``train=False``.

    In training mode items are ``(index, seed)`` pairs produced by :class:`SeededSampler`.
    """

    def __init__(self, images, degradation, patch_size: int, train: bool, seed: int, noise_level=None):
        # noise_level: (parameter name, low, high) to draw the level per patch; None keeps the configured level.
        self.noise_level = noise_level
        self.images = images
        self.degradation = degradation
        self.patch_size = patch_size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item):
        if self.train:
            index, sample_seed = item
            rng = np.random.default_rng([self.seed, sample_seed])
        else:
            index = item
            rng = np.random.default_rng([self.seed, index])
        image = self.images[index]
        size = self.patch_size
        pad = [(0, max(size - image.shape[0], 0)), (0, max(size - image.shape[1], 0))]
        pad += [(0, 0)] * (image.ndim - 2)
        image = np.pad(image, pad, mode="edge")
        if self.train:
            top = int(rng.integers(0, image.shape[0] - size + 1))
            left = int(rng.integers(0, image.shape[1] - size + 1))
        else:
            top, left = (image.shape[0] - size) // 2, (image.shape[1] - size) // 2
        clean = image[top : top + size, left : left + size]
        if self.train:
            clean = np.rot90(clean, int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                clean = clean[:, ::-1]
        clean = np.ascontiguousarray(clean)
        degradation = copy.copy(self.degradation)
        if self.noise_level is not None:
            name, low, high = self.noise_level
            setattr(degradation, name, float(rng.uniform(low, high)))
        if hasattr(degradation, "seed"):
            degradation.seed = rng
        noisy = degradation(clean)
        return to_tensor(noisy), to_tensor(clean)


def _noise_level(task: DictConfig):
    """Parse ``task.noise``: ``constant`` (degradation config as is) or ``variable`` (uniform in [min, max])."""
    noise = task.get("noise") or {}
    mode = str(noise.get("mode", "constant"))
    if mode == "constant":
        return None
    if mode != "variable":
        raise ValueError(f"task.noise.mode must be 'constant' or 'variable', got {mode!r}")
    low, high = noise.get("min"), noise.get("max")
    if low is None or high is None or float(low) > float(high):
        raise ValueError("task.noise.min and task.noise.max are required with min <= max for variable noise")
    return str(noise.get("param", "sigma")), float(low), float(high)


def _image_metrics(prediction: np.ndarray, reference: np.ndarray, data_range: float) -> dict[str, float]:
    """Metrics for one (C, H, W) image pair."""
    if prediction.shape[0] == 1:
        prediction, reference = prediction[0], reference[0]
    else:
        prediction, reference = np.moveaxis(prediction, 0, -1), np.moveaxis(reference, 0, -1)
    return {
        "psnr": psnr(prediction, reference, data_range),
        "ssim": ssim(prediction, reference, data_range),
        "mse": mse(prediction, reference),
        "mae": mae(prediction, reference),
    }


def test_model(config: DictConfig, model: nn.Module, device: torch.device, output_dir: Path) -> dict:
    """Evaluate ``model`` on the test split with the configured degradation."""
    _, samples = load_samples(config, split="test", max_samples=config.task.get("test_max_samples"))
    degradation = instantiate(config.degradation)
    restorer = TorchRestorer(model, device=str(device))
    data_range = float(config.metrics.get("data_range", 1.0))
    rows = []
    for index, sample in enumerate(samples):
        seed_component(degradation, int(config.seed) + index)
        observed = degradation(sample.image)
        start = perf_counter()
        restored = restorer(observed)
        runtime = perf_counter() - start
        ref = np.asarray(sample.image)
        rows.append(
            {
                "image_id": sample_id(sample, index),
                "psnr": psnr(restored, ref, data_range),
                "ssim": ssim(restored, ref, data_range),
                "mse": mse(restored, ref),
                "mae": mae(restored, ref),
                "runtime": runtime,
                "psnr_noisy": psnr(observed, ref, data_range),
                "ssim_noisy": ssim(observed, ref, data_range),
            }
        )
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image_id"}
    summary["n_images"] = len(rows)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "test_metrics.json").write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2))
    if bool(config.task.get("save_figures", True)):
        plot_test_boxplots(rows, output_dir / "figures")
    LOGGER.info("Test summary: %s", summary)
    return summary


def run_train(config: DictConfig) -> Path:
    task = config.task
    output_dir = Path(config.paths.output_dir)
    seed = int(config.seed)
    device = setup_training(config)
    trainer = config.trainer
    patch_size = int(task.get("patch_size", 128))
    data_range = float(config.metrics.get("data_range", 1.0))

    _, train_samples = load_samples(config, split="train", exclusive_train=True)
    _, val_samples = load_samples(config, split="validation")
    degradation = instantiate(config.degradation)
    noise_level = _noise_level(task)
    if noise_level is not None and not hasattr(degradation, noise_level[0]):
        raise ValueError(f"Degradation has no parameter {noise_level[0]!r} to vary")
    train_set = PatchDataset([s.image for s in train_samples], degradation, patch_size, True, seed, noise_level)
    val_set = PatchDataset([s.image for s in val_samples], degradation, patch_size, False, seed, noise_level)
    workers = int(trainer.get("num_workers", 0))
    batch_size = int(trainer.batch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=SeededSampler(len(train_set), seed),
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=workers)

    model, history = fit(
        config, train_loader, val_loader, device, nn.MSELoss(),
        lambda prediction, reference: _image_metrics(prediction, reference, data_range),
    )
    if bool(task.get("save_figures", True)) and history:
        plot_training_curves(history, output_dir / "figures")
    if bool(task.get("run_test", True)):
        test_model(config, model, device, output_dir)
    return output_dir
