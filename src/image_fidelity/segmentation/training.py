"""Supervised training of binary segmentation networks."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from image_fidelity.datasets import Sample, load_samples, sample_id, seed_component
from image_fidelity.metrics.segmentation import dice_coefficient, jaccard_index
from image_fidelity.nn.torch_utils import to_tensor
from image_fidelity.segmentation.evaluation import evaluate_instances
from image_fidelity.segmentation.nn import TorchSegmenter
from image_fidelity.training.engine import SeededSampler, fit, setup_training
from image_fidelity.visualization import plot_training_curves

LOGGER = logging.getLogger(__name__)


class SegmentationPatchDataset(Dataset):
    """Aligned random image/mask crops, with deterministic validation crops."""

    def __init__(self, samples: list[Sample], degradation, patch_size: int, train: bool, seed: int):
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        for index, sample in enumerate(samples):
            if sample.target is None:
                raise ValueError(f"Missing segmentation mask for {sample_id(sample, index)}")
            if sample.target.ndim != 2 or sample.target.shape != sample.image.shape[:2]:
                raise ValueError(f"Invalid segmentation mask shape for {sample_id(sample, index)}")
        self.samples, self.degradation = samples, degradation
        self.patch_size, self.train, self.seed = patch_size, train, seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item):
        if self.train:
            index, sample_seed = item
            rng = np.random.default_rng([self.seed, sample_seed])
        else:
            index = item
            rng = np.random.default_rng([self.seed, index])
        sample = self.samples[index]
        image, mask = sample.image, sample.target
        size = self.patch_size
        pad_h, pad_w = max(size - image.shape[0], 0), max(size - image.shape[1], 0)
        image = np.pad(image, [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2), mode="edge")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")
        if self.train:
            top = int(rng.integers(0, image.shape[0] - size + 1))
            left = int(rng.integers(0, image.shape[1] - size + 1))
        else:
            top, left = (image.shape[0] - size) // 2, (image.shape[1] - size) // 2
        image = image[top:top + size, left:left + size]
        mask = mask[top:top + size, left:left + size]
        if self.train:
            turns = int(rng.integers(0, 4))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if rng.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
        degradation = copy.copy(self.degradation)
        if hasattr(degradation, "seed"):
            degradation.seed = rng
        observed = degradation(np.ascontiguousarray(image))
        return to_tensor(observed), to_tensor(mask > 0)


def _binary_metrics(logits: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    # Compare logits directly to avoid numerical overflow in sigmoid.
    foreground = logits >= np.log(threshold / (1 - threshold))
    return {"dice": dice_coefficient(foreground, target), "iou": jaccard_index(foreground, target)}


def evaluate_test(config: DictConfig, model: nn.Module, device: torch.device, output_dir: Path) -> None:
    split = str(config.task.get("test_split", "test"))
    _, samples = load_samples(config, split=split, max_samples=config.task.get("test_max_samples"))
    degradation = instantiate(config.degradation)
    segmenter = TorchSegmenter(
        model, device=str(device), threshold=float(config.task.threshold),
        connectivity=int(config.task.connectivity),
    )
    rows = []
    for index, sample in enumerate(samples):
        if sample.target is None:
            LOGGER.warning("No ground truth for %s; skipping segmentation test metrics", sample_id(sample, index))
            continue
        seed_component(degradation, int(config.seed) + index)
        prediction = segmenter(degradation(sample.image))
        rows.append({"image_id": sample_id(sample, index), **evaluate_instances(prediction, sample.target)})
    if not rows:
        LOGGER.warning("No annotated samples in %s split; segmentation test metrics unavailable", split)
        return
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image_id"}
    summary["n_images"] = len(rows)
    (output_dir / "results" / "test_metrics.json").write_text(
        json.dumps({"summary": summary, "per_image": rows}, indent=2)
    )
    LOGGER.info("Segmentation test summary (%s): %s", split, summary)


def run_train(config: DictConfig) -> Path:
    task, trainer = config.task, config.trainer
    if not 0 < float(task.threshold) < 1:
        raise ValueError("task.threshold must be strictly between 0 and 1")
    if int(task.connectivity) not in (1, 2):
        raise ValueError("task.connectivity must be 1 or 2")
    if bool(config.model.get("residual", False)) or int(config.model.out_channels) != 1:
        raise ValueError("Binary segmentation requires residual=false and out_channels=1")
    device = setup_training(config)
    seed = int(config.seed)
    _, train_samples = load_samples(config, split="train", exclusive_train=True)
    _, val_samples = load_samples(config, split="validation")
    degradation = instantiate(config.degradation)
    train_set = SegmentationPatchDataset(train_samples, degradation, int(task.patch_size), True, seed)
    val_set = SegmentationPatchDataset(val_samples, degradation, int(task.patch_size), False, seed)
    workers = int(trainer.get("num_workers", 0))
    train_loader = DataLoader(
        train_set, batch_size=int(trainer.batch_size), sampler=SeededSampler(len(train_set), seed),
        num_workers=workers, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=int(trainer.batch_size), num_workers=workers)
    model, history = fit(
        config, train_loader, val_loader, device, nn.BCEWithLogitsLoss(),
        lambda prediction, target: _binary_metrics(prediction, target, float(task.threshold)),
    )
    output_dir = Path(config.paths.output_dir)
    if bool(task.get("save_figures", True)) and history:
        plot_training_curves(history, output_dir / "figures", labels={"loss": "Loss (BCE)", "dice": "Dice", "iou": "IoU"})
    if bool(task.get("run_test", True)):
        evaluate_test(config, model, device, output_dir)
    return output_dir
