"""Task-independent training loop for PyTorch image models."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Sampler

from image_fidelity.nn.torch_utils import load_checkpoint

LOGGER = logging.getLogger(__name__)
MetricFunction = Callable[[np.ndarray, np.ndarray], dict[str, float]]


class SeededSampler(Sampler):
    """Shuffle and seed each sample independently of the data-loader worker count."""

    def __init__(self, length: int, seed: int):
        self.length = length
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        order = self._rng.permutation(self.length)
        seeds = self._rng.integers(0, 2**62, size=self.length)
        return iter(zip(order.tolist(), seeds.tolist()))


def setup_training(config: DictConfig) -> torch.device:
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(str(config.trainer.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        LOGGER.warning("MPS unavailable, falling back to CPU")
        return torch.device("cpu")
    return device


def run_epoch(model, loader, device, loss_fn, metrics: MetricFunction, optimizer=None, scaler=None, clip=None, amp=False):
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    count = 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(inputs)
                loss = loss_fn(output, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip:
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
            predictions = output.detach().float().cpu().numpy()
            references = targets.detach().float().cpu().numpy()
            batch = {"loss": float(loss.detach()) * inputs.shape[0]}
            for prediction, reference in zip(predictions, references):
                for name, value in metrics(prediction, reference).items():
                    batch[name] = batch.get(name, 0.0) + value
            for name, value in batch.items():
                sums[name] = sums.get(name, 0.0) + value
            count += inputs.shape[0]
    if not count:
        raise ValueError("Training or validation loader is empty")
    return {name: value / count for name, value in sums.items()}


def _save(path: Path, model: nn.Module, epoch: int, best: float, history: list) -> None:
    torch.save({"model": model.state_dict(), "epoch": epoch, "best_val_loss": best, "history": history}, path)


def fit(config: DictConfig, train_loader, val_loader, device: torch.device, loss_fn, metrics: MetricFunction):
    """Train and reload the best weights, preserving the existing checkpoint format."""
    task, trainer = config.task, config.trainer
    output_dir = Path(config.paths.output_dir)
    checkpoints_dir = Path(config.paths.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model = instantiate(config.model).to(device)
    if bool(task.get("resume", False)) and task.get("checkpoint"):
        load_checkpoint(model, task.checkpoint, device)
        LOGGER.info("Loaded weights from %s", task.checkpoint)
    optimizer = instantiate(config.optimizer)(model.parameters())
    amp = bool(trainer.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None
    clip = trainer.get("gradient_clip_norm")
    early = trainer.get("early_stopping", {})
    patience = int(early.get("patience", 15)) if early.get("enabled", False) else None
    best, best_epoch, history = float("inf"), -1, []
    for epoch in range(int(trainer.epochs)):
        train_metrics = run_epoch(model, train_loader, device, loss_fn, metrics, optimizer, scaler, clip, amp)
        val_metrics = run_epoch(model, val_loader, device, loss_fn, metrics, amp=amp)
        val_loss = val_metrics["loss"]
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        LOGGER.info("epoch %d train=%s validation=%s", epoch, train_metrics, val_metrics)
        improved = val_loss < best
        if improved:
            best, best_epoch = val_loss, epoch
        _save(checkpoints_dir / "last.pt", model, epoch, best, history)
        if improved and bool(task.get("save_best", True)):
            _save(checkpoints_dir / "best.pt", model, epoch, best, history)
        if patience is not None and epoch - best_epoch >= patience:
            LOGGER.info("Early stopping at epoch %d", epoch)
            break
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    best_path = checkpoints_dir / "best.pt"
    if history and bool(task.get("save_best", True)) and best_path.exists():
        load_checkpoint(model, best_path, device)
    return model, history
