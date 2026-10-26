"""Figures for learned-model training: per-epoch curves and test-set boxplots."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .images import _import_pyplot

CURVE_METRICS = {
    "loss": "Loss (MSE)",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "mse": "MSE",
    "mae": "MAE",
}
BOXPLOT_METRICS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "mse": "MSE", "mae": "MAE"}


def plot_training_curves(history: Sequence[Mapping[str, float]], output_dir: str | Path) -> list[Path]:
    """One figure per metric, train and validation on the same axes. ``history`` rows hold ``train_<m>``/``val_<m>``."""
    plt = _import_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] + 1 for row in history]
    paths = []
    for key, label in CURVE_METRICS.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for split in ("train", "val"):
            ax.plot(epochs, [row[f"{split}_{key}"] for row in history], marker="o", markersize=3, label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs epoch")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"{key}_vs_epoch.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_test_boxplots(per_image: Sequence[Mapping[str, float]], output_dir: str | Path) -> list[Path]:
    """One boxplot figure per metric over the per-image test distribution (rows hold ``psnr``/``ssim``/``mse``/``mae``)."""
    plt = _import_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, label in BOXPLOT_METRICS.items():
        fig, ax = plt.subplots(figsize=(3.5, 4))
        ax.boxplot([row[key] for row in per_image], tick_labels=["test"], showmeans=True)
        ax.set_ylabel(label)
        ax.set_title(f"{label} (n={len(per_image)})")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path = output_dir / f"test_{key}_boxplot.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
