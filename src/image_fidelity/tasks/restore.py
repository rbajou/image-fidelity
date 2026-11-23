"""Restoration task handler."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from image_fidelity.datasets import load_samples, sample_id, seed_component
from image_fidelity.benchmark import write_results_tables
from image_fidelity.io.naming import safe_name
from image_fidelity.metrics import select_group_columns
from image_fidelity.restoration.evaluation import RestorationEvaluator
from image_fidelity.metrics import mse
from image_fidelity.visualization import save_before_after, save_run_figures

from .common import NON_GROUP_COLUMNS, degradation_columns, image_sample_count


def run_restore(config: DictConfig) -> Path:
    """Degrade, restore, evaluate, and optionally persist each selected image."""
    _, samples = load_samples(config)
    restorer = instantiate(config.method)
    degradation = instantiate(config.degradation)
    evaluator = RestorationEvaluator(data_range=float(config.metrics.get("data_range", 1.0)))
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_name = str(config.method._target_).rsplit(".", 1)[-1]
    method_parameters = OmegaConf.to_container(config.method, resolve=True)
    method_parameters.pop("_target_", None)
    degradation_info = degradation_columns(config)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(f"image_fidelity.restore.{output_dir.name}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logs_dir / "restore.log", encoding="utf-8")
    logger.addHandler(handler)
    records = []
    metric_columns: set[str] = {"runtime"}
    preview: tuple[list, list, list] = ([], [], [])
    preview_ids: list[str] = []
    image_samples = image_sample_count(config)
    try:
        logger.info("Starting restore %s", output_dir.name)
        for index, sample in enumerate(samples):
            seed_component(degradation, int(config.seed) + index)
            observed = degradation(sample.image)
            result = evaluator.evaluate(
                restorer,
                observed,
                sample.image,
                experiment_id=index + 1,
                image_id=sample_id(sample, index),
                seed=int(config.seed) + index,
                method_name=method_name,
                parameters=method_parameters,
            )
            available_metrics = {name.lower(): (name, value) for name, value in result.metrics.items()}
            available_metrics["mse"] = ("mse", mse(result.reconstruction, sample.image))
            selected_metrics = {}
            for metric_name in config.metrics.get("names", []):
                normalized_name = str(metric_name).lower()
                if normalized_name not in available_metrics:
                    raise ValueError(f"Unsupported restoration metric {metric_name!r}")
                column, value = available_metrics[normalized_name]
                selected_metrics[column] = value
            metric_columns.update(selected_metrics)
            if bool(config.metrics.get("compute_runtime", True)):
                selected_metrics["runtime"] = result.metrics["runtime"]
            if len(preview[0]) < image_samples:
                preview[0].append(sample.image)
                preview[1].append(observed)
                preview[2].append(result.reconstruction)
                preview_ids.append(sample_id(sample, index))
            records.append(
                {
                    "experiment_id": index + 1,
                    "image_id": sample_id(sample, index),
                    "seed": int(config.seed) + index,
                    "method": method_name,
                    **selected_metrics,
                    **method_parameters,
                    **degradation_info,
                }
            )
            logger.info(
                "image=%s method=%s runtime=%.6f",
                sample_id(sample, index),
                method_name,
                selected_metrics.get("runtime", result.metrics["runtime"]),
            )
            if bool(config.task.save.get("predictions", False)):
                prediction_path = output_dir / "predictions" / f"{safe_name(sample_id(sample, index))}.npy"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prediction_path, result.reconstruction)

        if preview[0]:
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)
            save_before_after(images_dir / "samples.png", *preview, titles=preview_ids)

        if bool(config.task.save.get("results", False)):
            group_columns = select_group_columns(records, exclude=(*NON_GROUP_COLUMNS, *metric_columns))
            write_results_tables(records, output_dir / "results", group_by=("method", *group_columns))
        if records and bool(config.task.save.get("figures", True)):
            save_run_figures(records, output_dir / "figures")
        logger.info("Completed restore with %d result rows", len(records))
    except Exception:
        logger.exception("Restore failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return output_dir
