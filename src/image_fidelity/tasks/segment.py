"""Segmentation task handler: segment images and score them against ground-truth instance masks."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from image_fidelity.benchmark import write_results_tables
from image_fidelity.datasets import load_samples, sample_id, seed_component
from image_fidelity.evaluation import select_group_columns
from image_fidelity.segmentation.evaluation import evaluate_instances
from image_fidelity.visualization import save_run_figures, save_segmentation_preview

from .common import NON_GROUP_COLUMNS, degradation_columns, image_sample_count, safe_name


def run_segment(config: DictConfig) -> Path:
    """Segment selected images and evaluate those with ground-truth masks."""
    _, samples = load_samples(config)
    segmenter = instantiate(config.method)
    degradation = instantiate(config.degradation)
    degradation_info = degradation_columns(config)
    method_name = str(config.method._target_).rsplit(".", 1)[-1]
    parameters = OmegaConf.to_container(config.method, resolve=True)
    parameters.pop("_target_", None)
    iou_thresholds = config.metrics.get("thresholds")
    metric_names = [str(name).lower() for name in config.metrics.get("names", [])]

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(f"image_fidelity.segment.{output_dir.name}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logs_dir / "segment.log", encoding="utf-8")
    logger.addHandler(handler)

    records = []
    image_samples = image_sample_count(config)
    previews_saved = 0
    metric_columns: set[str] = {"runtime"}
    try:
        logger.info("Starting segment %s", output_dir.name)
        for index, sample in enumerate(samples):
            image_id = sample_id(sample, index)
            seed_component(degradation, int(config.seed) + index)
            observed = degradation(sample.image)
            start = time.perf_counter()
            prediction = segmenter(observed)
            runtime = time.perf_counter() - start

            scores = None
            if sample.target is None:
                logger.warning("No ground truth for %s; segmentation metrics are unavailable", image_id)
            else:
                scores = evaluate_instances(
                    prediction, sample.target, **({"thresholds": iou_thresholds} if iou_thresholds else {})
                )
                unknown = [name for name in metric_names if name not in scores]
                if unknown:
                    raise ValueError(f"Unsupported segmentation metric(s) {unknown}; available: {sorted(scores)}")
                selected = {name: scores[name] for name in metric_names} if metric_names else dict(scores)
                metric_columns.update(selected)
                if bool(config.metrics.get("compute_runtime", True)):
                    selected["runtime"] = runtime

                records.append(
                    {
                        "experiment_id": index + 1,
                        "image_id": image_id,
                        "seed": int(config.seed),
                        "method": method_name,
                        **selected,
                        **parameters,
                        **degradation_info,
                    }
                )
            logger.info("image=%s method=%s runtime=%.6f", image_id, method_name, runtime)
            if bool(config.task.save.get("masks", False)):
                mask_path = output_dir / "masks" / f"{safe_name(image_id)}.npy"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mask_path, prediction)
            if previews_saved < image_samples:
                save_segmentation_preview(
                    output_dir / "images" / f"{safe_name(image_id)}.png",
                    sample.image,
                    observed,
                    prediction,
                    sample.target,
                    image_id=image_id,
                    method=method_name,
                    degradation=degradation_info,
                    scores=scores,
                )
                previews_saved += 1

        if not records:
            logger.warning("No samples with ground-truth masks; metric tables and figures will not be generated")
        if records and bool(config.task.save.get("results", False)):
            group_columns = select_group_columns(records, exclude=(*NON_GROUP_COLUMNS, *metric_columns))
            write_results_tables(records, output_dir / "results", group_by=("method", *group_columns))
        if records and bool(config.task.save.get("figures", True)):
            save_run_figures(records, output_dir / "figures")
        logger.info("Completed segment with %d images and %d evaluated result rows", len(samples), len(records))
    except Exception:
        logger.exception("Segment failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return output_dir
