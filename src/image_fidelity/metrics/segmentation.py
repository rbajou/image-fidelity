"""Level 2 - scientific information conservation: segmentation, detection and quantitative-measure agreement.

Rather than pixel fidelity, these metrics ask whether restoration preserves
the information a downstream analysis pipeline actually relies on: instance
masks (segmentation), detected objects (detection), and scalar measurements
derived from them (counts, areas, densities, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .base import Metric


def dice_coefficient(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Dice/F1 overlap between two binary (or equally-labeled) masks."""
    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)
    total = prediction.sum() + ground_truth.sum()
    if total == 0:
        return 1.0
    intersection = np.logical_and(prediction, ground_truth).sum()
    return float(2.0 * intersection / total)


def jaccard_index(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Intersection-over-union between two binary (or equally-labeled) masks."""
    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)
    union = np.logical_or(prediction, ground_truth).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(prediction, ground_truth).sum()
    return float(intersection / union)


def _instance_ids(label_mask: np.ndarray) -> np.ndarray:
    ids = np.unique(label_mask)
    return ids[ids != 0]


def instance_count_error(prediction: np.ndarray, ground_truth: np.ndarray) -> int:
    """Absolute difference in number of labeled instances (e.g. detected nuclei) between two masks."""
    return abs(len(_instance_ids(prediction)) - len(_instance_ids(ground_truth)))


def detection_precision_recall_f1(
    prediction: np.ndarray, ground_truth: np.ndarray, iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Instance-detection precision/recall/F1, greedily matching predicted to ground-truth instances at `iou_threshold`.

    `prediction` and `ground_truth` are instance-labeled masks (0 = background,
    1..N = instance ids), as produced e.g. by `BBBC038Dataset`.
    """
    pred_ids = _instance_ids(prediction)
    true_ids = _instance_ids(ground_truth)

    matched_true = set()
    true_positives = 0
    for pred_id in pred_ids:
        pred_region = prediction == pred_id
        best_iou, best_true_id = 0.0, None
        for true_id in true_ids:
            if true_id in matched_true:
                continue
            iou = jaccard_index(pred_region, ground_truth == true_id)
            if iou > best_iou:
                best_iou, best_true_id = iou, true_id
        if best_true_id is not None and best_iou >= iou_threshold:
            matched_true.add(best_true_id)
            true_positives += 1

    false_positives = len(pred_ids) - true_positives
    false_negatives = len(true_ids) - true_positives

    precision = true_positives / (true_positives + false_positives) if pred_ids.size else 1.0
    recall = true_positives / (true_positives + false_negatives) if true_ids.size else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def measurement_relative_error(predicted_value: float, true_value: float) -> float:
    """Relative error of a scalar quantitative measurement (e.g. mean object area, density) vs. ground truth."""
    if true_value == 0:
        return 0.0 if predicted_value == 0 else float("inf")
    return abs(predicted_value - true_value) / abs(true_value)


@dataclass
class SegmentationAgreement(Metric):
    """Runs a segmentation callable on both images and scores Dice/IoU agreement between the two masks.

    Useful without ground-truth annotations: measures whether restoration
    preserves the segmentation obtained from the reference image, rather than
    the pixels themselves.
    """

    segmenter: Callable[[np.ndarray], np.ndarray]

    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        predicted_mask = self.segmenter(restored)
        reference_mask = self.segmenter(reference)
        return {
            "dice": dice_coefficient(predicted_mask, reference_mask),
            "iou": jaccard_index(predicted_mask, reference_mask),
        }


@dataclass
class DetectionAgreement(Metric):
    """Runs an instance-detector callable on both images and scores precision/recall/F1 agreement.

    `detector` must return an instance-labeled mask (0 = background, 1..N =
    instance ids) for a given image.
    """

    detector: Callable[[np.ndarray], np.ndarray]
    iou_threshold: float = 0.5

    def __call__(self, restored: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        predicted_instances = self.detector(restored)
        reference_instances = self.detector(reference)
        return detection_precision_recall_f1(predicted_instances, reference_instances, self.iou_threshold)
