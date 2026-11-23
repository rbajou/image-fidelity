"""Comparison of an instance segmentation against ground truth (e.g. BBBC038)."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..metrics.segmentation import dice_coefficient, jaccard_index

DSB_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU matrix (n_pred x n_gt) from a contingency table; label 0 is ignored."""
    pred_ids, pred_inv = np.unique(pred, return_inverse=True)
    gt_ids, gt_inv = np.unique(gt, return_inverse=True)
    inter = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    np.add.at(inter, (pred_inv.ravel(), gt_inv.ravel()), 1)
    pred_area = inter.sum(axis=1, keepdims=True)
    gt_area = inter.sum(axis=0, keepdims=True)
    iou = inter / np.maximum(pred_area + gt_area - inter, 1)
    return iou[pred_ids != 0][:, gt_ids != 0]


def _count_matches(iou: np.ndarray, threshold: float) -> int:
    """Size of a maximum one-to-one matching of pairs whose IoU exceeds `threshold`.

    A bare threshold comparison can count an object twice: with an inclusive IoU
    threshold of 0.5, a predicted object merging two equal objects reaches exactly 0.5 with each.
    """
    candidates = iou > threshold
    if not candidates.any():
        return 0
    rows, cols = linear_sum_assignment(candidates.astype(float), maximize=True)
    return int(candidates[rows, cols].sum())


def evaluate_instances(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Iterable[float] = DSB_THRESHOLDS,
) -> Dict[str, float]:
    """Per-image metrics: binary-mask Dice/IoU, count, P/R/F1 at IoU=0.5 and mean AP (DSB 2018 style).

    Each predicted and each ground-truth object is matched at most once; the
    matching is a maximum one-to-one assignment, not a greedy heuristic.
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {ground_truth.shape}")
    iou = _iou_matrix(prediction, ground_truth)
    n_pred, n_gt = iou.shape

    def prf(threshold: float) -> tuple[float, float, float, float]:
        tp = _count_matches(iou, threshold) if iou.size else 0
        fp, fn = n_pred - tp, n_gt - tp
        denom = tp + fp + fn
        ap = tp / denom if denom else 1.0
        precision = tp / n_pred if n_pred else float(n_gt == 0)
        recall = tp / n_gt if n_gt else float(n_pred == 0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1, ap

    precision, recall, f1, _ = prf(0.5 - 1e-9)  # IoU >= 0.5
    mean_ap = float(np.mean([prf(t)[3] for t in thresholds]))
    return {
        "dice": dice_coefficient(prediction > 0, ground_truth > 0),
        "iou": jaccard_index(prediction > 0, ground_truth > 0),
        "n_pred": n_pred,
        "n_gt": n_gt,
        "count_error": abs(n_pred - n_gt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": mean_ap,
    }
