"""Comparaison d'une segmentation d'instances à une vérité terrain (ex. BBBC038)."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np

from ..metrics.segmentation import dice_coefficient, jaccard_index

DSB_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Matrice IoU (n_pred x n_gt) via table de contingence ; les étiquettes 0 sont ignorées."""
    pred_ids, pred_inv = np.unique(pred, return_inverse=True)
    gt_ids, gt_inv = np.unique(gt, return_inverse=True)
    inter = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    np.add.at(inter, (pred_inv.ravel(), gt_inv.ravel()), 1)
    pred_area = inter.sum(axis=1, keepdims=True)
    gt_area = inter.sum(axis=0, keepdims=True)
    iou = inter / np.maximum(pred_area + gt_area - inter, 1)
    return iou[pred_ids != 0][:, gt_ids != 0]


def evaluate_instances(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Iterable[float] = DSB_THRESHOLDS,
) -> Dict[str, float]:
    """Métriques par image : Dice/IoU du masque binaire, comptage, P/R/F1 à IoU=0.5 et AP moyenne (style DSB 2018).

    Pour un seuil d'IoU >= 0.5, l'appariement prédiction/vérité terrain est
    unique, donc le calcul est exact et sans appariement glouton.
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Formes différentes : {prediction.shape} vs {ground_truth.shape}")
    iou = _iou_matrix(prediction, ground_truth)
    n_pred, n_gt = iou.shape

    def prf(threshold: float) -> tuple[float, float, float, float]:
        tp = int((iou > threshold).sum()) if iou.size else 0
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


def evaluate_dataset(
    dataset,
    segmenter: Callable[[np.ndarray], np.ndarray],
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Applique `segmenter` à chaque échantillon annoté et renvoie métriques par image + moyennes.

    `dataset` fournit des `Sample` avec `target` = masque d'instances (cf. `BBBC038Dataset`).
    Les échantillons sans vérité terrain sont ignorés.
    """
    per_image: Dict[str, Dict[str, float]] = {}
    for i in range(len(dataset) if max_samples is None else min(max_samples, len(dataset))):
        sample = dataset[i]
        if sample.target is None:
            continue
        pred = segmenter(sample.image)
        per_image[sample.metadata["sample_id"]] = evaluate_instances(pred, sample.target)
    keys = next(iter(per_image.values())).keys() if per_image else []
    mean = {k: float(np.mean([m[k] for m in per_image.values()])) for k in keys}
    return {"per_image": per_image, "mean": mean}
