"""Restoration-evaluation metrics, organized in three levels of increasing task relevance.

Level 1 (`restoration`): pixel-wise image quality (MSE, PSNR, SSIM, NRMSE, gradient error, bias, flux ratio).
Level 2 (`segmentation`): conservation of scientific information (segmentation,
detection, quantitative-measure agreement).
Level 3 (`downstream`): relevance for the final application (structure
detection, density/measurement, anomaly identification, ...).
"""

from __future__ import annotations

from .base import Metric
from .downstream import (
    AnomalyScoreTask,
    DownstreamRelevance,
    DownstreamTask,
    NucleusCountTask,
    ObjectDensityTask,
)
from .results import EvaluationResult
from .restoration import MSE, PSNR, SSIM, bias, flux_ratio, gradient_error, mae, mse, nrmse, psnr, ssim
from .statistics import select_group_columns, summarize_results
from .segmentation import (
    DetectionAgreement,
    SegmentationAgreement,
    detection_precision_recall_f1,
    dice_coefficient,
    instance_count_error,
    jaccard_index,
    measurement_relative_error,
)

__all__ = [
    "Metric",
    # Result tables
    "EvaluationResult",
    "select_group_columns",
    "summarize_results",
    # Level 1 - image quality
    "mae",
    "mse",
    "psnr",
    "ssim",
    "nrmse",
    "gradient_error",
    "bias",
    "flux_ratio",
    "MSE",
    "PSNR",
    "SSIM",
    # Level 2 - scientific information conservation
    "dice_coefficient",
    "jaccard_index",
    "instance_count_error",
    "detection_precision_recall_f1",
    "measurement_relative_error",
    "SegmentationAgreement",
    "DetectionAgreement",
    # Level 3 - relevance for the final application
    "DownstreamTask",
    "NucleusCountTask",
    "ObjectDensityTask",
    "AnomalyScoreTask",
    "DownstreamRelevance",
]
