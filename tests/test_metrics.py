import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_fidelity.metrics import (
    MSE,
    PSNR,
    SSIM,
    bias,
    DetectionAgreement,
    DownstreamRelevance,
    NucleusCountTask,
    ObjectDensityTask,
    SegmentationAgreement,
    detection_precision_recall_f1,
    dice_coefficient,
    instance_count_error,
    flux_ratio,
    gradient_error,
    jaccard_index,
    measurement_relative_error,
    mse,
    nrmse,
    psnr,
    ssim,
)


# Level 1 - image quality


def test_mse_and_psnr_are_zero_error_and_infinite_for_identical_images():
    """Identical images have zero MSE and infinite PSNR through both APIs."""
    image = np.random.default_rng(0).random((32, 32)).astype(np.float32)

    assert mse(image, image) == 0.0
    assert psnr(image, image) == float("inf")
    assert MSE()(image, image) == 0.0
    assert PSNR()(image, image) == float("inf")


def test_psnr_decreases_as_noise_increases():
    """PSNR decreases when the reference image receives stronger noise."""
    rng = np.random.default_rng(1)
    image = rng.random((64, 64)).astype(np.float32)

    small_noise = image + rng.normal(0, 0.01, image.shape)
    large_noise = image + rng.normal(0, 0.2, image.shape)

    assert psnr(small_noise, image) > psnr(large_noise, image)


def test_ssim_is_one_for_identical_images_and_lower_for_noisy_ones():
    """SSIM is perfect for identical images and lower for a noisy image."""
    rng = np.random.default_rng(2)
    image = rng.random((64, 64)).astype(np.float32)
    noisy = image + rng.normal(0, 0.2, image.shape).astype(np.float32)

    assert ssim(image, image) == 1.0
    assert SSIM()(image, image) == 1.0
    assert ssim(noisy, image) < 1.0


def test_common_restoration_metrics_have_expected_identity_and_offset_behavior():
    """Restoration metrics handle identical images and a constant offset as expected."""
    reference = np.arange(64, dtype=np.float32).reshape(8, 8)
    restored = reference + 1.0

    assert nrmse(reference, reference) == 0.0
    assert gradient_error(reference, reference) == 0.0
    assert gradient_error(restored, reference) == 0.0
    assert bias(restored, reference) == 1.0
    assert np.isclose(flux_ratio(restored, reference), np.sum(restored) / np.sum(reference))
    assert np.isclose(nrmse(restored, reference), 1.0 / np.ptp(reference))


# Level 2 - conservation of scientific information


def test_dice_and_jaccard_agree_on_identical_and_disjoint_masks():
    """Dice and Jaccard scores are one for equal masks and zero for disjoint masks."""
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[2:6, 2:6] = True
    mask_b = mask_a.copy()

    assert dice_coefficient(mask_a, mask_b) == 1.0
    assert jaccard_index(mask_a, mask_b) == 1.0

    disjoint = np.zeros((10, 10), dtype=bool)
    disjoint[6:9, 6:9] = True
    assert dice_coefficient(mask_a, disjoint) == 0.0
    assert jaccard_index(mask_a, disjoint) == 0.0


def test_instance_count_error_and_detection_metrics():
    """Instance count error and detection scores reflect a missed object."""
    ground_truth = np.zeros((10, 10), dtype=np.int32)
    ground_truth[0:3, 0:3] = 1
    ground_truth[6:9, 6:9] = 2

    prediction = np.zeros((10, 10), dtype=np.int32)
    prediction[0:3, 0:3] = 1  # matches instance 1 exactly, misses instance 2

    assert instance_count_error(prediction, ground_truth) == 1

    scores = detection_precision_recall_f1(prediction, ground_truth, iou_threshold=0.5)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 0.5


def test_measurement_relative_error():
    """Relative measurement error handles finite values and a zero reference."""
    assert measurement_relative_error(110, 100) == 0.1
    assert measurement_relative_error(0, 0) == 0.0
    assert measurement_relative_error(1, 0) == float("inf")


def test_segmentation_and_detection_agreement_use_provided_callables():
    """Agreement metrics evaluate the supplied segmentation and detection functions."""
    reference = np.zeros((10, 10), dtype=np.float32)
    reference[2:6, 2:6] = 1.0
    restored = reference.copy()

    def segmenter(image: np.ndarray) -> np.ndarray:
        return image > 0.5

    def detector(image: np.ndarray) -> np.ndarray:
        return (image > 0.5).astype(np.int32)

    seg_scores = SegmentationAgreement(segmenter)(restored, reference)
    assert seg_scores["dice"] == 1.0
    assert seg_scores["iou"] == 1.0

    det_scores = DetectionAgreement(detector)(restored, reference)
    assert det_scores["f1"] == 1.0


# Level 3 - relevance for the final application


def test_nucleus_count_task_counts_connected_components():
    """The nucleus-count task counts separate thresholded image components."""
    image = np.zeros((20, 20), dtype=np.float32)
    image[1:3, 1:3] = 1.0
    image[10:13, 10:13] = 1.0

    assert NucleusCountTask(threshold=0.5).analyze(image) == 2.0


def test_downstream_relevance_reports_relative_error_between_restored_and_reference():
    """Downstream relevance reports task values and their relative error."""
    reference = np.zeros((20, 20), dtype=np.float32)
    reference[1:3, 1:3] = 1.0
    reference[10:13, 10:13] = 1.0

    restored = reference.copy()
    restored[10:13, 10:13] = 0.0  # restoration destroys one structure

    result = DownstreamRelevance(NucleusCountTask(threshold=0.5))(restored, reference)

    assert result["restored_value"] == 1.0
    assert result["reference_value"] == 2.0
    assert result["relative_error"] == 0.5


def test_object_density_task_reports_foreground_fraction():
    """The object-density task returns the fraction of foreground pixels."""
    image = np.zeros((10, 10), dtype=np.float32)
    image[:5, :] = 1.0

    assert ObjectDensityTask(threshold=0.5).analyze(image) == 0.5
