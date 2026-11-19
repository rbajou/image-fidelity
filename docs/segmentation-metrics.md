# Segmentation metrics

This guide explains the scores written by the segmentation workflow (`imfid experiment=segment_watershed`, `imfid experiment=segment_unet`): what each metric measures, what it is useful for, and how it can mislead you. For restoration scores, see [Restoration metrics](restoration-metrics.md). For general advice on result tables, see [Reading the results](reading-results.md).

## What is being compared

The segmentation task degrades each image, runs a segmenter on the degraded image, and compares the predicted **instance labels** with the ground-truth annotation:

```text
image ──► degradation ──► segmenter ──► predicted instances ─┐
                                                              ├─► metrics
ground-truth instance mask ──────────────────────────────────┘
```

An instance mask assigns background `0` and one positive integer to each object, for example each nucleus. The metrics ignore label values: only overlaps between regions matter, so predicted label `3` can match ground-truth label `17`.

Things to keep in mind:

- The score reflects the **combined** effect of degradation and segmenter. There is no restoration step in this task.
- Only images with ground-truth masks are scored. Others are still segmented, but they produce no metrics. With BBBC038, the official test images have no masks.
- Scores are computed per image, then averaged over images (see [Aggregation](#aggregation-and-comparison)).
- The annotation is the reference, but it is a human product. Boundaries, missed objects and inconsistent decisions in the annotation limit what any score can mean.

## Quick reference

| Column | Level | Measures | Better |
| --- | --- | --- | --- |
| `dice`, `iou` | Pixels (foreground vs background) | Overlap of the union of all objects | Higher |
| `n_pred`, `n_gt` | Objects | Number of predicted and annotated objects | n/a |
| `count_error` | Objects | Absolute difference in object counts | Lower |
| `precision`, `recall`, `f1` | Objects, IoU ≥ 0.5 | Proportion of correct detections | Higher |
| `ap` | Objects, IoU 0.5 to 0.95 | Detection quality with increasingly strict overlap | Higher |
| `runtime` | n/a | Time spent segmenting one image | Lower |

Restrict the columns with `metrics.names`. For example `'metrics.names=[dice,f1,ap]'`. Names are case-insensitive and an unknown name raises an error listing the available ones.

## Pixel-level overlap: `dice` and `iou`

Both compare the **foreground**, that is, every pixel with a label greater than zero, ignoring which object it belongs to:

```text
dice = 2 · |P ∩ G| / (|P| + |G|)
iou  = |P ∩ G| / |P ∪ G|
```

where `P` is the predicted foreground and `G` the ground-truth foreground. `dice` is the Dice coefficient, also known as the F1 score of the pixels. `iou` is the intersection over union, or Jaccard index. They are monotonically related: `iou = dice / (2 − dice)`, so they rank results identically, and `iou` is always the lower of the two.

*Interest.* They are simple and robust, and answer "did the segmenter find the right regions?". They are the natural quality measure for the binary foreground output of a U-Net, and the training history of the segmentation U-Net also tracks them.

*Limits.* They are blind to the object structure:

- Two touching nuclei predicted as one blob, or one nucleus split in two, can leave the foreground unchanged, so the score stays high even though the objects are wrong. This is common for connected-component instances, which merge touching objects.
- Large objects dominate. Missing a few small nuclei barely changes the score.
- They are tolerant of small boundary errors on large objects and very sensitive on small ones.
- When both masks are empty, the score is `1`.

Use them with the object-level metrics below.

## Counting objects: `n_pred`, `n_gt`, `count_error`

`n_pred` and `n_gt` are the numbers of predicted and annotated instances. `count_error` is their absolute difference:

```text
count_error = |n_pred − n_gt|
```

*Interest.* In many biological analyses, the number of objects is the final quantity of interest, for example nuclei per field, and a count is easy to understand.

*Limits.*

- It does not say whether the objects are the right ones. A missed nucleus and a spurious detection cancel, giving a count error of 0.
- It is absolute rather than relative. An error of 5 is minor on an image with 200 nuclei and serious with 10, and images with more nuclei tend to have larger errors.
- The sign is lost. To check for systematic over- or under-segmentation, compute `n_pred − n_gt` from the two columns in `runs.parquet`.

## Detecting objects: `precision`, `recall`, `f1`

These metrics treat segmentation as detection. A predicted object counts as a **true positive** when it overlaps a ground-truth object with an IoU of at least 0.5. Each ground-truth object can be matched only once.

```text
precision = TP / n_pred    (how many detections are correct)
recall    = TP / n_gt      (how many true objects were found)
f1        = 2 · precision · recall / (precision + recall)
```

- Low precision with high recall: many false detections, typically over-segmentation or noise detected as objects.
- High precision with low recall: objects are missed, for example dim or small nuclei, or merged objects.
- `f1` summarises both with a harmonic mean.

*Interest.* They work at the level of objects, so they penalise merged and split nuclei, unlike `dice`. They separate false alarms from misses, which makes them more informative than the count error.

*Limits.*

- The 0.5 threshold is fixed. A prediction that covers a bit more than half of a nucleus counts the same as a perfect one, so boundary quality is barely measured.
- Small objects are fragile: a shift of a pixel or two can bring the IoU below 0.5.
- Matching is all or nothing. A near miss earns nothing.
- Edge cases: with no predicted objects, precision is `1` if there are no ground-truth objects either, otherwise `0`. With no ground-truth objects, recall is `1` if there are no predictions, otherwise `0`.

## Strict detection quality: `ap`

`ap` follows the convention of the 2018 Data Science Bowl, the competition behind BBBC038. At each IoU threshold `t` from 0.50 to 0.95 in steps of 0.05, it computes

```text
TP(t) / (TP(t) + FP(t) + FN(t))
```

for the image, where FP are unmatched predictions and FN unmatched annotations, and then averages over the ten thresholds.

Reading it:

- A segmentation that detects all objects but with imprecise boundaries loses score as the threshold increases, so `ap` is clearly lower than `f1`.
- It never exceeds `f1`. A large gap between the two indicates detections that are correct but poorly delineated.
- It reaches 1 only if all objects are found with an IoU close to 1 at every threshold.

*Interest.* It rewards accurate boundaries, which `f1` at 0.5 does not, and gives a single number per image that follows the competition's metric.

*Limits.*

- It is **not** the average precision of the COCO or Pascal VOC benchmarks. There are no confidence scores, no ranking of detections and no precision-recall curve. Do not compare values with papers that use those definitions.
- It is strict (an overlap must be strictly above the threshold), so values are low and hard to interpret without a baseline.
- The thresholds can be changed with `metrics.thresholds`, which then changes what `ap` means.
- A perfect result for an image without any object in either mask is `1`.

## Runtime

`runtime` is the time spent in the segmenter for one image, in seconds. The degradation, the metrics and file output are excluded. It is a single measurement and depends on your hardware, so compare methods only within the same session and on the same machine.

## Reading metrics together

| Pattern | Likely interpretation |
| --- | --- |
| High `dice`, low `f1` or `ap` | The foreground is right but objects are merged, split or poorly delineated. |
| High `f1`, much lower `ap` | Objects are detected but boundaries are imprecise. |
| High precision, low recall | Missed objects: small, dim or merged. |
| Low precision, high recall | Spurious objects: noise detected as nuclei, or over-splitting. |
| Low `count_error` with low `f1` | Counts match by compensation, not because the objects are correct. |
| Large spread between images | Performance depends on the image type, which is expected for diverse datasets like BBBC038. |

## Aggregation and comparison

`runs.parquet` has one row per annotated image, and `summary.parquet` the number of images `n` with the mean and standard deviation of each metric (`<metric>_mean`, `<metric>_std`) per group, where the standard deviation uses the population formula.

- Every image has the same weight, whether it contains 5 or 300 nuclei. A global count of detections across all images would weigh images differently, and this is not what is reported.
- The standard deviation describes variation between images. It is not a confidence interval or a test.
- Compare segmenters on the same images, degradation and seed, and look at per-image differences, not only averages.
- A single noise realisation is drawn per image. Repeat with different `seed` values to check that a conclusion is stable.

## Known limitations

- Scores depend on annotation quality and on the annotation convention, for example how touching nuclei are split.
- The image-level scores cannot say where the errors are. Use the saved previews, which show ground-truth contours and predicted instances, to investigate. Preview colours are arbitrary.
- The U-Net workflow produces instances by connected components of the foreground mask, so touching nuclei are merged by construction. A low object-level score does not necessarily indicate a poor foreground prediction.
- The metrics are computed on a synthetic degradation of the image and say nothing about robustness to other kinds of degradation.

## Library utilities

The `image_fidelity.metrics` package also contains functions that are **not** used by the command-line tasks: segmentation and detection agreement between a restored and a reference image (`SegmentationAgreement`, `DetectionAgreement`), relative measurement errors, and downstream-task relevance scores (`DownstreamRelevance` with nucleus count, object density and anomaly tasks). They can be used from Python. Be aware that `detection_precision_recall_f1` in `metrics.segmentation` matches objects greedily, whereas the CLI uses the exact IoU-matrix matching described here, so results can differ slightly.
