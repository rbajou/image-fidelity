# Restoration metrics

This guide explains the scores written by `imfid restore` and `imfid benchmark`: what each one measures physically, what it is good for, and where it can mislead you. For general advice on reading result tables, see [Reading the results](reading-results.md). For instance segmentation scores, see [Segmentation metrics](segmentation-metrics.md).

## What is being compared

Each reference image is degraded synthetically, restored by a method, and compared **pixel by pixel** with the original reference:

```text
reference ──► degradation ──► observed ──► restoration ──► reconstruction
    └──────────────────────── metrics ◄──────────────────────────┘
```

Two consequences follow:

- Every score measures *agreement with the reference*, not image quality in an absolute sense. If the reference itself contains noise, as acquired microscopy images do, a method that removes that noise is penalised for departing from it. Scores have a ceiling set by the reference quality.
- Pixel-wise metrics cannot tell whether a difference matters for your analysis. Use them to rank methods under controlled conditions, then check images and, when relevant, the downstream result you care about.

## Quick reference

| Column | Measures | Better | Ideal | Unit |
| --- | --- | --- | --- | --- |
| `PSNR` | Error energy relative to the signal range | Higher | `inf` (identical) | dB |
| `SSIM` | Local similarity of mean, contrast and structure | Higher | 1 | none |
| `NRMSE` | Root-mean-square error relative to the reference range | Lower | 0 | none |
| `grad_error` | Error in edges and fine detail | Lower | 0 | none |
| `bias` | Average intensity offset | Closer to 0 | 0 | image intensity units |
| `flux_ratio` | Total intensity conservation | Closer to 1 | 1 | none |
| `runtime` | Time spent restoring one image | Lower | n/a | seconds |

`PSNR`, `SSIM`, `NRMSE`, `grad_error`, `bias` and `flux_ratio` are the defaults. A `mse` column is also available: add `mse` to `metrics.names` to request it.

## Intensity scale

Integer images are converted to floating point and divided by their dtype's maximum value, so 8-bit data lie in [0, 1]. Floating-point inputs keep their scale. `PSNR` and `SSIM` assume a data range of `1.0`; if your values span a different range, set `+metrics.data_range=...`. The `sigma` of a noise degradation is expressed in the same units, so `degradation.sigma=0.1` means noise with a standard deviation of 10% of the full range for 8-bit data.

The metrics do not clip reconstructions: values outside the data range count as errors.

## Error magnitude: PSNR, NRMSE and MSE

**What they measure.** All three summarise the same quantity, the mean squared error `MSE = mean((restored − reference)²)`, which is the power of the residual. They differ in how they scale it.

### PSNR (peak signal-to-noise ratio)

```text
PSNR = 10 · log10(data_range² / MSE)    [dB]
```

PSNR compares the residual power with the largest signal power the data format allows. The logarithmic scale makes gains easy to read:

- +3 dB halves the MSE, +10 dB divides it by ten.
- Adding Gaussian noise with `sigma=0.1` to an image in [0, 1] gives an MSE of about 0.01, so about **20 dB before any restoration**. A method scoring 26 dB reduces the residual power by roughly a factor of four.

*Interest.* It is simple, widely reported and useful for comparing methods on the same data. It is the quantity that is minimised when training with a mean-squared-error loss.

*Limits.*

- It depends on `data_range`. The same reconstruction gives different PSNR values with different ranges, so values are comparable only for the same scale.
- It counts all errors equally. Blurring edges and leaving a little noise in flat areas can produce the same PSNR, even though they look and behave very differently.
- It tends to favour smooth reconstructions. A blurry result can outscore a sharper one that has slightly misplaced details.
- It is `inf` when the images are identical, and a single `inf` makes the mean over images infinite.
- Averaging PSNR over images averages decibel values; this is not the PSNR of the average error.

### NRMSE (normalised root-mean-square error)

```text
NRMSE = sqrt(MSE) / (max(reference) − min(reference))
```

It expresses the typical error as a fraction of the reference's dynamic range, for example 0.05 means a typical error of 5% of the range.

*Interest.* It is easy to relate to intensity levels, and it is expressed in the units of the image rather than in decibels.

*Limits.* The normalisation uses the range of **each reference image**, whereas PSNR uses one fixed range for all images. Two images with identical absolute errors therefore get different NRMSE if their ranges differ, and a single bright pixel in a reference inflates the range and shrinks the score. A constant reference gives `0` for a perfect result and `inf` otherwise.

### MSE

The raw residual power, in squared intensity units. It is mainly useful to relate scores to a training loss or to compute error budgets. It is not normalised, so it cannot be compared between datasets with different scales.

## Local structure: SSIM

**What it measures.** The structural similarity index compares small windows of the two images using three ingredients: mean intensity (luminance), variance (contrast) and covariance (structure). A value of 1 means the windows are identical; values near 0 mean that the local patterns are uncorrelated. The score reported is the mean over all window positions.

*Interest.* It is more sensitive than PSNR to loss of texture, contrast and edges, which makes it a useful complement when a method over-smooths. A method with good PSNR but low SSIM probably removes detail.

*Limits.*

- The implementation uses a uniform 7 × 7 window with the usual stabilising constants (`k1=0.01`, `k2=0.03`). Implementations with Gaussian windows or other boundary handling can give slightly different values, so avoid comparing numbers with published tables.
- It depends on `data_range`, like PSNR.
- The sliding window is local. It does not capture global shifts well and responds weakly to a smooth intensity offset.
- It is not a perceptual or diagnostic quality measure. High SSIM does not prove that biologically relevant structures are preserved.
- For multichannel images, scores are computed per channel and averaged.

## Edges and fine detail: `grad_error`

```text
grad_error = ‖∇restored − ∇reference‖₂ / ‖∇reference‖₂
```

Gradients are finite differences over the two spatial axes. The score is the relative size of the error in the gradient field, so it measures how well edges, boundaries and fine details are reproduced.

It has a useful anchor value. A perfectly flat reconstruction has zero gradient, so its `grad_error` is exactly **1**:

- Close to 0: edges and details closely follow the reference.
- Around 1: the reconstruction preserves about as little gradient information as a flat image. Strong smoothing tends to approach this value.
- Above 1: the reconstruction adds more gradient error than it keeps, typically because of residual noise, ringing or other artefacts. Differentiation amplifies high frequencies, so noisy output is penalised heavily.

*Interest.* It reveals what PSNR hides: loss of sharpness and amplified noise. It is relevant when later analysis relies on contours, for example cell boundaries.

*Limits.* It is dominated by the strongest edges and by noise. It does not distinguish between a misplaced edge and a missing edge. If the reference is constant, it is `0` for a perfect result and `inf` otherwise. When the reference itself is noisy, much of its gradient energy is noise, which can make this score hard to interpret.

## Intensity conservation: `bias` and `flux_ratio`

These two metrics answer a question that error-based scores do not: does the restoration change the **amount of signal**?

```text
bias       = mean(restored − reference)
flux_ratio = sum(restored) / sum(reference)
```

They carry the same information, an overall shift in intensity, in two forms. `flux_ratio = 1 + bias / mean(reference)`.

- `bias` is expressed in intensity units. A positive value means the reconstruction is brighter than the reference on average.
- `flux_ratio` is dimensionless. A value of 0.95 means that 5% of the total intensity has been lost.

*Interest.* In quantitative microscopy, total or integrated intensity is often the measured quantity, for example for fluorescence quantification. A visually pleasing restoration can still shift the intensity enough to compromise such a measurement, and `PSNR` will not always show it.

*Limits.*

- They measure only the average. Errors of opposite signs cancel, so `bias=0` does not mean a good reconstruction. Always read them together with an error metric.
- Because of the same cancellation, the **mean of `bias` over images** can be close to zero even if individual images are strongly biased in different directions. Inspect the per-image values and the spread.
- `flux_ratio` is sensitive to dark references: when the reference has almost no total intensity, small absolute errors give large ratios. It returns `1` when both images have zero total intensity, and `inf` if only the reference does.

## Runtime

`runtime` is the wall-clock time of the restoration call for one image, in seconds. Metric computation and file output are excluded. Setting `metrics.compute_runtime=false` omits the column.

It is a single measurement per image, on your hardware, with your settings. It is not repeated, it can include one-off start-up costs, and CPU and GPU timings are not comparable. Use it to compare methods run in the same session on the same machine, not as an absolute benchmark.

## Reading metrics together

No single score captures restoration quality. These patterns help diagnose what a method is doing:

| Pattern | Likely interpretation |
| --- | --- |
| Good PSNR, but low SSIM and `grad_error` near 1 | Over-smoothing: noise is removed at the cost of edges and texture. |
| `grad_error` above 1 | Residual noise or artefacts; the output has more spurious gradients than useful ones. |
| `bias` away from 0, `flux_ratio` away from 1 | Systematic intensity shift, for example from a normalisation, clipping or an operation that does not conserve intensity. |
| Good SSIM but poor `bias` or `flux_ratio` | Structures are preserved but quantitative intensities are not. Be careful with intensity measurements. |
| Large `std` between images | Performance depends on image content; look at the worst per-image cases. |
| `inf` in any aggregate | At least one image is a perfect match or has a degenerate reference. Inspect `runs.parquet`. |

## Aggregation and comparison

`runs.parquet` contains one row per image; `summary.parquet` contains, for each group, the number of images `n` and the mean and standard deviation of each metric (`<metric>_mean`, `<metric>_std`). The standard deviation describes variation between images, using the population formula. It is **not** a confidence interval and it is not a statistical test.

Some practical guidance:

- Compare methods on exactly the same images, degradation and seed. The `benchmark` command does this for you.
- Prefer paired comparisons: look at the per-image difference between two methods rather than at the difference between averages.
- One noise realisation per image is drawn. To check that a conclusion does not depend on it, repeat the experiment with several values of `seed`.
- Interpret a score against the *unrestored* input. The score of the degraded image is the baseline for improvement. Training runs report it as `psnr_noisy` and `ssim_noisy` in their test output. The restoration and benchmark tasks do not, but for additive Gaussian noise the estimate given in the PSNR section approximates the PSNR of the unrestored input.

## Known limitations

- The reference is the acquired image, not a noise-free ground truth.
- All metrics are computed per image on the full frame, including borders and background.
- Metrics evaluate synthetic degradations. Performance on a real acquisition, with a different noise mixture, can differ.
- Absolute thresholds such as "PSNR above 30 dB is good" do not transfer between datasets. Use relative comparisons and your own domain criteria.
