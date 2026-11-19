# Reading the results

Image Fidelity scores a method by comparing its output with a reference: a reconstruction with the original image, or a predicted segmentation with an annotation. Scores describe that agreement. They do not by themselves establish that a result is scientifically or visually useful.

## Choose a guide

| Task | Guide |
| --- | --- |
| `restore`, `benchmark` | [Restoration metrics](restoration-metrics.md): PSNR, SSIM, NRMSE, gradient error, bias, flux ratio and runtime |
| `segment` | [Segmentation metrics](segmentation-metrics.md): Dice, IoU, object counts, precision, recall, F1 and AP |

Each guide explains what a metric measures, when it is informative, and where it can mislead.

## Where the scores are

Result tables are written to the run's `results/` directory:

- `runs.parquet` contains one row per image, with the parameters of the method and the degradation.
- `summary.parquet` contains, per group, the number of images `n` and the mean and standard deviation of each metric (`<metric>_mean`, `<metric>_std`).

See the [configuration guide](../configs/README.md#outputs-and-reproducibility) for the output layout.

## Good practice

- **Compare like with like.** Use the same images, degradation, noise level and seed for every method. The `benchmark` command does this for you.
- **Look at the images.** Scores summarise; the saved previews show what a method actually does.
- **Look at individual images.** An average can hide the few images where a method fails. Inspect `runs.parquet` and the spread, not only the mean.
- **Treat the standard deviation as spread, not as evidence.** It describes variation between images. It is neither a confidence interval nor a significance test.
- **Check robustness.** One noise realisation is drawn per image. Repeat the experiment with several `seed` values before drawing conclusions.
- **Mind the scale.** Integer inputs are divided by their dtype's maximum value; floating-point inputs keep their scale. PSNR and SSIM assume a data range of `1.0`: set `+metrics.data_range=...` if yours differs, and make sure noise parameters suit the same scale.
- **Expect degenerate cases.** Constant references, zero total intensity or identical images can yield infinite scores, which propagate to averages. Inspect those images.
- **Use relative judgements.** There is no universal "good" value. Compare with a baseline and with your own domain requirements.
