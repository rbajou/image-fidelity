# Image Fidelity

Image Fidelity is a Python package for comparing image restoration methods under controlled synthetic degradations. It loads reference images, applies noise, restores the degraded images, and measures reconstruction quality and runtime.

```text
Reference images → Degradation → Restoration → Evaluation → Results and figures
```

Use it to compare classical filters and a PyTorch U-Net, study sensitivity to noise levels, and run parameter sweeps. A classical watershed segmentation workflow is also available for images with instance masks.

## Installation

Requires **Python 3.10 or later** and a local copy of this repository. From its root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[benchmark,visualization]"
imfid --help
```

On Windows, use `.venv\Scripts\Activate.ps1` in PowerShell or `.venv\Scripts\activate.bat` in Command Prompt to activate the environment.

This is the recommended installation for the command-line examples. Run them from the repository root. See [INSTALL](INSTALL) for optional dependencies, learning and development setups, and troubleshooting.

## Your first restoration

Place a few reference images in `data/my_images/`:

```text
data/my_images/
├── image_001.png
├── image_002.png
└── image_003.png
```

Start with grayscale 2D PNG images, for example 64 × 64 pixels. The loader searches recursively; `data.grayscale=true` converts RGB/RGBA images to grayscale.

Add Gaussian noise, restore with a Gaussian filter, and evaluate every image:

```bash
imfid restore \
  data=folder data.name=my_images data.grayscale=true \
  data.val_fraction=0 data.test_fraction=1 task.max_samples=null \
  degradation=gaussian_noise degradation.sigma=0.1 \
  method=gaussian method.sigma=1.0
```

The zero validation fraction and unit test fraction put all images in the evaluation split. `task.max_samples=null` removes the restoration task's default limit of 20 images. Use separate partitions for training and model selection.

Find the run in `outputs/restore/<date>_<time>_restore/`:

- `images/samples.png`: reference, degraded and restored images for up to four samples.
- `results/runs.parquet`: per-image measurements.
- `results/summary.parquet`: aggregated scores.
- `predictions/`: reconstructed arrays in NumPy format.
- `figures/`: metric distributions and boxplots.

## Common workflows

### Compare methods on the same degraded images

```bash
imfid benchmark \
  data=folder data.name=my_images data.grayscale=true \
  data.val_fraction=0 data.test_fraction=1 \
  degradation=gaussian_noise degradation.sigma=0.1 \
  method=gaussian,median,nlm
```

The leading `benchmark` command interprets `method=gaussian,median,nlm` as a comparison within one run. All methods receive the same degraded samples.

### Sweep a filter parameter

```bash
imfid restore \
  data=folder data.name=my_images data.grayscale=true \
  data.val_fraction=0 data.test_fraction=1 task.max_samples=null \
  degradation=gaussian_noise degradation.sigma=0.1 \
  method=gaussian method.sigma=0.5,1.0,2.0
```

Comma-separated overrides trigger a Hydra multirun, except for the method list handled by the leading `benchmark` command. Each job has its own directory; result tables and sweep figures are aggregated at the end.

### Preview noise without restoring

```bash
imfid task=degrade \
  data=folder data.name=my_images data.grayscale=true \
  data.val_fraction=0 data.test_fraction=1 \
  degradation=poisson_noise
```

This writes reference/degraded image pairs for up to eight samples.

### Use BBBC038 and train a U-Net

The [Broad Bioimage Benchmark Collection (BBBC)](https://bbbc.broadinstitute.org/) provides biological microscopy image datasets for developing and evaluating image analysis methods, with reference annotations available for many datasets.

The BBBC038 presets use [images of cell nuclei assembled for the Kaggle 2018 Data Science Bowl](https://bbbc.broadinstitute.org/BBBC038). These 2D images span different organisms, biological samples, staining techniques (including fluorescence and histology), magnifications and illumination conditions. Training images include individual nucleus masks for instance segmentation. In this package's restoration workflows, the acquired images serve as references to which synthetic noise is added; the nucleus masks are used by the segmentation workflow.

Download BBBC038 with:

```bash
imfid download bbbc 038
```

Before using BBBC038 presets, check the extracted directory layout against the [data preparation guide](configs/README.md#data-and-partitions). Then install the learning dependencies and start training on CPU:

```bash
python -m pip install -e ".[learning,benchmark,visualization]"
imfid experiment=train_unet trainer.device=cpu trainer.epochs=1 trainer.num_workers=0
```

This is a one-epoch trial. For a full experiment, choose the epoch count and device explicitly. The preset uses residual prediction; restoration must load a matching model and a checkpoint from your run. See the [training and checkpoint recipes](configs/README.md#train-and-evaluate-a-u-net).

## Reading the results

Restoration tasks treat the input images as references, synthetically degrade them, and compare the reconstruction against the reference.

| Metric                                        | Interpretation                                      |
| --------------------------------------------- | --------------------------------------------------- |
| PSNR, SSIM                                    | Higher is better                                    |
| NRMSE, relative gradient error (`grad_error`) | Lower is better                                     |
| Signed intensity bias (`bias`)                | Closer to zero is better                            |
| Total flux ratio (`flux_ratio`)               | Closer to one is better                             |
| Runtime                                       | Restoration time, excluding metrics and file output |

Integer inputs are converted to floating point and divided by their dtype's maximum value. Floating-point inputs retain their intensity scale. PSNR and SSIM use a default data range of 1.0; set `+metrics.data_range=...` if your scale differs. Noise parameters must also suit that scale.

Constant references or zero total intensity can produce infinite normalized scores. Inspect the images and per-image tables alongside aggregate scores.

## Configuration and outputs

Hydra groups select the task, data, degradation, restoration method, metrics and training settings. An `experiment` preset combines these choices:

```bash
imfid experiment=benchmark_restore_classical seed=123 task.max_samples=10
```

This preset requires BBBC038 data. The [configuration guide](configs/README.md) lists usable presets and explains overrides, partitions, plotting, sweeps and reproduction.

Typical restoration and benchmark outputs are:

```text
outputs/<task>/<date>_<time>_<run-name>/
├── .hydra/             Saved configuration and overrides
├── results/            Per-image and summary Parquet tables
├── predictions/        Reconstruction arrays, when enabled
├── images/             Image comparisons
├── figures/            Metric plots
└── logs/               Execution logs
```

Training also writes `checkpoints/`. Multirun jobs occupy numbered subdirectories. Output switches and task-specific differences are documented in the configuration guide.

## Scope and project layout

```text
configs/                  Hydra components and experiment presets
src/image_fidelity/
├── cli/                   Command-line interface
├── tasks/                 Workflow orchestration
├── io/                    Image and array readers/writers
├── datasets/              Loading, partitions and patches
├── degradation/          Synthetic degradations
├── restoration/           Classical and neural restoration
├── segmentation/          Classical segmentation and evaluation
├── training/              U-Net training
├── metrics/               Image and segmentation metrics
├── evaluation/            Evaluation and statistics
├── benchmark/             Comparisons and aggregation
└── visualization/         Image previews and metric plots
tests/                     Automated tests
```

The package is under active development. Implemented classical restoration methods include Gaussian, median, bilateral and non-local means filters, Wiener and Richardson–Lucy deconvolution, and Tikhonov and total variation regularization. Only some have ready-made presets in `configs/method/`.

The documented learning workflow is a 2D U-Net. Learned segmentation and DnCNN configurations are placeholders. Generic array readers do not imply that every task supports volumetric data.

See [INSTALL](INSTALL#development-and-tests) for development prerequisites and tests. Seeds and saved configurations support reproducibility, but identical numerical results across hardware and software environments are not guaranteed.

## Data credits and license

Consult each BBBC dataset's terms and citation requirements before using or redistributing its data. A project license has not yet been included in this repository.
