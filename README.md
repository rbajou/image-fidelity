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

## Explore further

The [configuration guide](configs/README.md) covers data and partitions, BBBC038, presets, comparisons, parameter sweeps, training, segmentation, plotting and output files. Jump directly to:

- [Prepare data and partitions](configs/README.md#data-and-partitions)
- [Compare methods](configs/README.md#restore-and-compare)
- [Run parameter sweeps](configs/README.md#parameter-sweeps)
- [Preview a degradation](configs/README.md#preview-a-degradation)
- [Train and evaluate a restoration U-Net](configs/README.md#train-and-evaluate-a-restoration-u-net)
- [Train and use a segmentation U-Net](configs/README.md#train-and-use-a-segmentation-u-net)

For help interpreting scores, see [Reading the results](docs/reading-results.md), with guides to [restoration metrics](docs/restoration-metrics.md) and [segmentation metrics](docs/segmentation-metrics.md). For optional dependencies, development setup and tests, see [INSTALL](INSTALL).

## Implementation status

The package is under active development. Implemented classical restoration methods include Gaussian, median, bilateral and non-local means filters, Wiener and Richardson–Lucy deconvolution, and Tikhonov and total variation regularization. Only some have ready-made presets in `configs/method/`.

The documented learning workflow is a 2D U-Net. Generic array readers do not imply that every task supports volumetric data.

See [INSTALL](INSTALL#development-and-tests) for development prerequisites and tests. Seeds and saved configurations support reproducibility, but identical numerical results across hardware and software environments are not guaranteed.

## Data credits and license

Consult each BBBC dataset's terms and citation requirements before using or redistributing its data. A project license has not yet been included in this repository.
