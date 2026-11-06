# Configuration guide

Start with the [first restoration](../README.md#your-first-restoration) and [installation guide](../INSTALL). Commands below assume the recommended extras are installed and the current directory is the repository root. Multi-line commands use a POSIX shell; on Windows, enter them on one line.

## How configuration works

`config.yaml` selects reusable Hydra groups. Command-line overrides replace a group or change one of its values:

| Group                  | Controls                                           |
| ---------------------- | -------------------------------------------------- |
| `task`                 | Workflow, split, sample limits and outputs         |
| `data`                 | Paths, grayscale conversion and partitions         |
| `degradation`          | Synthetic degradation and parameters               |
| `method`               | Restoration or segmentation component              |
| `model`                | Training architecture                              |
| `optimizer`, `trainer` | Optimization and training settings                 |
| `metrics`              | Measurements and runtime reporting                 |
| `plot`                 | Figure specifications, composed under `task.plots` |
| `experiment`           | Presets combining several groups                   |

For example, `method=gaussian` selects a YAML file; `method.sigma=1.0` changes an existing value. Use `+key=value` to add a key absent from the composed configuration.

Inspect a configuration before running it:

```bash
imfid restore method=gaussian degradation=gaussian_noise --cfg job
imfid experiment=train_unet trainer.device=cpu --cfg job
```

`imfid restore`, `benchmark`, `train`, `plot` and `download` support leading task names. For degradation and segmentation, use `imfid task=degrade` and `imfid task=segment`, or an experiment preset.

The default configuration selects `task=restore`, `data=bbbc038_small`, `method=nlm` and `degradation=none`. A restoration run therefore needs an explicit degradation to evaluate noise removal.

## Data and partitions

### A folder of reference images

`data=folder` recursively loads PNG, JPEG, BMP, GIF and TIFF images. TIFF requires the `tiff` extra. Set `data.name` to select a directory under `paths.data_dir`, or supply a root explicitly:

```bash
imfid restore data=folder data.name=my_images \
  data.root_dir=/absolute/path/to/images data.grayscale=true \
  data.val_fraction=0 data.test_fraction=1 task.max_samples=null \
  degradation=gaussian_noise method=gaussian
```

Without predefined train/test directories, the loader splits the image collection using `data.val_fraction`, `data.test_fraction`, `data.seed` and `data.shuffle`. The remaining images form the training split. The example above assigns everything to evaluation.

The task chooses `task.split=train`, `validation` (or `val`), or `test`. `task.max_samples` limits the number loaded; `null` removes that limit. Any `data.limit` for the split still applies. Restoration defaults to at most 20 images, degradation previews to eight, and benchmarks have no task-level limit.

### BBBC038

[BBBC038](https://bbbc.broadinstitute.org/BBBC038), part of the Broad Bioimage Benchmark Collection, contains 2D microscopy images of cell nuclei collected for the Kaggle 2018 Data Science Bowl. It combines diverse biological samples and imaging conditions, including fluorescence and histological staining. The training set provides one binary PNG mask per annotated nucleus; the loader combines these into an instance-label map.

Restoration uses the acquired images as references for synthetic degradation experiments. These references are experimental images, rather than guaranteed noise-free acquisitions. Segmentation evaluates predicted instances against the supplied nucleus annotations.

Download from the repository root:

```bash
imfid download bbbc 038
```

`paths.data_dir` defaults to `./data`; set the `DATA_DIR` environment variable or pass `paths.data_dir=/absolute/path/to/data` to change it. The downloader extracts each ZIP into `<data_dir>/bbbc038/<archive-stem>/`.

The BBBC038 presets expect this layout:

```text
data/bbbc038/
├── stage1_train/
│   └── <sample-id>/
│       ├── images/<image>.png
│       └── masks/<instance>.png
└── stage1_test/
    └── <sample-id>/
        └── images/<image>.png
```

Check the archive names and extracted folders after downloading. If they differ, arrange them as above or override `data.train_dir` and `data.test_dir` relative to `data.root_dir`.

The `bbbc038` group uses predefined train/test folders and draws validation from the training folder. Training excludes validation samples. For direct evaluation with `task.split=train`, the loader uses the full training folder. Validation samples also have masks; the supplied watershed preset uses the training split because the official test images have no ground-truth masks.

`bbbc038_small` uses the same files, with limits of 128 training, 32 validation and 32 test images. It does not download a separate small dataset.

Other downloads and options:

```bash
imfid download bbbc 010 038 'task.files=[foreground]'
imfid download bbbc 038 task.extract=false
```

`task.files` filters links by filename substring. `task.force=true` downloads again and replaces the corresponding extracted archive directory; `task.keep_archives=false` removes ZIPs after extraction.

## Experiment presets

Select a preset with `imfid experiment=<name>`. The name is the filename under `experiment/`, without `.yaml`.

| Preset                                         | Purpose and prerequisites                                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `restore_nlm`                                  | NLM restoration on BBBC038                                                                                 |
| `benchmark_restore_classical`                  | Gaussian, median, NLM and TV comparison on BBBC038                                                         |
| `benchmark_restore_sweep`                      | Richardson–Lucy on BBBC010 across Gaussian noise levels; requires `-m` and an explicit data-path check     |
| `train_unet`                                   | Residual U-Net training on BBBC038; defaults to GPU                                                        |
| `finetune_unet`                                | Load initial weights and train at a lower learning rate; requires a checkpoint and matching model settings |
| `restore_unet`                                 | BBBC038 U-Net restoration; override the preset's local checkpoint path                                     |
| `benchmark_restore_unet_vs_classical`          | U-Net/classical comparison on BBBC038; override the checkpoint path                                        |
| `benchmark_restore_unet_static_vs_variable_nl` | Comparison of several trained U-Nets across noise levels; requires three supplied checkpoints and `-m`     |
| `segment_watershed`                            | Classical instance segmentation on BBBC038 training images                                                 |

`degrade_gaus` is a preview preset; its sample count should be set with `task.max_samples` (its root-level `max_samples` does not control the loader). `segment_unet`, the `segmentation_unet` model and the `dncnn` model are placeholders. `benchmark_segment_classical` targets the restoration benchmark handler and is not a supported segmentation comparison recipe.

## Restore and compare

With BBBC038 prepared:

```bash
imfid restore data=bbbc038_small method=nlm degradation=gaussian_noise
imfid experiment=benchmark_restore_classical task.max_samples=10
```

To compare methods without an experiment preset:

```bash
imfid benchmark data=bbbc038_small degradation=gaussian_noise method=gaussian,median,nlm
```

The leading `benchmark` command maps that list to `task.methods`; all restorers share the degraded samples. This shorthand is specific to `imfid benchmark`. `imfid task=benchmark method=gaussian,median,nlm` does not perform the same translation.

## Parameter sweeps

Comma-separated overrides automatically enable multirun. An explicit `-m` is also accepted:

```bash
imfid restore -m data=bbbc038_small degradation=gaussian_noise method=gaussian method.sigma=0.5,1.0,2.0
```

Keep a leading task name immediately after `imfid`, before Hydra flags, so the CLI can translate it.

Grids stored in `hydra.sweeper.params` require `-m`; the CLI does not automatically activate them. The current BBBC010 sweep varies `degradation.sigma` over `0.1,0.2,0.5,1`:

```bash
imfid -m experiment=benchmark_restore_sweep \
  data.root_dir=./data/bbbc010/BBBC010_v1_foreground
```

Check that this path matches the extracted archive; the preset's original data root is `./data/BBBC010_v1_foreground`.

Each job writes into a numbered directory. When result tables are available, a Hydra callback concatenates them into the sweep root's `results/runs.parquet`, writes grouped `summary.parquet`, and generates `figures/metrics_vs_<parameter>.png`. Keep the benchmark and visualization extras installed for this aggregation.

## Train and evaluate a U-Net

Install `.[learning,benchmark,visualization]` as described in [INSTALL](../INSTALL). Prepare BBBC038, then try one epoch on CPU:

```bash
imfid experiment=train_unet trainer.device=cpu trainer.epochs=1 trainer.num_workers=0
```

The preset uses a 2D U-Net with residual prediction, random 128 × 128 crops and Gaussian noise. It defaults to 100 epochs and CUDA; choose settings appropriate to your machine. Validation runs each epoch. Training saves `checkpoints/best.pt` and `checkpoints/last.pt`, then evaluates the best weights on the test split; disable that pass with `task.run_test=false`.

To train with a noise level drawn separately for each patch:

```bash
imfid experiment=train_unet trainer.device=cpu \
  task.noise.mode=variable task.noise.param=sigma \
  task.noise.min=0.02 task.noise.max=0.2
```

Use `task.noise.*`: the preset currently also contains a root-level `noise` block, which the training pipeline does not read. The default effective mode is constant. The test pass uses `degradation.sigma` even after variable-noise training.

Replace `/absolute/path/to/best.pt` with your saved checkpoint:

```bash
imfid restore data=bbbc038_small method=unet_restorer \
  method.checkpoint=/absolute/path/to/best.pt method.model.residual=true \
  method.device=cpu degradation=gaussian_noise
```

Match all architecture settings to training, including channel counts, width, depth and residual prediction. For a model trained with `model.residual=false`, use `method.model.residual=false`.

Compare a trained residual model with classical methods on validation images, then report test performance after choosing parameters:

```bash
imfid experiment=benchmark_restore_unet_vs_classical task.split=validation \
  task.methods.unet.checkpoint=/absolute/path/to/best.pt project.device=cpu
imfid experiment=benchmark_restore_unet_vs_classical task.split=test \
  task.methods.unet.checkpoint=/absolute/path/to/best.pt project.device=cpu
```

The finetuning preset loads model weights, then starts a new optimizer and training history; it does not resume the complete optimizer state. Supply `task.checkpoint` and matching `model.*` settings explicitly.

## Segment instances

Use the classical watershed preset:

```bash
imfid experiment=segment_watershed method.threshold=otsu
imfid -m experiment=segment_watershed method.threshold=otsu,local
```

The first command runs once. The second also activates the preset's configured sweep of noise levels. Each image is degraded before segmentation, while metrics compare the prediction to the original ground-truth mask.

The segmenter is configured under `method`, not `task.method`. Outputs include instance maps in `masks/*.npy`, Parquet score tables and metric figures. By default, the first four processed images also have previews in `images/<image_id>.png`, showing the original, degraded input, ground-truth contours and predicted instances. Preview titles include the method, degradation and available segmentation scores. Instance colors are for visualization only and do not imply correspondence between predicted and ground-truth labels.

Set `task.image_samples` to change the preview count, or `task.save.images=false` to disable previews. This does not change how many images are processed; use `task.max_samples` for that. Each job in a multirun writes its own previews.

Images without ground-truth masks are still segmented and can have masks and previews saved; their preview indicates that ground truth is unavailable. Metrics and their tables only include annotated images. If no image has annotations, the run logs a warning and produces no metric tables or metric figures.

## Regenerate figures

`plot` reads saved tables without repeating restoration. Replace the example run path with an existing run directory:

```bash
imfid plot 'task.inputs=[outputs/benchmark/YOUR_RUN]'
imfid plot 'task.inputs=[outputs/benchmark/YOUR_RUN]' '~task.plots.metric_sweep'
imfid plot 'task.inputs=[outputs/benchmark/YOUR_RUN]' \
  +plot@task.plots.boxplots=metric_boxplots task.plots.boxplots.group_by=method
```

With several inputs, pass a quoted list such as `'task.inputs=[outputs/benchmark/RUN_A,outputs/benchmark/RUN_B]'`.

Figure specifications live in `plot/` and are enabled under `task.plots`. Distributions are split by method by default; set `task.plots.metric_distributions.split_by=null` for one combined figure. Plots without the required variation or tables are skipped.

For one input, figures are written into that run's `figures/`, replacing same-named files. For multiple inputs, they go into the new plot run's `figures/` directory.

## Outputs and reproducibility

Runs use `outputs/<task>/<date>_<time>_<run-name>/`; `run_name` defaults to the task name. Multiruns place jobs in numbered subdirectories.

| Task        | Main outputs and controls                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `restore`   | Tables, reconstruction arrays, image previews and metric figures; `task.save.results/predictions/images/figures` |
| `benchmark` | Tables, per-method previews and optional predictions; same switches; predictions default to disabled             |
| `degrade`   | Image pairs; `task.save.arrays=true` also saves degraded arrays                                                  |
| `train`     | Checkpoints, history and figures, optional test metrics; `task.save_best`, `task.save_figures`, `task.run_test`  |
| `segment`   | Mask arrays, tables, previews and figures; `task.save.masks/results/images/figures`                              |
| `plot`      | Figures regenerated from previous result tables                                                                  |
| `download`  | Archives and extracted files under `paths.data_dir`                                                              |

`task.image_samples` controls restoration, benchmark and segmentation preview counts (default four). Restoration and benchmark runtime excludes metrics and output writing. See the [metric conventions](../README.md#reading-the-results) before comparing scores.

Hydra stores the composed configuration and overrides under `.hydra/`. To rerun a saved training configuration into a new output directory, replace `YOUR_RUN` below:

```bash
imfid --config-path "$PWD/outputs/train/YOUR_RUN/.hydra" --config-name config \
  hydra.run.dir=outputs/train/YOUR_RUN_repro
```

Keep the original data, code version, dependencies and model files available. `seed` controls randomized operations; saved configuration alone does not guarantee identical numerical results across machines or software versions.
