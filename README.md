# GDRNPP Modernized

GDRNPP Modernized is a maintained and modernized implementation of GDRNPP, a geometry-guided 6D object pose-estimation system. The repository keeps the original GDRNPP workflow—BOP datasets, YOLOX detection, GDRN pose estimation, rendering, evaluation, and inference—while addressing compatibility and reliability problems in contemporary development environments.

This repository is focused on GDRNPP. It is not intended to become a general robotics, VLM, VLA, or multi-model pose-estimation platform. A future OpenPose3D or PoseToolkit project can use this repository as one of its pose-estimation backends.

## Project status

The modernization is incremental. The repository currently includes:

- Compatibility work for Python 3.10-era environments, newer NumPy releases, PyTorch changes, and Blender 4.x binary PLY exports.
- The original GDRNPP pose-estimation pipeline under `core/`.
- A YOLOX-based object-detection pipeline under `det/yolox/`.
- BOP-format dataset references, including a custom `mydataset_pbr` integration.
- A BOP dataset integrity checker in [`bop_integrity.py`](bop_integrity.py).
- Training-path fixes for annotation image dimensions, batch-size-scaled YOLOX learning rates, and robust L1-loss handling. See [`PATCH_REPORT.md`](PATCH_REPORT.md).
- Documentation for architecture, installation, decisions, roadmap, and troubleshooting.

The environment is not yet represented by one fully pinned, reproducible dependency lockfile. Dependency versions—especially Detectron2, MMCV, PyTorch, CUDA, PyTorch3D, and compiled extensions—must be selected for the target machine. See [`docs/INSTALL.md`](docs/INSTALL.md) and [`troubleshoot.md`](troubleshoot.md) before installing.

## Pipeline overview

```text
BOP dataset or synthetic BOP dataset
                  |
                  v
        Dataset registration and checks
                  |
                  v
        YOLOX object detection (optional)
                  |
                  v
      Detection boxes in original image space
                  |
                  v
          GDRNPP ROI pose estimation
                  |
                  v
       BOP pose results and evaluation outputs
```

The detector and pose-estimation stages are separate. A detector can be replaced at the detection-file boundary without changing the GDRN model internals. The current data, detection, and pose interfaces are documented in [`docs/DATA_DETECTION_POSE_ARCHITECTURE_AUDIT.md`](docs/DATA_DETECTION_POSE_ARCHITECTURE_AUDIT.md).

## Repository layout

```text
src/gdrnpp/
├── configs/                  GDRN and YOLOX configurations
├── core/                     GDRNPP models, datasets, engines, and tools
├── det/yolox/                YOLOX detector, data loading, and evaluation
├── lib/                      Renderers, utilities, layers, and extensions
├── ref/                      BOP dataset metadata and custom dataset refs
├── requirements/             Python dependency list
├── scripts/                  Dependency setup, environment, and build scripts
├── tools/                    Checkpoint and inference utilities
├── evaluators/               Evaluation helpers
├── docs/                     Architecture and project documentation
├── bop_integrity.py          BOP dataset integrity checker
├── troubleshoot.md           Known setup and runtime issues
└── TODO.md                   Active modernization work
```

## Installation

The original project contains native CUDA/C++ extensions and graphics dependencies, so installation is environment-sensitive.

1. Create and activate a Python environment appropriate for your CUDA and PyTorch versions. Python 3.10 is the current modernization baseline where supported.

2. Install PyTorch and torchvision using the command appropriate for the installed CUDA toolkit.

3. Install Detectron2 from source as described in [`docs/INSTALL.md`](docs/INSTALL.md).

4. Install the repository dependencies:

   ```bash
   sh scripts/install_deps.sh python
   ```

   The default `sh scripts/install_deps.sh` also installs system packages with `sudo`.

5. Build the required native extensions when your workflow needs them:

   ```bash
   sh scripts/compile_all.sh
   ```

   Review the `CUDA_HOME` setting in the script before running it on a different machine.

6. From the repository root, load the environment needed by the uncertainty-PnP extension when applicable:

   ```bash
   source scripts/init_env.sh
   ```

For complete dependency notes, Ubuntu-specific graphics instructions, and individual extension builds, see [`docs/INSTALL.md`](docs/INSTALL.md).

## Dataset format

Training and evaluation expect BOP-style datasets. A typical custom dataset contains:

```text
datasets/BOP_DATASETS/mydataset/
├── models/
│   ├── obj_000001.ply
│   └── ...
├── train_pbr/
│   └── 000000/
│       ├── rgb/
│       ├── depth/
│       ├── mask/
│       ├── mask_visib/
│       ├── scene_camera.json
│       ├── scene_gt.json
│       └── scene_gt_info.json
└── test/
    └── ...
```

Before training, validate the dataset and inspect the report:

```bash
python bop_integrity.py \
  --root /path/to/train_pbr \
  --bbox-source visib \
  --cache-out /path/to/cache.pkl
```

The custom dataset references are currently split between the detector and pose-estimation paths. Keep image dimensions, file extensions, object IDs, and dataset roots consistent across the relevant configuration and reference files. The architecture audit documents the current boundaries and known integration risks.

## Training and evaluation

### GDRNPP pose estimation

Train with a GDRN configuration and one or more visible GPU IDs:

```bash
bash core/gdrn_modeling/train_gdrn.sh \
  configs/gdrn/mydataset_pbr/convnext_mydataset.py \
  0
```

Evaluate a checkpoint:

```bash
bash core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/mydataset_pbr/convnext_mydataset.py \
  0 \
  /path/to/checkpoint.pth
```

The configuration must point to valid dataset roots, object models, camera data, and output directories for the target machine.

### YOLOX detection

The detector uses Detectron2 LazyConfig files. Run the main entry point directly so that additional command-line options are passed consistently:

```bash
PYTHONPATH=. python det/yolox/tools/main_yolox.py \
  --config-file \
  configs/yolox/bop_pbr/yolox_x_1920_augCozyAAEhsv_ranger_30_epochs_mydataset_pbr_mydataset_test_primesense.py \
  --num-gpus 1
```

The YOLOX training and evaluation wrappers are also available under [`det/yolox/tools/`](det/yolox/tools/). Review the selected config before running because batch size, image size, dataset paths, checkpoint paths, and number of classes are dataset-specific.

## Compatibility work

The current modernization work includes:

- Replacing removed Python and NumPy APIs.
- Improving binary PLY parsing for Blender 4.x exports, including unsigned integer face-index data.
- Updating compatibility-sensitive training and dataset code.
- Reading image dimensions from the actual image files when preparing YOLOX annotations.
- Scaling the YOLOX optimizer learning rate from `basic_lr_per_img` and the configured batch size.
- Preventing an exception in the YOLOX L1-loss path from producing an undefined loss value.
- Improving error messages and documenting common build failures.

These changes aim to preserve the original model behavior. They do not constitute a complete migration to every current version of Python, PyTorch, CUDA, or the surrounding libraries.

## Documentation map

- [`docs/VISION.md`](docs/VISION.md) — project purpose and modernization principles.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current boundaries and data flow.
- [`docs/DATA_DETECTION_POSE_ARCHITECTURE_AUDIT.md`](docs/DATA_DETECTION_POSE_ARCHITECTURE_AUDIT.md) — passive audit of the current detector and pose pipeline.
- [`docs/INSTALL.md`](docs/INSTALL.md) — dependencies, native extensions, and system setup.
- [`docs/MODERNIZATION_PLAN.md`](docs/MODERNIZATION_PLAN.md) — phased engineering plan.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — milestones and future work.
- [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) — durable project decisions.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — contribution scope and patch guidance.
- [`PATCH_REPORT.md`](PATCH_REPORT.md) — detailed YOLOX training fixes.
- [`TODO.md`](TODO.md) — active known work.
- [`troubleshoot.md`](troubleshoot.md) — setup and runtime troubleshooting.

## Scope and future direction

This repository should remain focused on making GDRNPP reliable as a 6D pose-estimation backend. Dataset validation, mesh tooling, environment compatibility, tests, logging, and documentation are in scope.

General multi-model orchestration, robotics integration, foundation-model pipelines, and a full OpenPose3D platform belong in a separate project. See [`VISION_GDRNPP_MODERNIZE.md`](VISION_GDRNPP_MODERNIZE.md) for the longer-term direction.

## License and upstream

This project retains the upstream project structure and is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE).

The implementation builds on the original [GDRNPP BOP2022 repository](https://github.com/shanice-l/gdrnpp_bop2022). Please preserve upstream attribution and review the license before redistributing modified code or trained models.
