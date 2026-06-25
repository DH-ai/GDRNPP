# GDRNPP Modernization Plan

## Phase 1: Modern Runtime Baseline

Target a working baseline on:

- Python 3.10 to 3.13 where feasible.
- CUDA 12-era systems.
- PyTorch 2.x.
- Current Linux development environments.

Initial tasks:

- Fix Python API removals such as moved `collections.abc` imports.
- Update deprecated PyTorch AMP calls, including `torch.cuda.amp.GradScaler(...)`.
- Document tested Python, CUDA, PyTorch, and GPU combinations.
- Make setup failures clear and reproducible.

## Phase 2: Dependency Cleanup

Clean up compatibility around:

- NumPy 2.x.
- Detectron2.
- MMCV.
- YOLOX.
- PyTorch Lightning.
- Build tools and compiled extensions.

Tasks:

- Replace removed NumPy aliases such as `np.float`, `np.int`, `np.bool`, `np.object`, and `np.str`.
- Pin or document known-good dependency ranges.
- Separate hard requirements from optional tooling where possible.
- Add troubleshooting notes for compiled dependency failures.

## Phase 3: Dataset Improvements

Make custom datasets easier to validate before training.

Tasks:

- Add a dataset validator for BOP-style datasets.
- Detect missing RGB, depth, mask, `scene_gt`, `scene_gt_info`, and `scene_camera` entries.
- Report affected scene and image IDs.
- Add dataset statistics and summary output.
- Add visualization checks for masks, bounding boxes, and pose annotations.
- Add repair paths for deterministic issues where safe.

## Phase 4: Mesh Tooling

Improve mesh handling for modern synthetic data workflows.

Tasks:

- Improve PLY parsing for modern Blender exports.
- Support additional scalar types such as `uint` where needed.
- Add or document mesh triangulation.
- Generate or validate `models_info.json`.
- Verify mesh units, diameter, bounding boxes, and symmetries.
- Add tests for representative PLY files.

## Phase 5: Developer Experience

Make the repository easier to use and debug.

Tasks:

- Replace cryptic assertions with actionable error messages.
- Add structured logging for dataset loading and training setup.
- Improve README and troubleshooting docs.
- Add examples for custom dataset training.
- Add environment setup docs for modern systems.
- Track known compatibility issues in `TODO.md`.

## Phase 6: CI and Tests

Introduce automated checks for modernization work.

Tasks:

- Add GitHub Actions for linting and targeted tests.
- Add unit tests for compatibility-sensitive utilities.
- Add regression tests for dataset validation.
- Add parser tests for PLY and metadata handling.
- Add compatibility smoke tests where GPU access is not required.

## Phase 7: Performance and Export

After compatibility and correctness are stable, explore performance improvements.

Potential tasks:

- Mixed precision cleanup.
- `torch.compile` investigation.
- Data loading improvements.
- Memory usage reduction.
- ONNX export review.
- TensorRT deployment investigation.

These should wait until the training path is stable and tested.

## Phase 8: Future Backend Role

Keep this repository suitable as a backend for a future OpenPose3D or PoseToolkit project.

This means:

- Keep GDRNPP-specific code here.
- Avoid adding unrelated foundation model or robotics orchestration here.
- Document backend boundaries clearly.
- Make setup, training, inference, and evaluation callable from external tooling where reasonable.
