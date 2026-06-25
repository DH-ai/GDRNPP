# Roadmap

## Near Term: Weekend Modernization

- Keep folder structure stable.
- Document the GDRNPP modernization vision.
- Record the modernization plan, architecture, and design decisions.
- Fix the current training blockers one at a time.
- Track discovered issues in `TODO.md` or GitHub Issues.
- Keep commits focused and easy to review.

## Milestone 1: Environment Compatibility

- Python 3.10+ compatibility.
- NumPy 2.x compatibility.
- PyTorch 2.x compatibility.
- CUDA 12-era setup documentation.
- Detectron2, MMCV, YOLOX, and PyTorch Lightning compatibility notes.

## Milestone 2: Dataset Reliability

- BOP dataset validator.
- Dataset repair utilities for safe deterministic fixes.
- Dataset statistics and visualization.
- Better messages for missing files, missing JSON entries, and path resolution failures.
- Custom dataset examples.

## Milestone 3: Mesh Reliability

- Modern PLY parser support.
- Mesh triangulation guidance.
- `models_info.json` generator or validator.
- Symmetry metadata helpers.
- Mesh unit and diameter checks.

## Milestone 4: Developer Experience

- Improved installation docs.
- Troubleshooting docs for dependency and dataset errors.
- Better logging during dataset loading and training startup.
- Examples for training and evaluating custom objects.
- Changelog entries for compatibility changes.

## Milestone 5: CI and Regression Coverage

- GitHub Actions for lightweight checks.
- Unit tests for patched utilities.
- Dataset validator tests.
- PLY parser tests.
- Smoke tests that do not require a GPU.

## Milestone 6: Performance and Deployment

- Mixed precision cleanup.
- Data loading profiling.
- Memory optimization.
- ONNX export review.
- TensorRT deployment investigation.

## Future: OpenPose3D Backend

GDRNPP Modernized should eventually be usable by a separate OpenPose3D or PoseToolkit repository as one backend among several.

That future project should own:

- Multi-backend pose estimation interfaces.
- MegaPose, FoundationPose, FoundationStereo, and future model adapters.
- Foundation model integrations.
- Robotics and VLA pipelines.
- Higher-level orchestration.

This repository should remain focused on GDRNPP.
