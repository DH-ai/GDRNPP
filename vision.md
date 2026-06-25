# GDRNPP Modernized Vision

> A maintained, modernized, and developer-friendly GDRNPP codebase for 6D object pose estimation.

## Motivation

GDRNPP remains one of the strongest open-source baselines for 6D object pose estimation, but the surrounding Python, PyTorch, CUDA, NumPy, Detectron2, and dataset tooling ecosystems have moved on. The original repository was designed around older dependency versions and benchmark-style datasets that are assumed to be complete and correctly formatted.

This repository exists to make GDRNPP practical to use in a modern development environment while staying close enough to upstream that fixes remain understandable, reviewable, and maintainable.

## Current Problems

Modern users run into a mix of compatibility and workflow issues:

- Python APIs removed or moved in Python 3.10 and newer.
- NumPy aliases removed in NumPy 2.x.
- PyTorch and AMP APIs changing across PyTorch 2.x releases.
- Detectron2, MMCV, CUDA, and build tooling compatibility drift.
- PLY and mesh parsing assumptions that fail on modern Blender exports.
- Dataset loaders that fail late with assertions, missing keys, or unclear errors.
- Limited guidance for custom datasets, dataset validation, repair, and debugging.
- Sparse test coverage and no clear CI contract for supported environments.

These are engineering maintenance problems, not reasons to redesign the whole project.

## Project Vision

The goal of this repository is to answer a narrow question:

> What would GDRNPP look like if it were actively maintained in 2026?

This project should modernize GDRNPP without turning it into a completely different framework. The repository should remain recognizable to upstream users while adding the compatibility fixes, documentation, validation tools, and developer experience improvements needed for real custom-dataset work.

The intended outcome is a healthy GDRNPP fork that can be used directly for training, evaluation, and inference, and that can also serve as a reliable backend for larger future systems.

## Design Principles

### Stay Close to Upstream

Changes should be small, logical, and easy to review. A compatibility fix, dataset improvement, parser fix, or logging improvement should be understandable as its own patch.

### Modernize Before Rewriting

The first priority is to make the existing system build, run, train, and fail clearly on modern machines. Large architecture changes should wait until the modernization baseline is stable.

### Prefer Explicit Failures

Dataset and dependency failures should explain what failed, where it failed, and how to fix it. A clear validation report is better than a late `AssertionError`, `KeyError`, or `JSONDecodeError`.

### Treat Custom Datasets as First-Class

Official benchmark datasets are still important, but custom BOP-style datasets, synthetic datasets, BlenderProc outputs, and partially generated datasets should be documented and validated before training begins.

### Keep Fixes Reusable

Every fix should be clean enough that it could reasonably be proposed upstream or maintained independently. Avoid one-off hacks that only work for a single local dataset path or machine.

## Modernization Goals

This repository should support:

- Python 3.10 through 3.13 where feasible.
- PyTorch 2.x and current CUDA toolchains.
- NumPy 2.x compatibility.
- Updated dependency constraints with documented compatibility windows.
- Better handling of Detectron2, MMCV, YOLOX, and build dependencies.
- Modern AMP APIs such as `torch.amp.GradScaler("cuda", ...)`.
- Improved PLY parsing and mesh metadata generation.
- Dataset validation, repair, statistics, and visualization tools.
- Better logging, error messages, and troubleshooting docs.
- CI, unit tests, regression tests, and compatibility checks.
- Examples for custom datasets and synthetic data workflows.

## Developer Experience

Developers should be able to understand failures quickly. Instead of training crashing deep inside a loader because a sample is incomplete, the repository should provide tools that answer:

- Which files or annotations are missing?
- How many samples are affected?
- Is the dataset repairable?
- Can incomplete samples be skipped or removed?
- Which command should be run next?

Good developer experience also means predictable setup, clear docs, modern dependency files, reproducible examples, and a changelog that records compatibility decisions.

## Long-Term Maintenance

This project should be maintained like a reusable library rather than a private research folder. Important modernization work should be captured as focused commits and, where useful, patch files or design notes.

GitHub Issues should track actionable work items such as:

- Python 3.12 support.
- NumPy 2.x compatibility.
- Dataset path resolution fixes.
- Dataset validator implementation.
- Dead download link replacement.
- Checksum verification for downloaded assets.

Documentation should track the roadmap, architecture, design decisions, TODOs, and changes so that work is not lost while debugging training runs.

## Relationship to Future Projects

This repository is not OpenPose3D. It should not grow into a full perception platform with every model, foundation model, robotics adapter, and VLA pipeline.

OpenPose3D or a future pose toolkit should be a separate project that uses this repository as one backend among many. In that future architecture, GDRNPP Modernized is responsible for being a robust GDRNPP implementation, while the broader toolkit owns orchestration, multiple backends, robotics integration, foundation model integration, and higher-level APIs.

This project also serves as a stable backend for future 6D pose estimation tooling, synthetic data pipelines, and robotics applications.
