# GDRNPP Modernized Vision

## Motivation

GDRNPP is a strong 6D object pose estimation baseline, but the original codebase reflects the Python, PyTorch, CUDA, NumPy, Detectron2, and dataset tooling ecosystem of its time.

This repository exists to keep GDRNPP usable on modern systems without turning it into a different project.

## Current Problems

The modernization work is driven by recurring issues:

- Old Python APIs and imports.
- NumPy aliases removed in NumPy 2.x.
- PyTorch 2.x and AMP API changes.
- CUDA and build dependency drift.
- Detectron2, MMCV, and YOLOX compatibility gaps.
- PLY parsing assumptions that fail on newer Blender exports.
- Dataset failures that appear late and produce unclear errors.
- Limited documentation for custom datasets and synthetic data workflows.

## Project Vision

GDRNPP Modernized should be the version of GDRNPP people would expect if the project were actively maintained in 2026.

It should:

- Build on modern environments.
- Train and evaluate using current PyTorch and CUDA stacks.
- Support BOP-style custom datasets.
- Provide clear validation and repair tooling.
- Document setup, training, evaluation, troubleshooting, and compatibility decisions.
- Stay close enough to upstream that fixes remain reviewable.

## Design Principles

- Stay close to upstream unless a deeper change is necessary.
- Prefer focused compatibility patches over broad rewrites.
- Make dataset failures clear before training starts.
- Keep custom datasets and synthetic data workflows documented.
- Treat documentation, tests, and CI as part of the modernization effort.
- Keep this repository focused on GDRNPP, not a full pose estimation platform.

## Modernization Goals

The repository should support:

- Python 3.10 through 3.13 where feasible.
- NumPy 2.x.
- PyTorch 2.x.
- CUDA 12-era development environments.
- Modern dependency constraints.
- Robust PLY and mesh metadata handling.
- Dataset validation, repair, statistics, and visualization.
- Better logging and actionable error messages.
- CI and regression tests for compatibility-sensitive code paths.

## Developer Experience

Developers should be able to answer these questions quickly:

- Is my environment supported?
- Are my dataset files complete?
- Are my BOP annotations consistent?
- Can a dataset issue be repaired automatically?
- Which command should I run next?

The project should favor clear setup instructions, explicit validation reports, predictable config behavior, and troubleshooting notes over silent assumptions.

## Long-Term Maintenance

Modernization work should be tracked through focused commits, docs, TODO entries, and GitHub Issues.

GitHub Issues should be used for actionable tasks. Documentation should capture broader direction, rationale, and decisions.

## Relationship to Future Projects

This repository is not OpenPose3D. It is the maintained GDRNPP backend that a future OpenPose3D or PoseToolkit project can depend on.

OpenPose3D should live in a separate repository with its own architecture and vision. It can integrate GDRNPP Modernized alongside MegaPose, FoundationPose, FoundationStereo, SAM, DINO, Florence, YOLO, VLMs, VLA pipelines, and robotics stacks.

This repository should remain focused: modernize GDRNPP, keep it healthy, and make it reliable as a reusable backend.
