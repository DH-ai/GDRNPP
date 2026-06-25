# Vision: GDRNPP Modernization

This document describes the near-term modernization effort for this repository. It is intentionally scoped to GDRNPP itself.

## Purpose

Modernize GDRNPP into a healthy, maintainable codebase that works with current Python, PyTorch, CUDA, NumPy, and dataset workflows while staying close to upstream.

The goal is not to build OpenPose3D here. The goal is to make GDRNPP reliable enough to use directly and clean enough to serve as a backend for future pose estimation toolkits.

## Immediate Scope

This modernization effort covers:

- Python 3.10 to 3.13 compatibility.
- NumPy 2.x compatibility.
- PyTorch 2.x compatibility.
- CUDA 12-era compatibility.
- Dependency cleanup and documented version ranges.
- GDRNPP, YOLOX, Detectron2, and MMCV compatibility fixes.
- Dataset path resolution fixes.
- BOP-style custom dataset support.
- Dataset validation and repair tools.
- PLY and mesh metadata improvements.
- Better logging, errors, docs, tests, and CI.

## Non-Goals

This repository should not become the full OpenPose3D toolkit.

Out of scope for this repository:

- A unified abstraction across GDRNPP, MegaPose, FoundationPose, and future pose models.
- Robotics stack integration.
- VLM or VLA orchestration.
- SAM, DINO, Florence, or open-vocabulary perception pipelines.
- A general data generation platform beyond what is needed to support GDRNPP workflows.

Those belong in a separate future repository.

## Patch Philosophy

Every modernization change should be easy to understand and easy to review.

Preferred commit shape:

- One compatibility target per commit.
- One dataset behavior fix per commit.
- One parser or mesh tooling fix per commit.
- One documentation or CI improvement per commit.

Examples:

- `Python 3.12 compatibility`
- `NumPy 2 compatibility`
- `Update AMP GradScaler API`
- `Improve PLY uint parsing`
- `Add custom dataset validator`
- `Fix dataset path resolution`

Where useful, patches can also be exported into a `patches/` directory later, but the current priority is clean commits and clear documentation.

## Weekend Success Criteria

A successful first modernization pass should make the repository easier to build, run, and debug on a modern workstation.

Minimum useful outcome:

- The existing training path runs far enough to expose real dataset or model issues, not dependency breakage.
- Known compatibility fixes are documented.
- Dataset failure modes are captured in `TODO.md` or issues.
- Dead links, missing downloads, and checksum gaps are listed as modernization tasks.
- The repository has a clear roadmap that separates GDRNPP modernization from the future OpenPose3D project.

## Future Backend Role

When OpenPose3D or a broader pose toolkit exists, this repository should be consumed as a backend:

```text
OpenPose3D / PoseToolkit
        |
        +-- GDRNPPBackend
        +-- MegaPoseBackend
        +-- FoundationPoseBackend
        +-- FutureBackends
```

The toolkit should adapt to GDRNPP, not mutate GDRNPP into the whole toolkit.
