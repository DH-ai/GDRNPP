# Architecture

## Current Repository Role

This repository is a modernized GDRNPP codebase. Its job is to keep the original GDRNPP training, evaluation, inference, detection, dataset, and utility paths working on modern environments.

The current source layout should remain stable during the first modernization phase:

```text
src/gdrnpp/
├── configs/
├── core/
├── det/
├── docs/
├── lib/
├── ref/
├── requirements/
├── scripts/
├── tools/
├── README.md
├── troubleshoot.md
└── vision.md
```

## Modernization Boundary

The modernization effort should improve code inside the existing GDRNPP boundaries:

- Compatibility fixes.
- Dataset loading and validation improvements.
- Mesh parsing and metadata utilities.
- Training and evaluation usability.
- Logging and error messages.
- Tests and CI.
- Documentation.

It should not introduce a new top-level platform architecture inside this repository.

## Dataset Flow

The intended GDRNPP dataset flow is:

```text
Custom data or synthetic data
        |
        v
BOP-compatible dataset
        |
        v
Dataset validation
        |
        v
Optional repair or cleanup
        |
        v
GDRNPP training / evaluation / inference
```

The validator should fail early with clear reports instead of allowing incomplete datasets to fail deep inside training.

## Mesh Flow

The intended mesh flow is:

```text
CAD / Blender mesh
        |
        v
Mesh validation and conversion
        |
        v
PLY parser and metadata checks
        |
        v
models_info.json validation or generation
        |
        v
Dataset registration and training
```

Mesh tooling should support modern Blender exports and catch unit, diameter, topology, and symmetry issues before training.

## External Toolkit Boundary

A future OpenPose3D or PoseToolkit repository should depend on this repository rather than being built inside it.

That future architecture may look like:

```text
OpenPose3D / PoseToolkit
        |
        +-- backends/
        |   +-- gdrnpp/
        |   +-- megapose/
        |   +-- foundationpose/
        |   +-- future_backends/
        |
        +-- dataset/
        +-- mesh/
        +-- evaluation/
        +-- robotics/
        +-- foundation_models/
```

In that setup, GDRNPP Modernized is an implementation backend, not the center of the whole platform.

## Patch Organization

During modernization, prefer focused changes:

- One compatibility target per patch.
- One parser behavior per patch.
- One dataset behavior per patch.
- One developer experience improvement per patch.

If this becomes a long-lived fork, a future `patches/` directory can be added to export upstream-relative patches, but clean commits are the first priority.
