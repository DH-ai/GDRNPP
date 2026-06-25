# Contributing

## Scope

Contributions to this repository should improve GDRNPP as a modern, maintainable 6D pose estimation codebase.

Good contributions include:

- Python, NumPy, PyTorch, CUDA, Detectron2, MMCV, and YOLOX compatibility fixes.
- Dataset validation, repair, statistics, and visualization improvements.
- Mesh parsing and metadata tooling.
- Logging and error message improvements.
- Tests, CI, documentation, and examples.

Out-of-scope contributions include turning this repository into the full OpenPose3D toolkit, adding unrelated foundation model orchestration, or embedding robotics stack code directly into GDRNPP.

## Patch Guidelines

Keep changes focused.

Preferred patch size:

- One compatibility fix.
- One dataset behavior.
- One mesh parsing behavior.
- One documentation topic.
- One test target.

Avoid mixing broad refactors with urgent compatibility fixes.

## Documentation

When a change affects setup, supported versions, dataset expectations, or training commands, update the relevant docs.

Use:

- `docs/VISION.md` for repository purpose.
- `docs/MODERNIZATION_PLAN.md` for engineering phases.
- `docs/ROADMAP.md` for milestones.
- `docs/ARCHITECTURE.md` for boundaries and data flow.
- `docs/DESIGN_DECISIONS.md` for durable decisions.
- `TODO.md` for active known work.
- `CHANGELOG.md` for notable completed changes.

## Tests

Add tests when changing compatibility-sensitive code, parsers, dataset loading, or repair behavior.

Useful test targets:

- NumPy dtype compatibility.
- PLY parser variants.
- Dataset validator outputs.
- Missing file and missing annotation reporting.
- Utility functions used during training startup.

## Issue Tracking

Use GitHub Issues for actionable tasks.

Examples:

- Fix dataset path resolution.
- Add Python 3.12 support.
- Add NumPy 2.x compatibility.
- Replace dead links.
- Verify download checksums.

Use docs for the broader direction and rationale.
