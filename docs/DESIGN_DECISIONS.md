# Design Decisions

## Decision: GDRNPP Modernized Is Separate From OpenPose3D

GDRNPP Modernized and OpenPose3D have different scopes, users, and lifecycles.

GDRNPP Modernized is an upstream-quality modernization effort:

- Python compatibility.
- NumPy compatibility.
- PyTorch and CUDA compatibility.
- Dependency cleanup.
- Dataset validation and repair.
- Mesh tooling.
- Documentation, tests, and CI.

OpenPose3D is a future platform:

- Multiple pose backends.
- Dataset and mesh orchestration.
- Foundation model integrations.
- Robotics and VLA pipelines.
- Evaluation and deployment tooling.

These should not be merged into one repository too early.

## Decision: Use a Separate Fork or Repository for Long-Term GDRNPP Work

For long-term maintenance, prefer a separate fork or repository such as:

```text
gdrnpp-modernized
```

This repository should stay close to upstream GDRNPP and carry clean modernization commits.

Recommended ownership:

- If you want to preserve upstream history and make patches reviewable, fork upstream GDRNPP.
- If upstream history is already embedded in this repo and you are not ready to extract it today, continue on a dedicated branch now, then split or fork later.
- Do not start from scratch unless the goal is the future OpenPose3D toolkit.

## Decision: Use a Branch for Short-Term Weekend Work

Inside the current repo, use a dedicated branch for the GDRNPP modernization work:

```bash
git switch -c gdrnpp-modernized
```

Use this branch for compatibility fixes, docs, validators, and training blockers while the main branch remains focused on synthetic data generation work.

This is the best immediate workflow when the current repository still contains both the data generation work and the vendored GDRNPP source.

## Decision: Do Not Keep Vendor Patches Hidden

Avoid silently modifying third-party code without tracking why.

Preferred approach:

- Keep each change as a focused commit.
- Write clear commit messages.
- Document compatibility decisions.
- Add TODO entries or issues for incomplete work.

Later, if this becomes a maintained fork, consider exporting logical patches:

```text
patches/
├── 0001-python312-compatibility.patch
├── 0002-numpy2-compatibility.patch
├── 0003-ply-uint-support.patch
└── 0004-dataset-validation.patch
```

## Decision: GitHub Issues Are for Actionable Work

Use docs for vision, architecture, and plans.

Use GitHub Issues for concrete work items:

- Fix dataset path resolution.
- Add NumPy 2.x compatibility.
- Add Python 3.12 support.
- Add dataset validator.
- Replace dead download links.
- Download assets automatically if missing.
- Verify checksums for downloaded assets.
- Update deprecated AMP calls.

If an issue is discovered while debugging training, add it to `TODO.md` immediately or open an issue before switching context.

## Decision: OpenPose3D Should Depend on GDRNPP Modernized

OpenPose3D should treat GDRNPP as a backend, not as the core architecture.

The future toolkit should define an interface such as:

```python
class PoseBackend:
    def train(self, config):
        raise NotImplementedError

    def infer(self, inputs):
        raise NotImplementedError

    def evaluate(self, dataset):
        raise NotImplementedError
```

Then GDRNPP, MegaPose, FoundationPose, and future models can implement adapters without forcing dataset, evaluation, and robotics code to depend on one model.

## Decision: Keep This Repository Focused

The modernization repository may mention future backend usage, but it should not absorb the full OpenPose3D roadmap.

The correct ending for this repo's vision is:

> This project also serves as a robust backend for future pose estimation toolkits.

That is enough.
