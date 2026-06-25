# Vision

> Building the next-generation open-source toolkit for custom 6D object pose estimation, synthetic data generation, and robotic perception.

---

# Motivation

GDRNPP is one of the strongest open-source frameworks for 6D object pose estimation. However, it was primarily designed for benchmark datasets such as BOP, where the data is assumed to be complete, consistent, and correctly formatted.

During the integration of custom synthetic datasets generated with BlenderProc, several shortcomings became apparent.

The current framework assumes that every dataset is perfect.

Typical examples include:

```python
assert osp.exists(rgb_path)
assert osp.exists(depth_path)

gt_info[str_im_id]
```

When something is missing, the program immediately crashes with errors such as

```
AssertionError
KeyError: '112'
JSONDecodeError
```

without answering the questions developers actually need:

* What is missing?
* How many samples are affected?
* Can the dataset be repaired?
* Is the entire dataset corrupted or only a few images?
* Can training safely continue?

While this assumption works well for official benchmark datasets, it becomes a major obstacle when working with custom datasets, synthetic data generation pipelines, or large-scale experimentation.

This project aims to bridge that gap.

---

# Vision

The goal is not simply to patch compatibility issues or support one custom dataset.

The vision is to transform GDRNPP into a modern, extensible, and developer-friendly platform covering the complete lifecycle of 6D object perception.

The framework should support everything from CAD model preparation and synthetic data generation to dataset validation, model training, evaluation, deployment, and robotic integration.

Rather than assuming datasets are correct, the framework should help users build correct datasets.

Long term, the project aims to become the standard development environment for custom 6D pose estimation research and production deployments.

---

# Core Principles

## Reliability over Assumptions

Datasets should never be assumed to be valid.

Every dataset should be validated before training begins.

Missing files, corrupted annotations, inconsistent metadata, partial renders, invalid meshes, interrupted exports, and inconsistent dataset structures should be detected automatically.

Errors should be descriptive rather than cryptic.

---

## Automation over Manual Repair

Most dataset issues are deterministic.

If a problem can be repaired automatically, the framework should provide a repair utility instead of requiring users to manually edit JSON files or regenerate datasets.

Examples include:

* rebuilding `scene_gt_info`
* rebuilding `scene_camera`
* regenerating bounding boxes from masks
* rebuilding `models_info.json`
* triangulating meshes
* repairing interrupted dataset exports
* removing incomplete samples
* validating BOP compliance

---

## Modern Software Engineering

The original repository reflects the Python ecosystem available when it was created.

This project modernizes the stack by supporting:

* Python 3.10+
* Modern PyTorch
* Current Detectron2
* NumPy 2.x
* Current CUDA toolchains
* Improved dependency management
* Type hints
* Structured logging
* Unit tests
* CI/CD pipelines
* Modular architecture
* Comprehensive documentation

---

## Developer Experience

Good tooling is as important as good models.

Instead of failing with

```
KeyError: '112'
```

the framework should report

```
Dataset Validation Error

Scene : 000000
Image : 112

Missing

✓ RGB
✓ Depth
✗ scene_gt_info
✓ Masks

Suggested Fix

python tools/dataset/repair.py \
    --scene 000000 \
    --image 112
```

Developers should spend their time improving models—not debugging datasets.

---

# Documentation First

Excellent software deserves excellent documentation.

The project should provide documentation for users at every experience level.

Documentation should include:

* Beginner setup guides
* Installation on Linux, Windows, and Docker
* Dataset creation walkthroughs
* BlenderProc integration tutorials
* BOP format explanations
* Custom dataset onboarding
* Training pipelines
* Evaluation workflows
* Inference examples
* API documentation
* System architecture documentation
* Developer guides
* Contribution guides

A new user should be able to go from a CAD model to a trained pose estimator without reading source code.

---

# Dataset-First Design

Custom datasets are first-class citizens.

The repository should include tooling for:

* dataset validation
* automatic repair
* visualization
* statistics
* mesh verification
* mesh conversion
* annotation verification
* metadata generation
* dataset versioning
* dataset comparison
* BOP compatibility checking

Training should never begin before dataset integrity has been verified.

Example:

```
Loading dataset...

Checking integrity...

✓ RGB Images ............. 1000
✓ Depth Images ........... 1000
✓ Masks .................. 1000
✓ scene_gt.json .......... OK
✗ scene_gt_info.json ..... Missing 1 entry

Found 1 corrupted sample.

Run:

python tools/dataset/repair.py
```

instead of discovering the issue several minutes later through a runtime exception.

---

# Mesh Processing Pipeline

Preparing CAD models is often one of the most time-consuming parts of building a custom dataset.

The framework should automate:

* mesh validation
* triangulation
* topology verification
* manifold checking
* watertight verification
* unit verification
* symmetry definition
* diameter computation
* bounding box computation
* coordinate system validation
* `models_info.json` generation

The goal is to eliminate repetitive manual preprocessing.

---

# Robust Dataset Generation

Synthetic dataset generation should be resilient to interruptions.

Interrupted rendering currently results in problems such as:

* partially written JSON files
* missing RGB images
* missing depth maps
* missing masks
* inconsistent image counts
* corrupted annotations

Future exporters should use validation and atomic writes so partially generated datasets can always be detected, repaired, or resumed safely.

---

# Support for Multiple Pose Estimation Models

The toolkit should not be tied to a single research paper.

GDRNPP is only one backend.

Future support should include additional 6D pose estimation models through a unified interface.

Examples include:

* GDRNPP
* FoundationPose
* MegaPose
* CosyPose
* Gen6D
* SurfEmb
* Foundation-model-based pose estimators
* Future research models

Users should be able to switch models with minimal configuration changes while reusing the same datasets, preprocessing, and evaluation pipelines.

---

# BOP as the Standard Dataset Interface

The BOP dataset format has become the de facto standard for 6D pose estimation.

The toolkit should adopt BOP as its internal representation.

Every supported dataset should be importable into a BOP-compatible structure.

Similarly, outputs should remain compatible with existing BOP evaluation tools.

This allows interoperability across multiple training frameworks and research projects.

---

# Foundation Models and Vision-Language Integration

Modern robotic systems increasingly rely on foundation models.

Future development should include optional integration with:

* Vision-Language Models (VLMs)
* Vision-Language-Action (VLA) models
* Open-vocabulary object detection
* Referring expression grounding
* Semantic scene understanding
* Automatic dataset annotation

The objective is to bridge traditional geometric pose estimation with modern multimodal perception systems.

---

# Modular Architecture

Rather than treating functionality as standalone scripts, the repository should be organized into reusable modules.

```
dataset/
├── loaders/
├── validators/
├── repair/
├── converters/
├── visualization/
└── statistics/

mesh/
├── validation/
├── processing/
└── conversion/

training/
├── detection/
├── pose/
└── evaluation/

models/
├── gdrnpp/
├── megapose/
├── foundationpose/
└── ...

tools/
├── train_detection.py
├── train_pose.py
├── infer.py
├── evaluate.py
├── export.py
└── dataset/
```

Each module should have a single responsibility while exposing reusable APIs for future extensions.

---

# Long-Term Goal

The long-term objective extends beyond maintaining an improved fork of GDRNPP.

The project aims to become a complete open-source perception toolkit.

```
OpenPose3D Toolkit

├── CAD Model Processing
├── Synthetic Data Generation
├── Dataset Validation
├── Dataset Repair
├── Dataset Conversion
├── Dataset Visualization
├── Dataset Statistics
├── Object Detection Training
├── 6D Pose Training
├── Multiple Pose Estimation Backends
├── Foundation Model Integration
├── Evaluation
├── Deployment
├── Robotics Integration
└── Production Tooling
```

GDRNPP serves as the foundation, but the toolkit expands beyond it by emphasizing robustness, automation, reproducibility, extensibility, and developer experience.

The ultimate goal is to provide a framework that enables researchers and engineers to spend less time debugging datasets and infrastructure, and more time building better perception systems.
