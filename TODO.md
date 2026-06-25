# TODO

Track actionable modernization work here until each item becomes a GitHub Issue or a completed patch.

## Environment Compatibility

- [ ] Document known-good Python, PyTorch, CUDA, NumPy, Detectron2, MMCV, and YOLOX combinations.
- [ ] Replace deprecated `torch.cuda.amp.GradScaler(...)` calls with the modern `torch.amp.GradScaler("cuda", ...)` API where compatible.
- [ ] Audit remaining deprecated PyTorch AMP APIs.
- [ ] Audit Python 3.12 and 3.13 compatibility issues.
- [ ] Audit NumPy 2.x compatibility issues beyond removed aliases.

## Dependencies and Downloads

- [ ] Replace dead download links.
- [ ] Download required assets automatically when missing.
- [ ] Verify checksums for downloaded assets.
- [ ] Add automatic pretrained weight management for missing checkpoints.
      Desired behavior:
      `Checking pretrained weights...`
      `YOLOX-X weights not found.`
      `Download now? [Y/n]`
      `Downloading...`
      `Verifying SHA256...`
      `Saved to: pretrained_models/yolox/yolox_x.pth`
      `Ready.`
- [ ] Separate required dependencies from optional development or visualization dependencies.
- [ ] Document offline or manual download fallback paths.

## Dataset Reliability

- [ ] Fix dataset path resolution issues discovered during custom training.
- [ ] Add a BOP-style dataset validator.
- [ ] Report missing RGB, depth, mask, `scene_gt`, `scene_gt_info`, and `scene_camera` entries before training starts.
- [ ] Add dataset repair utilities for deterministic safe repairs.
- [ ] Add dataset statistics output.
- [ ] Add dataset visualization checks for masks, bounding boxes, and pose annotations.

## Mesh Tooling

- [ ] Add tests for Blender 4.x PLY files.
- [ ] Validate support for `property list uchar uint vertex_indices`.
- [ ] Add or document mesh triangulation.
- [ ] Add `models_info.json` validation or generation.
- [ ] Add unit, diameter, bounding box, and symmetry checks.

## Developer Experience

- [ ] Replace remaining generic assertions with actionable errors.
- [ ] Add structured logging for dataset loading and training startup.
- [ ] Improve setup instructions for modern Linux environments.
- [ ] Add a custom dataset training example.
- [ ] Add troubleshooting entries for common dependency build failures.

## Tests and CI

- [ ] Add lightweight CI checks that do not require a GPU.
- [ ] Add unit tests for compatibility utilities.
- [ ] Add regression tests for dataset validation behavior.
- [ ] Add parser tests for PLY and mesh metadata handling.
- [ ] Add smoke tests for configuration loading.
