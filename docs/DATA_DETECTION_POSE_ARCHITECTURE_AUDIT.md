# Data, Detection, and Pose Architecture Audit

This report is a passive code-reading audit of the current GDRNPP submodule. No
training, evaluation, inference, dataset conversion, or project script was run.

## Executive Summary

The repo is two coupled systems:

```text
BOP-style dataset
        |
        +-- det/yolox             -> object detection boxes
        |
        +-- core/gdrn_modeling    -> ROI crop, geometry prediction, pose
```

The detector and pose sides both use Detectron2 `DatasetCatalog`/
`MetadataCatalog`, but they do not share one dataset implementation. The custom
dataset exists in both `det/yolox/data/datasets/mydataset_pbr.py` and
`core/gdrn_modeling/datasets/mydataset_pbr.py`, and the copies have diverged:

- YOLOX custom loader expects `rgb/*.png`.
- GDRN custom loader expects `rgb/*.jpg`.
- YOLOX custom train/test splits declare `height=600,width=960`.
- GDRN custom splits declare `height=1200,width=1920`.
- `ref/mydataset.py` declares the camera and original image size as
  `1920x1200`.

The clean detector replacement boundary is the detection JSON consumed by
`core.utils.dataset_utils.load_detections_into_dataset`. Any detector, including
YOLOv11, can be used if it exports `obj_id`, `bbox_est` as original-image
`xywh`, `score`, and optional `time`, grouped by `scene_id/image_id`.

The manual scale factor of `3` comes from YOLOX letterbox inference:
`1920x1200 -> 640x640` gives `ratio = min(640/1200, 640/1920) = 1/3`.
Postprocessed YOLOX boxes are in model-input coordinates and must be divided by
that ratio. Multiplying by `3` is only this one special case.

## Folder Responsibilities

### `configs/`

- `configs/yolox/`: Detectron2 LazyConfig/OmegaConf detector configs.
- `configs/gdrn/`: MMCV-style GDRN pose configs.
- `configs/_base_/`: shared GDRN defaults.

The split works, but dataset root, image size, object list, camera metadata, and
detector output paths are not centralized.

### `core/`

`core/` owns GDRNPP pose estimation and shared training utilities.

- `core/gdrn_modeling/datasets/`: pose-side BOP registration and ROI mappers.
- `core/gdrn_modeling/models/`: GDRN modules, backbones, heads, losses, pose
  conversion.
- `core/gdrn_modeling/engine/`: LightningLite train/test loops and evaluators.
- `core/gdrn_modeling/tools/`: dataset-specific legacy helper scripts.
- `core/csrc/`: CUDA/C++ helpers for FPS, RANSAC, flow, NND, uncertainty PnP.
- `core/utils/`: shared data, image, geometry, optimizer, scheduler,
  checkpoint, and distributed utilities.

### `det/`

`det/` owns detection. The active stack is YOLOX:

- `det/yolox/models/`: YOLOX model, CSPDarknet/PAN-FPN, head, losses.
- `det/yolox/data/`: preprocessing, mosaic/mixup, dataloading, detector-side
  BOP dataset copies.
- `det/yolox/engine/`: training, inference, setup, standalone predictor.
- `det/yolox/evaluators/`: COCO/BOP result conversion.
- `det/yolox/tools/`: train/export/demo helpers.

### `lib/`

`lib/` is low-level support:

- `lib/pysixd/`: BOP/SIXD IO, pose utilities, pose errors.
- `lib/egl_renderer`, `lib/meshrenderer`, `lib/render_vispy`: rendering.
- `lib/torch_utils`: layers, optimizers, schedulers, Torch helpers.
- `lib/utils`: masks, logging, filesystem, config utilities.
- `lib/vis_utils`: visualization.

### `ref/`

`ref/` defines dataset metadata: object IDs/names, dataset roots, model dirs,
camera intrinsics, image dimensions, and model metadata loaders.

For the custom dataset, `ref/mydataset.py` currently hard-codes an absolute
BlenderProc output root, `width=1920`, `height=1200`, and the original-resolution
camera matrix. Treat this as the canonical dataset geometry unless you create a
true downsampled BOP dataset.

### `scripts/` and `tools/`

`scripts/` has setup helpers. Top-level `tools/` has small utility scripts for
inference, BOP result merging, timing cleanup, and checkpoint cleanup. Most
dataset-specific tools are under `core/gdrn_modeling/tools/`.

## BOP Dataset Loading

### Registration

The BOP path is:

```text
config DATASETS.TRAIN / DATASETS.TEST
        |
        v
register_datasets_in_cfg(...)
        |
        v
DatasetCatalog.register(name, DatasetCallable(...))
MetadataCatalog.get(name).set(...)
```

Both YOLOX and GDRN call `get_detection_dataset_dicts(...)` after registration.
This is useful, but it means the dataset records must satisfy both Detectron2
and custom pose-estimation assumptions.

### Files Read

The custom BOP loaders read:

```text
scene_gt.json
scene_gt_info.json
scene_camera.json
rgb/
depth/
mask/
mask_visib/
models/models_info.json
models/obj_000001.ply
```

Each image becomes a Detectron2-style record with image path, depth path,
height, width, `scene_im_id`, camera matrix, depth factor, and annotations.

Each annotation carries contiguous class ID, BOP bbox, pose, quaternion,
translation in meters, projected 2D centroid, visible/full masks, visibility,
model info, `xyz_path`, and model 3D bbox metadata.

### GDRN Train Loader

`core/gdrn_modeling/datasets/data_loader.py` builds the pose train loader:

```text
get_detection_dataset_dicts
        |
        v
filter invalid annotations
        |
        v
GDRN_DatasetFromList or GDRN_Online_DatasetFromList
        |
        v
flat_dataset_dicts: one ROI instance per sample
        |
        v
read full image/depth/masks/xyz
        |
        v
Detectron2 resize transform
        |
        v
scale camera K if full image was resized
        |
        v
choose bbox type and apply DZI augmentation
        |
        v
crop_resize_by_warp_affine to INPUT_RES/OUTPUT_RES
        |
        v
return ROI tensors and pose targets
```

The full image can be large, but the pose model trains on fixed ROI crops,
commonly `INPUT_RES=256` and `OUTPUT_RES=64`.

### GDRN Test Loader

At test time, `build_gdrn_test_loader(...)` optionally calls
`load_detections_into_dataset(...)`. That replaces/augments image annotations
with detector proposals:

```python
obj_id = det["obj_id"]
bbox_est = det["bbox_est"]  # xywh
score = det.get("score", 1.0)
time = det.get("time", 0.0)
```

These boxes should be in the same coordinate system as the image record loaded
by GDRN. In practice, use original BOP image coordinates unless you have
regenerated the whole BOP dataset at a smaller size.

## YOLOX Implementation

### Config

`configs/yolox/bop_pbr/yolox_base.py` defines the default YOLOX model, optimizer,
data loaders, evaluator, train `img_size=(640,640)`, and `test.test_size=(640,640)`.

The custom config
`yolox_x_1920_augCozyAAEhsv_ranger_30_epochs_mydataset_pbr_mydataset_test_primesense.py`
sets custom datasets, `num_classes=3`, YOLOX-X depth/width, batch size, and
augmentation. The filename says `1920`, but the inherited model input remains
`640x640` unless explicitly overridden.

### Data Path

`det/yolox/data/datasets/Base_DatasetFromList` wraps Detectron2 records for
YOLOX. It:

- reads `dataset_dict["height"]` and `dataset_dict["width"]`;
- clips boxes to those dimensions;
- computes label scale from those dimensions;
- separately reads the actual image and computes image scale from actual shape.

Therefore, dataset record dimensions must match the physical image files. If the
record says `960x600` but the image is actually `1920x1200`, boxes and images
are scaled differently.

### Preprocessing and Output Coordinates

YOLOX `preproc` keeps aspect ratio and pads to the target input:

```python
r = min(input_h / image_h, input_w / image_w)
```

For `1920x1200 -> 640x640`, `r = 1/3`. Raw postprocessed boxes are still in
the `640x640` padded input coordinate system. To draw or export in original
coordinates:

```python
boxes_original = boxes_model_input / r
```

The built-in YOLOX demo does this. The custom standalone predictor stores the
ratio but uses a hard-coded `scale = 3` when drawing, which should be replaced.

### Model

The detector is plain PyTorch:

- `YOLOX.forward` calls backbone/FPN and head.
- `YOLOPAFPN` builds PAN/FPN features from `CSPDarknet`.
- `YOLOXHead` predicts box, objectness, and classes at strides 8/16/32.
- Training uses YOLOX dynamic-k assignment and IoU/object/class losses.
- Inference uses `postprocess(...)` for confidence filtering and NMS.

## Replacing YOLOX

Do not replace GDRN internals just to use another detector. Replace or bypass
YOLOX at the detection-file boundary:

```text
YOLOv11 or other detector
        |
        v
adapter/exporter
        |
        v
GDRN detection JSON in original-image xywh
        |
        v
load_detections_into_dataset
        |
        v
GDRN ROI pose estimation
```

Minimum detection record:

```python
{
    "0/0": [
        {"obj_id": 1, "bbox_est": [x, y, w, h], "score": 0.99, "time": 0.0}
    ]
}
```

Rules:

- key must match `record["scene_im_id"]`;
- `obj_id` is the BOP object ID, not necessarily detector contiguous class ID;
- `bbox_est` is `xywh`;
- boxes are in original BOP coordinates unless the whole BOP dataset is
  consistently downsampled.

Lowest-risk YOLOv11 path:

1. Train/run YOLOv11 outside this repo or in a separate backend folder.
2. Convert class IDs to `ref/mydataset.py` BOP `obj_id`.
3. Undo the detector's resize/letterbox transform.
4. Convert `xyxy` to `xywh`.
5. Group by `scene_im_id`.
6. Feed the JSON to GDRN through `DATASETS.DET_FILES_TEST`.

If detector code must live in this repo, add a `det/common/` schema and
exporter instead of making GDRN know about YOLOX or YOLOv11 internals.

## GDRNPP PyTorch Implementation

`core/gdrn_modeling/main_gdrn.py` is the pose entrypoint. It loads an MMCV
config, registers datasets, sets up LightningLite, builds an optional renderer,
builds the selected GDRN module, and runs train/test/save-results mode.

`core/gdrn_modeling/models/GDRN.py::build_model_optimizer` builds:

```text
backbone -> optional neck -> geo head -> PnP head -> GDRN module
```

Backbones come from timm, torchvision, MM-style networks, PVNet-style networks,
or Detectron2 ResNet wrappers. Heads include top-down mask/xyz/region heads and
ConvPnP/PointPnP variants.

Forward pass:

```text
ROI image
        |
        v
backbone / neck
        |
        v
geometry head
        +-- mask
        +-- xyz coordinates
        +-- region logits
        |
        v
PnP head
        +-- rotation representation
        +-- centroid/z or translation
        |
        v
pose_from_pred...
        |
        v
rotation matrix + translation
```

Losses are manually composed in `GDRN.gdrn_loss(...)`: xyz, mask, region, point
matching, rotation, centroid, z, translation, bind, and optional multi-task
loss weighting.

Inference uses the GDRN evaluator to turn model outputs into BOP pose results.
It can use direct network pose output, optional PnP from predicted coordinates,
and optional depth refinement.

## Image Size and Bounding-Box Scaling

There are four coordinate spaces:

1. Original BOP image space, likely `1920x1200`.
2. Optional downsampled dataset image space, e.g. `960x600`.
3. YOLOX model input space, e.g. padded `640x640`.
4. GDRN ROI crop space, e.g. `256x256` input and `64x64` output.

### Factor 3

For full-resolution images with YOLOX `test_size=(640,640)`:

```text
ratio = min(640/1200, 640/1920) = 1/3
```

So:

```python
boxes_1920x1200 = boxes_640_input / (1/3)
```

This is why multiplying by `3` appears to work. The general fix is `boxes /=
ratio`.

### Factor 2

If the image actually used for comparison is `960x600` but the BOP annotations
are `1920x1200`, the factor is `2`. That is a dataset-coordinate mismatch, not
a YOLOX model-output issue.

### Correct Size Policies

Use one policy, not a mixture.

#### Policy A: Keep Canonical BOP at 1920x1200

Recommended for pose correctness.

- Keep BOP annotations, masks, depth, camera intrinsics, and `ref/mydataset.py`
  at `1920x1200`.
- Set dataset records to `height=1200,width=1920`.
- Train YOLOX at a smaller input size for GPU memory, such as `(600,960)` or
  `(640,640)`.
- Export detector boxes back to original coordinates by dividing by the detector
  preprocessing ratio.
- Feed original-coordinate `bbox_est` values to GDRN.

If GPU allows it, `(600,960)` is cleaner than `(640,640)` because it preserves
the original aspect ratio and uses `ratio=0.5` without bottom padding.

#### Policy B: Build a True 960x600 BOP Dataset

Only do this if you want the dataset files themselves smaller.

You must scale all coupled assets:

- RGB images by `0.5`.
- masks by `0.5` with nearest-neighbor interpolation.
- depth spatial resolution by `0.5`; depth values unchanged.
- `scene_gt_info.json` bboxes by `0.5`.
- `scene_camera.json` `fx`, `fy`, `cx`, `cy` by `0.5`.
- `ref/mydataset.py` `width`, `height`, and `K`.
- generated coordinate maps to the resized grid.

Do not only change loader dimensions.

## Modernization Problems

- Hard-coded absolute paths in custom dataset/ref files.
- Duplicated detector and pose dataset loaders.
- PNG/JPG mismatch between custom loaders.
- Image-size metadata drift between YOLOX, GDRN, and `ref`.
- Detection JSON schema is implicit, not validated.
- YOLOX and GDRN use different config systems with no shared dataset manifest.
- Pickle caches can hide stale dataset metadata.
- Detector-specific code leaks into workflows that only need boxes.

## Recommended Architecture Improvements

1. Add a documented detection schema and validator.
2. Fix the custom dataset geometry so image files, `height`, `width`, `K`, masks,
   and bboxes agree.
3. Replace hard-coded predictor scaling with `bbox /= ratio`.
4. Move shared BOP parsing into one module used by both detector and pose.
5. Add detector adapters that export the same GDRN detection JSON.
6. Make dataset root, image extension, object map, and image size configurable.
7. Add a BOP validator for missing files, mismatched dimensions, invalid boxes,
   missing `scene_gt_info`, and camera-scale mismatches.
8. Keep GDRN model internals stable until dataset and detector boundaries are
   clean.

## Practical Recommendation for This Dataset

For the current `1920x1200` data and GPU limit:

1. Keep canonical BOP data at `1920x1200`.
2. Set YOLOX custom dataset metadata back to `height=1200,width=1920` unless
   the physical images are truly resized.
3. Train YOLOX with `img_size=(600,960)` if possible, or keep `640x640` and
   always export boxes back with `boxes /= ratio`.
4. Feed GDRN detector boxes in original `1920x1200` `xywh`.
5. If you want true `960x600` data, regenerate all BOP JSON, masks, depth
   images, coordinate maps, and camera intrinsics consistently.

This removes the manual scale factors and gives a clean path for YOLOv11 or any
other detector.
