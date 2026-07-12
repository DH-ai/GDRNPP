# YOLOX Training Patch Report

Running log of patches applied to the YOLOX detection training path. One entry
per patch; one commit per patch.

---

## Patch 1 — Use the actual image size (not declared record dims) for GT bbox clipping/scaling

**File:** `det/yolox/data/datasets/base_data_from_list.py` (`Base_DatasetFromList.load_anno`, new `_get_image_hw`)

**Symptom reported:** YOLOX training does not converge (previously also produced
`l1_loss = inf`).

**Root cause**

`load_anno()` read the *declared* per-record dimensions
(`dataset_dict["width"]`, `dataset_dict["height"]`) and used them for two things:

1. Clipping every GT box: `x2 = min(x2, width)`, `y2 = min(y2, height)`.
2. Computing the box resize ratio: `r = min(img_size[0]/height, img_size[1]/width)`.

But the image pixels come from `load_resized_img()` / `preproc()`, which compute
their resize ratio from the **actual image file** dimensions. The training
dataset config (`det/yolox/data/datasets/mydataset_pbr.py`,
`SPLITS_MY_DATASET_PBR`) declares `height=600, width=960`, while the rendered
BOP images are `1200x1920` (see `ref/mydataset.py`: `width=1920, height=1200`).

Because the declared dims are half the real dims:
- **Clipping** dropped/truncated every GT box whose coordinates exceeded
  960/600, i.e. everything in the right/bottom of the frame.
- **Scaling** multiplied boxes by ~2x the ratio applied to the image, so even
  the surviving boxes no longer sit on their objects.

The result is systematically wrong (and mostly missing) GT targets, so the
detector cannot learn. Degenerate/zero-area boxes from the same clipping are
also the likely origin of the earlier `l1_loss = inf`.

**Fix**

Derive `(height, width)` in `load_anno()` from the actual image on disk (via a
header-only `PIL.Image.open`, no full decode), so clipping and the resize ratio
always match the pixels produced by `load_resized_img()`. This is correct
regardless of the declared dims (full-res *or* a downscaled dataset), and
restores the YOLOX invariant that annotation dims == image dims.

**Verification (real code, CPU)**

Ran the actual `Base_DatasetFromList` pipeline on a generated BOP frame
(`src/output/bop/train_pbr/000000`, real size `1200x1920`, 6 target instances)
with the buggy declared dims `600x960`:

| | GT boxes kept | Boxes aligned to objects |
|---|---|---|
| Before patch | **1 / 6** | No (floats in empty space, ~2x off) |
| After patch  | **6 / 6** | Yes |

Overlaying the loader's returned targets on the network-input image confirms the
boxes land on the objects only after the patch.

**Secondary observations (NOT changed here — flagged for follow-up)**

- `configs/.../yolox_base.py` `basic_lr_per_img` is unused: `build_optimizer`
  instantiates the config optimizer (`Ranger`, `lr=0.001`) as-is with no
  batch-size lr scaling. With `total_batch_size=4` verify `lr` is appropriate.
- `det/yolox/models/yolo_head.py get_losses`: the `try/except` around
  `loss_l1` does not assign `loss_l1` in the `except` branch, which would raise
  `NameError` when building `loss_dict` if the L1 computation ever throws.
  `get_l1_target` also computes `l1_target[:, 2:4]` twice (redundant).
