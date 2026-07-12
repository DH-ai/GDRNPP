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

---

## Patch 2 — Scale optimizer lr by batch size (wire in `basic_lr_per_img`)

**File:** `det/yolox/engine/yolox_trainer.py` (`YOLOX_DefaultTrainer.build_optimizer`)

**Problem**

`configs/yolox/bop_pbr/yolox_base.py` defines `train.basic_lr_per_img = 0.01/64`
(the upstream YOLOX base lr per image, matching the `# bs=64` comments on the
optimizer lr), but nothing ever consumed it. `build_optimizer` instantiated the
config optimizer with its literal lr (`Ranger`, `lr=0.001`), so the learning
rate did **not** adapt to the batch size. An lr tuned for `bs=64` was applied
unchanged at `total_batch_size=4`, which is far too hot for the small batch and
hurts convergence/stability.

**Fix**

In `build_optimizer`, if `train.basic_lr_per_img` is set, compute
`lr = basic_lr_per_img * total_batch_size` and assign it to `cfg.optimizer.lr`
before instantiating (this is the standard YOLOX lr rule). For the current
config this yields `1.5625e-4 * 4 = 6.25e-4` instead of `1e-3`. The value is
now batch-size-consistent and tunable via `basic_lr_per_img`; set it to `None`
to keep the optimizer's literal lr.

**Note**

This changes training dynamics and should be confirmed on a GPU run; it is a
principled default, not something reproducible on the CPU-only setup here.

---

## Patch 3 — Make L1 loss robust and drop duplicated `get_l1_target` lines

**File:** `det/yolox/models/yolo_head.py` (`get_losses`, `get_l1_target`)

**Problem**

1. In `get_losses`, the `try/except` guarding the `loss_l1` computation does
   **not** assign `loss_l1` in the `except` branch. If the L1 term ever raises
   (e.g. a shape mismatch), execution continues to `loss_dict["loss_l1"] =
   loss_l1` and dies with `NameError: loss_l1`, masking the real error.
2. `get_l1_target` computes `l1_target[:, 2]`/`l1_target[:, 3]` twice (once
   before the finiteness check and again after the `raise`), the second pair
   being dead/redundant.

**Fix**

1. Initialize `loss_l1 = 0.0` before the `try`, so it is always defined; the
   `except` now degrades gracefully (skips the L1 term for that step) instead
   of crashing, while still logging the failure.
2. Remove the duplicated post-`raise` recomputation of `l1_target[:, 2:4]`.

These are correctness/cleanup fixes; the finiteness guard behavior is
unchanged. With Patch 1 in place the L1 term no longer receives degenerate
targets, so this path should not trigger in normal training.
