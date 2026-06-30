#!/usr/bin/env python3
"""Standalone YOLOX inference for a single image or a folder of images.

This script is intentionally small and only depends on the existing GDRNPP / YOLOX
repo code. It does not use DatasetCatalog or the evaluator.

Examples
--------
python det/yolox/tools/infer.py \
    --config-file configs/yolox/bop_pbr/your_config.py \
    --ckpt output/yolox/.../model_final.pth \
    --source /path/to/image_or_folder \
    --output /path/to/save_dir
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import LazyConfig
from detectron2.engine.defaults import create_ddp_model

# Make repo imports work when running this file directly.
CUR_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(CUR_DIR, "../../../"))

from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer
from det.yolox.data import ValTransform
from det.yolox.utils import postprocess


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOX inference on a single image or folder")
    parser.add_argument("--config-file", required=True, help="Path to LazyConfig .py file")
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--source", required=True, help="Image file or folder of images")
    parser.add_argument("--output", required=True, help="Directory to save outputs")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--conf-thr", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--nms-thr", type=float, default=None, help="Override NMS threshold")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"), help="Override test size")
    parser.add_argument("--half", action="store_true", help="Run model in FP16")
    parser.add_argument("--augment", action="store_true", help="Use test-time augmentation if supported")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional config overrides: key=value ...")
    return parser.parse_args()


def load_cfg(config_file: str, opts: Optional[Sequence[str]] = None):
    cfg = LazyConfig.load(config_file)
    if opts:
        cfg = LazyConfig.apply_overrides(cfg, opts)

    # Minimal setup so the model config behaves as expected.
    dummy_args = SimpleNamespace(config_file=config_file, opts=opts or [], eval_only=True)
    default_yolox_setup(cfg, dummy_args)
    return cfg


def build_model(cfg, ckpt_path: str, device: str = "cuda"):
    model = YOLOX_DefaultTrainer.build_model(cfg)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to(device)

    # Use the same checkpointer path the repo uses during eval.
    MyCheckpointer(model, save_dir=cfg.train.output_dir).resume_or_load(ckpt_path, resume=False)
    model.eval()
    return model


def gather_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = [x for x in sorted(p.iterdir()) if x.suffix.lower() in IMG_EXTS]
        return files
    raise FileNotFoundError(f"Source not found: {source}")


def preprocess_image(img_bgr: np.ndarray, test_size: Tuple[int, int]):
    """Use the same preprocessing class as YOLOX test-time evaluation."""
    transform = ValTransform(legacy=False)
    processed, ratio = transform(img_bgr, None, test_size)
    tensor = torch.from_numpy(processed).unsqueeze(0)
    return tensor, ratio


def get_detection_tensor(outputs):
    """Normalize different possible model return formats into a tensor/list of detections."""
    det = outputs.get("det_preds", outputs)
    if isinstance(det, (list, tuple)):
        if len(det) == 1:
            return det[0]
        return det
    return det


def draw_detections(
    img_bgr: np.ndarray,
    dets,
    conf_thr: float = 0.25,
    class_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    out = img_bgr.copy()

    if dets is None:
        return out

    if isinstance(dets, (list, tuple)):
        dets = dets[0]

    if dets is None:
        return out

    if torch.is_tensor(dets):
        dets = dets.detach().cpu().numpy()

    if dets.size == 0:
        return out

    # Expected YOLOX format after postprocess:
    # [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
    for det in dets:
        if len(det) < 7:
            continue
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det[:7]
        score = float(obj_conf) * float(cls_conf)
        if score < conf_thr:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        label = f"{cls_id}:{score:.2f}"
        if class_names is not None and 0 <= cls_id < len(class_names):
            label = f"{class_names[cls_id]}:{score:.2f}"
        scale = 3 # Temporary scaling factor for visualization
        x1, y1, x2, y2 = [int(coord * scale) for coord in [x1, y1, x2, y2]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2) # Havn't resolved the scaling issue, so double the coordinates for now.
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return out


@torch.no_grad()
def predict_one(model, img_path: Path, cfg, device: str, half: bool = False, augment: bool = False):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    test_size = tuple(cfg.test.test_size)
    if len(test_size) != 2:
        raise ValueError(f"cfg.test.test_size must be (H, W), got {test_size}")

    tensor, ratio = preprocess_image(img_bgr, test_size)
    tensor = tensor.to(device)
    if half:
        tensor = tensor.half()

    outputs = model(tensor, augment=augment, cfg=cfg)
    dets = get_detection_tensor(outputs)
    dets = postprocess(
            dets,
            cfg.test.num_classes,
            cfg.test.conf_thr,
            cfg.test.nms_thr,
            )

    # print(type(dets))
    # print(len(dets))
    # print(type(dets[0]))
    return img_bgr, dets, ratio


def main():
    args = parse_args()
    cfg = load_cfg(args.config_file, args.opts)

    # Optional CLI overrides.
    if args.img_size is not None:
        cfg.test.test_size = tuple(args.img_size)
    if args.conf_thr is not None:
        cfg.test.conf_thr = args.conf_thr
    if args.nms_thr is not None:
        cfg.test.nms_thr = args.nms_thr

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Keep cfg consistent with runtime.
    cfg.train.device = device

    model = build_model(cfg, args.ckpt, device=device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = gather_images(args.source)
    print(f"Found {len(images)} image(s)")

    for img_path in images:
        img_bgr, dets, ratio = predict_one(
            model=model,
            img_path=img_path,
            cfg=cfg,
            device=device,
            half=args.half,
            augment=args.augment,
        )

        vis = draw_detections(
            img_bgr,
            dets,
            conf_thr=float(cfg.test.conf_thr),
            class_names=getattr(getattr(cfg, "metadata", None), "thing_classes", None),
        )

        stem = img_path.stem
        save_path = out_dir / f"{stem}_pred.png"
        cv2.imwrite(str(save_path), vis)

        # Save raw predictions too, useful for debugging.
        pred_path = out_dir / f"{stem}_pred.pt"
        torch.save(
            {
                "source": str(img_path),
                "ratio": ratio,
                "detections": dets.detach().cpu() if torch.is_tensor(dets) else dets,
            },
            pred_path,
        )

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
