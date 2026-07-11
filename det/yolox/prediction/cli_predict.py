"""Command-line entry point for :class:`YOLOXPredictor`.

Run with::

    python -m det.yolox.prediction.cli_predict --config CONFIG.py \
        --checkpoint model_final.pth --source images/ --output predictions/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .yolox_predictor import YOLOXPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GDRNPP YOLOX prediction")
    parser.add_argument("--config", required=True, help="Detectron2 LazyConfig file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--source", required=True, help="Image file or image folder")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--conf-thres", type=float, help="Confidence threshold override")
    parser.add_argument("--nms-thres", type=float, help="NMS IoU threshold override")
    parser.add_argument("--test-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--class-names", nargs="*", help="Class names in class-id order")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA half precision")
    parser.add_argument("--fuse", action="store_true", help="Fuse convolution and batch norm")
    parser.add_argument("--augment", action="store_true", help="Use configured test-time augmentation")
    parser.add_argument("--class-agnostic", action="store_true", help="Use class-agnostic NMS")
    parser.add_argument("--draw", action="store_true", help="Return/draw annotated images")
    parser.add_argument("--save-images", action="store_true", help="Save annotated images")
    parser.add_argument("--save-json", action="store_true", help="Save JSON predictions")
    parser.add_argument("--benchmark", type=int, metavar="ITERATIONS", help="Benchmark after prediction")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.save_images or args.save_json) and args.output is None:
        raise SystemExit("--output is required with --save-images or --save-json")

    predictor = YOLOXPredictor(
        args.config,
        args.checkpoint,
        device=args.device,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        test_size=args.test_size,
        class_names=args.class_names,
        fp16=args.fp16,
        fuse=args.fuse,
        augment=args.augment,
        class_agnostic=args.class_agnostic,
    )
    results = predictor.predict(
        args.source,
        batch_size=args.batch_size,
        recursive=args.recursive,
        output_dir=args.output,
        draw=args.draw,
        save_images=args.save_images,
        save_json=args.save_json,
    )
    result_list = results if isinstance(results, list) else [results]
    summary = {
        "images": len(result_list),
        "detections": sum(len(result["detections"]) for result in result_list),
    }
    print(json.dumps(summary, indent=2))

    if args.benchmark:
        print(json.dumps(predictor.benchmark(iterations=args.benchmark, batch_size=args.batch_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
