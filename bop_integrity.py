#!/usr/bin/env python3
"""BOP-style dataset integrity checker and parser.

This script scans a BOP-like dataset root containing scene folders such as:

    train_pbr/
        000000/
            rgb/
            depth/
            mask/
            mask_visib/
            scene_gt.json
            scene_gt_info.json
            scene_camera.json
        000001/
        ...

Default bbox source:
    - bbox_visib from scene_gt_info.json (recommended for detection)
    - fallback to bbox_obj if bbox_visib is missing

Usage
-----
python bop_data_integrity_checker.py \
    --root /path/to/train_pbr \
    --bbox-source visib \
    --cache-out /path/to/cache.pkl

You can also point --root to the parent folder containing scenes.
"""
#  TODO: Make this file part of modernize gdrnpp repo
# lib/
# └── data_integrity/
#     ├── __init__.py
#     ├── checker.py
#     ├── parser.py
#     ├── validators.py
#     ├── report.py
#     └── cache.py

# =============================================================================
# TODO: Replace the current ad-hoc BOP dataset loading with a dedicated
# Dataset Integrity & Parsing Pipeline.
#
# Goals:
#   1. Parse BOP datasets once and build Detectron2-style dataset dictionaries.
#   2. Perform comprehensive integrity validation before training begins.
#   3. Cache parsed records in RAM / disk to avoid repeated JSON parsing.
#   4. Expose a unified DatasetLoader so Detectron2 never reads raw BOP files.
#
# Validation checks:
#   [ ] Verify RGB and depth images exist and are readable by OpenCV.
#   [ ] Validate camera intrinsics (cam_K, depth_scale, finite values).
#   [ ] Validate poses (R orthonormal, det(R)=1, finite translations).
#   [ ] Validate bounding boxes:
#         - positive width/height
#         - inside image bounds
#         - finite coordinates
#         - bbox_visib preferred for detection
#   [ ] Validate masks:
#         - existence
#         - non-empty
#         - consistent with bbox_visib
#   [ ] Cross-check consistency between:
#         scene_gt.json
#         scene_gt_info.json
#         scene_camera.json
#         RGB / Depth / Mask / Mask_Visib
#   [ ] Detect duplicate images / annotations via hashing.
#   [ ] Compute dataset statistics:
#         class counts
#         bbox distributions
#         visibility statistics
#         pose distributions
#         image resolutions
#   [ ] Generate a detailed integrity report (JSON + HTML).
#   [ ] Save visual debugging outputs for failed validations.
#
# Architecture:
#
# DatasetLoader
#      │
#      ├── Cache Manager
#      ├── BOP Parser
#      ├── Integrity Validator
#      ├── Statistics Generator
#      └── Detectron2 Dataset Records
#
# Long-term objective:
# Make dataset validation a mandatory preprocessing stage before training or
# inference, replacing the current direct JSON parsing throughout GDRNPP.
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging 
from tqdm import tqdm


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class SceneSummary:
    scene_id: str
    num_images: int
    num_gt_entries: int
    num_gt_info_entries: int
    num_camera_entries: int
    missing_rgb: int = 0
    missing_depth: int = 0
    missing_gt: int = 0
    missing_gt_info: int = 0
    missing_camera: int = 0
    malformed_bbox: int = 0
    zero_area_bbox: int = 0
    missing_mask_files: int = 0
    missing_mask_visib_files: int = 0


@dataclass
class IntegrityReport:
    root: str
    bbox_source: str
    num_scenes: int
    num_images: int
    num_annotations: int
    num_records: int
    scenes: List[SceneSummary]
    counters: Dict[str, int]
    notes: List[str]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def _discover_scenes(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # If the root itself contains scene files, treat root as a single scene.
    if (root / "scene_gt.json").exists() and (root / "scene_gt_info.json").exists():
        return [root]
    
    
    scene_dirs = [] # 000001 or 000345 
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            scene_dirs.append(p)

    

    if not scene_dirs:
        raise RuntimeError(
            f"No scene folders found under {root}. Expected numeric folders like 000000/000001/." 
        )
    return scene_dirs


def _validate_bbox(bbox: Sequence[float]) -> Tuple[bool, bool]:
    """Return (is_valid, zero_area)."""
    if bbox is None or len(bbox) != 4:
        return False, False
    try:
        x, y, w, h = [float(v) for v in bbox]
    except Exception:
        return False, False
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return False, False
    if w <= 0 or h <= 0:
        return True, True
    return True, False


def _get_bbox_from_info(ann_info: Dict[str, Any], bbox_source: str) -> Optional[List[float]]:
    """Pick bbox_visib or bbox_obj from a scene_gt_info annotation.

    bbox_source:
        - 'visib' => prefer bbox_visib, fallback bbox_obj
        - 'obj'   => prefer bbox_obj, fallback bbox_visib
    """
    if bbox_source == "visib":
        return ann_info.get("bbox_visib") or ann_info.get("bbox_obj")
    return ann_info.get("bbox_obj") or ann_info.get("bbox_visib")


def _scan_scene(scene_dir: str, bbox_source: str) -> Tuple[List[Dict[str, Any]], SceneSummary, Dict[str, int]]:
    scene_path = Path(scene_dir)
    scene_id = scene_path.name

    gt_path = scene_path / "scene_gt.json"
    gt_info_path = scene_path / "scene_gt_info.json"
    cam_path = scene_path / "scene_camera.json"

    gt = _read_json(gt_path) if gt_path.exists() else {}
    gt_info = _read_json(gt_info_path) if gt_info_path.exists() else {}
    cam = _read_json(cam_path) if cam_path.exists() else {}

    rgb_dir = scene_path / "rgb"
    depth_dir = scene_path / "depth"
    mask_dir = scene_path / "mask"
    mask_visib_dir = scene_path / "mask_visib"

    image_ids = sorted(set(gt.keys()) | set(gt_info.keys()) | set(cam.keys()))
    records: List[Dict[str, Any]] = []

    counters = Counter()
    summary = SceneSummary(
        scene_id=scene_id,
        num_images=len(image_ids),
        num_gt_entries=len(gt),
        num_gt_info_entries=len(gt_info),
        num_camera_entries=len(cam),
    )

    for im_id in image_ids:
        im_id_str = str(im_id)
        int_im_id = int(im_id)
        rgb_path = rgb_dir / f"{int_im_id:06d}.png"
        depth_path = depth_dir / f"{int_im_id:06d}.png"

        if not rgb_path.exists():
            summary.missing_rgb += 1
            counters["missing_rgb"] += 1
        if not depth_path.exists():
            summary.missing_depth += 1
            counters["missing_depth"] += 1
        if im_id_str not in gt:
            summary.missing_gt += 1
            counters["missing_gt"] += 1
        if im_id_str not in gt_info:
            summary.missing_gt_info += 1
            counters["missing_gt_info"] += 1
        if im_id_str not in cam:
            summary.missing_camera += 1
            counters["missing_camera"] += 1

        # Build Detectron2-style record.
        # We keep it in RAM so the next stage can use it directly.
        record: Dict[str, Any] = {
            "dataset_name": f"{scene_id}",
            "scene_id": scene_id,
            "scene_im_id": f"{scene_id}/{int_im_id:06d}",
            "image_id": int_im_id,
            "file_name": str(rgb_path),
            "depth_file": str(depth_path),
            "height": None,
            "width": None,
            "annotations": [],
            "scene_gt_file": str(gt_path),
            "scene_gt_info_file": str(gt_info_path),
            "scene_camera_file": str(cam_path),
        }

        # Read camera intrinsics if present.
        cam_entry = cam.get(im_id_str) or cam.get(int_im_id)
        if isinstance(cam_entry, dict):
            # BOP scene_camera entries often have cam_K and depth_scale.
            if "cam_K" in cam_entry and isinstance(cam_entry["cam_K"], list) and len(cam_entry["cam_K"]) == 9:
                # height/width are not always stored, leave None if absent.
                pass
            if "depth_scale" in cam_entry:
                record["depth_scale"] = cam_entry["depth_scale"]
            if "cam_K" in cam_entry:
                record["cam_K"] = cam_entry["cam_K"]

        gt_list = gt.get(im_id_str, gt.get(int_im_id, []))
        info_list = gt_info.get(im_id_str, gt_info.get(int_im_id, []))
        if not isinstance(gt_list, list):
            gt_list = []
        if not isinstance(info_list, list):
            info_list = []

        if len(gt_list) != len(info_list):
            counters["gt_gt_info_count_mismatch"] += 1

        for anno_idx, (anno, ann_info) in enumerate(zip(gt_list, info_list)):
            obj_id = anno.get("obj_id")
            bbox = _get_bbox_from_info(ann_info, bbox_source)
            valid_bbox, zero_area = _validate_bbox(bbox)
            if not valid_bbox:
                summary.malformed_bbox += 1
                counters["malformed_bbox"] += 1
                continue
            if zero_area:
                summary.zero_area_bbox += 1
                counters["zero_area_bbox"] += 1
                continue

            # Optional mask checks. We validate only existence pattern, not pixel content.
            mask_name = f"{int_im_id:06d}_{anno_idx:06d}.png"
            mask_path = mask_dir / mask_name
            mask_visib_path = mask_visib_dir / mask_name
            if mask_dir.exists() and not mask_path.exists():
                summary.missing_mask_files += 1
                counters["missing_mask_files"] += 1
            if mask_visib_dir.exists() and not mask_visib_path.exists():
                summary.missing_mask_visib_files += 1
                counters["missing_mask_visib_files"] += 1

            record["annotations"].append(
                {
                    "category_id": obj_id,
                    "bbox": bbox,
                    "bbox_mode": "XYWH_ABS",
                    "bbox_source": "bbox_visib" if (bbox_source == "visib" and ann_info.get("bbox_visib") is not None) else "bbox_obj",
                    "obj_id": obj_id,
                    "annotation_index": anno_idx,
                    "scene_gt_index": anno_idx,
                    "mask_path": str(mask_path) if mask_dir.exists() else None,
                    "mask_visib_path": str(mask_visib_path) if mask_visib_dir.exists() else None,
                }
            )
            counters["annotations"] += 1

        if record["annotations"]:
            records.append(record)
            counters["records"] += 1

    return records, summary, dict(counters)


def build_integrity_report(root: Path, bbox_source: str, num_workers: int = 1) -> Tuple[List[Dict[str, Any]], IntegrityReport]:
    scene_dirs = _discover_scenes(root)
    records: List[Dict[str, Any]] = []
    scene_summaries: List[SceneSummary] = []
    aggregate = Counter()

    if num_workers <= 1:
        iterator = scene_dirs
        for scene_dir in tqdm(iterator, desc="Scanning scenes", unit="scene"):
            scene_records, summary, counters = _scan_scene(str(scene_dir), bbox_source)
            records.extend(scene_records)
            scene_summaries.append(summary)
            aggregate.update(counters)
            aggregate["images"] += summary.num_images
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_scan_scene, str(scene_dir), bbox_source): scene_dir for scene_dir in scene_dirs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning scenes", unit="scene"):
                scene_records, summary, counters = fut.result()
                records.extend(scene_records)
                scene_summaries.append(summary)
                aggregate.update(counters)
                aggregate["images"] += summary.num_images

    total_annotations = int(aggregate.get("annotations", 0))
    report = IntegrityReport(
        root=str(root),
        bbox_source=bbox_source,
        num_scenes=len(scene_dirs),
        num_images=int(aggregate.get("images", 0)),
        num_annotations=total_annotations,
        num_records=int(aggregate.get("records", 0)),
        scenes=scene_summaries,
        counters=dict(aggregate),
        notes=[
            "bbox_source='visib' uses bbox_visib from scene_gt_info.json and falls back to bbox_obj.",
            "A zero-area bbox (w<=0 or h<=0) is flagged and skipped.",
            "The returned records list is Detectron2-style and can be cached in RAM for the next stage.",
        ],
    )
    return records, report


def print_report(report: IntegrityReport) -> None:
    print("=" * 80)
    print("DATASET INTEGRITY REPORT")
    print("=" * 80)
    print(f"Root        : {report.root}")
    print(f"BBox source : {report.bbox_source}")
    print(f"Scenes      : {report.num_scenes}")
    print(f"Images      : {report.num_images}")
    print(f"Records     : {report.num_records}")
    print(f"Annotations : {report.num_annotations}")
    print("-")
    for k, v in sorted(report.counters.items()):
        print(f"{k:28s}: {v}")
    print("-")
    print("Per-scene summary (first 20 scenes):")
    for s in sorted(report.scenes, key=lambda x: x.scene_id)[:20]:
        print(
            f"  scene {s.scene_id} | images={s.num_images} | gt={s.num_gt_entries} | gt_info={s.num_gt_info_entries} | cam={s.num_camera_entries} | "
            f"missing_rgb={s.missing_rgb} | missing_depth={s.missing_depth} | malformed_bbox={s.malformed_bbox} | zero_area_bbox={s.zero_area_bbox}"
        )
    print("-")
    for note in report.notes:
        print(f"NOTE: {note}")
    print("=" * 80)


def save_cache(records: List[Dict[str, Any]], cache_out: Path) -> None:
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    with cache_out.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BOP dataset integrity checker and parser")
    parser.add_argument("--root", required=True, help="Path to train_pbr or the dataset root")
    parser.add_argument(
        "--bbox-source",
        choices=("visib", "obj"),
        default="visib",
        help="Which bbox to use from scene_gt_info.json (default: visib)",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for scene parsing")
    parser.add_argument("--cache-out", default=None, help="Optional path to save parsed records as pickle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    records, report = build_integrity_report(root, bbox_source=args.bbox_source, num_workers=max(1, args.num_workers))
    print_report(report)

    if args.cache_out:
        cache_out = Path(args.cache_out).expanduser().resolve()
        save_cache(records, cache_out)
        print(f"Saved RAM cache to: {cache_out}")


if __name__ == "__main__":
    main()
