"""Convert raw YOLOX outputs into JSON-friendly original-image detections."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from det.yolox.utils.boxes import postprocess as yolox_postprocess


def decode_predictions(
    raw_predictions: Any,
    metadata: Sequence[Dict[str, Any]],
    *,
    num_classes: int,
    conf_thres: float,
    nms_thres: float,
    class_agnostic: bool = False,
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if not torch.is_tensor(raw_predictions):
        raise TypeError(f"Expected raw predictions to be a torch.Tensor, got {type(raw_predictions).__name__}")
    if raw_predictions.ndim != 3 or raw_predictions.shape[0] != len(metadata):
        raise ValueError(
            f"Prediction shape {tuple(raw_predictions.shape)} does not match {len(metadata)} input image(s)"
        )

    # The repository helper mutates box coordinates, so clone to keep raw model
    # outputs safe for callers that want to inspect or reuse them.
    batches = yolox_postprocess(
        raw_predictions.clone(),
        num_classes,
        conf_thre,
        nms_thre,
        class_agnostic=class_agnostic,
    )
    results: List[Dict[str, Any]] = []
    for detections, info in zip(batches, metadata):
        formatted: List[Dict[str, Any]] = []
        if detections is not None:
            detections = detections.detach().float().cpu()
            boxes = detections[:, :4] / float(info["ratio"])
            boxes[:, 0::2].clamp_(0, float(info["width"]))
            boxes[:, 1::2].clamp_(0, float(info["height"]))
            for box, det in zip(boxes, detections):
                class_id = int(det[6].item())
                objectness = float(det[4].item())
                class_confidence = float(det[5].item())
                item: Dict[str, Any] = {
                    "bbox": [float(value) for value in box.tolist()],
                    "score": objectness * class_confidence,
                    "class_id": class_id,
                    "objectness": objectness,
                    "class_confidence": class_confidence,
                }
                if class_names is not None and 0 <= class_id < len(class_names):
                    item["class_name"] = str(class_names[class_id])
                formatted.append(item)
        result = {key: value for key, value in info.items() if key != "ratio"}
        result["detections"] = formatted
        results.append(result)
    return results
