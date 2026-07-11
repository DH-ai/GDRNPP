"""YOLOX letterbox preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from det.yolox.data.data_augment import preproc

from .io import read_image


def preprocess_images(
    sources: Sequence[Union[str, Path, np.ndarray]],
    test_size: Tuple[int, int],
    *,
    fp16: bool = False,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    if not sources:
        raise ValueError("At least one image is required")
    tensors = []
    metadata: List[Dict[str, Any]] = []
    for source in sources:
        image, file_name = read_image(source)
        height, width = image.shape[:2]
        processed, ratio = preproc(image, test_size)
        tensors.append(torch.from_numpy(processed))
        metadata.append(
            {
                "source": str(source) if not isinstance(source, np.ndarray) else None,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
                "ratio": float(ratio),
            }
        )
    tensor = torch.stack(tensors, dim=0)
    tensor = tensor.half() if fp16 else tensor.float()
    return tensor, metadata
