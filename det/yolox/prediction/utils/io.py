"""Image and JSON I/O for YOLOX prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def read_image(source: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, str]:
    if isinstance(source, np.ndarray):
        image = source
        name = "image.jpg"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {path}")
        name = path.name

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxW, HxWx3, or HxWx4 image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image data, got {image.dtype}")
    return np.ascontiguousarray(image), name


def image_files(
    folder: Union[str, Path],
    *,
    recursive: bool = False,
    extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS),
) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder}")
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in allowed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "rendered_image"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(value: Any, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def write_image(image: np.ndarray, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")
    return path
