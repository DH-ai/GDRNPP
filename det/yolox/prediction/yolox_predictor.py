"""High-level, parameter-driven YOLOX prediction API.

The public output format is deliberately independent of torch.  Every prediction
is a dictionary containing image metadata and a list of detections whose boxes
are in original-image ``[x1, y1, x2, y2]`` pixel coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from detectron2.config import LazyConfig, instantiate

from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.utils import fuse_model

from .utils.io import IMAGE_EXTENSIONS, image_files, read_image, write_image, write_json
from .utils.postprocess import decode_predictions
from .utils.preprocess import preprocess_images
from .visualize import draw_detections

ImageInput = Union[str, Path, np.ndarray]


class YOLOXPredictor:
    """Run a GDRNPP YOLOX model on images, batches, or folders.

    Args:
        config_file: Detectron2 LazyConfig used to construct the model.
        checkpoint: Model checkpoint. Required when ``model`` is not supplied.
        model: Optional already-constructed model, useful for embedding/testing.
        device: Torch device. Defaults to CUDA when available, otherwise CPU.
        conf_thres: Confidence threshold override.
        nms_thres: NMS IoU threshold override.
        test_size: ``(height, width)`` input size override.
        num_classes: Number of classes override. Required for a model without a
            config.
        class_names: Optional class names indexed by class id.
        fp16: Use half precision. CUDA is required.
        fuse: Fuse convolution and batch-normalization layers after loading.
        class_agnostic: Use class-agnostic NMS.
        augment: Use the config's YOLOX multi-scale test-time augmentation.
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        *,
        model: Optional[torch.nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        conf_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
        test_size: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        fp16: bool = False,
        fuse: bool = False,
        class_agnostic: bool = False,
        augment: Optional[bool] = None,
    ) -> None:
        if config_file is None and model is None:
            raise ValueError("config_file or model must be provided")

        self.config_file = str(config_file) if config_file is not None else None
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.cfg = LazyConfig.load(self.config_file) if self.config_file else None

        cfg_test = self.cfg.test if self.cfg is not None else None
        configured_size = self._cfg_value(cfg_test, "test_size", (640, 640))
        self.test_size = self._size(test_size if test_size is not None else configured_size)
        self.num_classes = int(
            num_classes if num_classes is not None else self._cfg_value(cfg_test, "num_classes", 0)
        )
        if self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")

        self.conf_thres = float(
            conf_thres if conf_thres is not None else self._cfg_value(cfg_test, "conf_thr", 0.25)
        )
        self.nms_thres = float(
            nms_thres if nms_thres is not None else self._cfg_value(cfg_test, "nms_thr", 0.45)
        )
        self.class_names = list(class_names) if class_names is not None else None
        self.class_agnostic = bool(class_agnostic)
        self.augment = bool(
            augment if augment is not None else self._cfg_value(cfg_test, "augment", False)
        )

        requested_device = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(requested_device)
        self.fp16 = bool(fp16)
        if self.fp16 and self.device.type != "cuda":
            raise ValueError("fp16 inference requires a CUDA device")

        self.model: torch.nn.Module
        self.load_model(model=model, checkpoint=self.checkpoint, fuse=fuse)

    @staticmethod
    def _cfg_value(section: Any, name: str, default: Any) -> Any:
        if section is None:
            return default
        if isinstance(section, dict):
            return section.get(name, default)
        return getattr(section, name, default)

    @staticmethod
    def _size(value: Sequence[int]) -> Tuple[int, int]:
        if len(value) != 2 or any(int(v) <= 0 for v in value):
            raise ValueError(f"test_size must be (height, width), got {value!r}")
        return int(value[0]), int(value[1])

    def load_model(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        fuse: bool = False,
    ) -> torch.nn.Module:
        """Construct/load the model and put it in inference mode."""
        supplied_model = model is not None
        if model is None and checkpoint is None:
            checkpoint = self.checkpoint
        if model is None:
            if self.cfg is None:
                raise ValueError("A config is required to construct the model")
            model = instantiate(self.cfg.model)
        if checkpoint is not None:
            checkpoint = str(checkpoint)
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            MyCheckpointer(model).load(checkpoint)
            self.checkpoint = checkpoint
        elif not supplied_model and self.checkpoint is None and self.cfg is not None:
            raise ValueError("checkpoint is required when constructing a model from config_file")

        model = model.to(self.device).eval()
        if fuse:
            model = fuse_model(model)
        if self.fp16:
            model = model.half()
        self.model = model
        return model

    def preprocess(self, images: Union[ImageInput, Sequence[ImageInput]]):
        """Read and letterbox one or more images into a model-ready tensor."""
        sources = [images] if isinstance(images, (str, Path, np.ndarray)) else list(images)
        return preprocess_images(sources, self.test_size, fp16=self.fp16)

    @torch.inference_mode()
    def infer(self, tensor: torch.Tensor) -> Any:
        """Run the model and return its raw detection tensor."""
        tensor = tensor.to(self.device, non_blocking=self.device.type == "cuda")
        kwargs: Dict[str, Any] = {}
        if self.augment:
            if self.cfg is None:
                raise ValueError("augment=True requires a config with test-time scales")
            kwargs = {"augment": True, "cfg": self.cfg.test}
        outputs = self.model(tensor, **kwargs)
        if isinstance(outputs, dict):
            if "det_preds" not in outputs:
                raise KeyError("Model output dictionary does not contain 'det_preds'")
            return outputs["det_preds"]
        return outputs

    def postprocess(self, raw_predictions: Any, metadata: Sequence[Dict[str, Any]]):
        """Apply confidence filtering/NMS and restore original image coordinates."""
        return decode_predictions(
            raw_predictions,
            metadata,
            num_classes=self.num_classes,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres,
            class_agnostic=self.class_agnostic,
            class_names=self.class_names,
        )

    def predict(
        self,
        source: Union[ImageInput, Sequence[ImageInput]],
        *,
        batch_size: int = 1,
        recursive: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        draw: bool = False,
        save_images: bool = False,
        save_json: bool = False,
        **draw_kwargs: Any,
    ):
        """Dispatch prediction based on whether ``source`` is an image/list/folder."""
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            return self.predict_folder(
                source,
                batch_size=batch_size,
                recursive=recursive,
                output_dir=output_dir,
                draw=draw,
                save_images=save_images,
                save_json=save_json,
                **draw_kwargs,
            )
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, Path, np.ndarray)):
            return self.predict_batch(
                source,
                batch_size=batch_size,
                output_dir=output_dir,
                draw=draw,
                save_images=save_images,
                save_json_files=save_json,
                **draw_kwargs,
            )
        return self.predict_image(
            source, output_dir=output_dir, draw=draw, save_image=save_images, save_json=save_json, **draw_kwargs
        )

    def predict_image(
        self,
        image: ImageInput,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        draw: bool = False,
        save_image: bool = False,
        save_json: bool = False,
        **draw_kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict one image and optionally draw/save its results."""
        return self.predict_batch(
            [image],
            batch_size=1,
            output_dir=output_dir,
            draw=draw,
            save_images=save_image,
            save_json_files=save_json,
            **draw_kwargs,
        )[0]

    def predict_batch(
        self,
        images: Sequence[ImageInput],
        *,
        batch_size: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        draw: bool = False,
        save_images: bool = False,
        save_json_files: bool = False,
        **draw_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Predict images in configurable mini-batches."""
        images = list(images)
        if not images:
            return []
        batch_size = len(images) if batch_size is None else int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (save_images or save_json_files) and output_dir is None:
            raise ValueError("output_dir is required when saving results")

        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            tensor, metadata = self.preprocess(chunk)
            t0 = time.perf_counter()
            raw = self.infer(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            chunk_results = self.postprocess(raw, metadata)
            per_image_ms = elapsed_ms / len(chunk)
            for source, result in zip(chunk, chunk_results):
                result["inference_ms"] = per_image_ms
                if draw or save_images:
                    rendered = self.draw(source, result["detections"], **draw_kwargs)
                    if draw:
                        result["rendered_image"] = rendered
                    if save_images:
                        result["output_image"] = str(self.save_image(rendered, output_dir, result["file_name"]))
                if save_json_files:
                    result["output_json"] = str(
                        self.save_json(result, Path(output_dir) / f"{Path(result['file_name']).stem}_pred.json")
                    )
            results.extend(chunk_results)
        return results

    def predict_folder(
        self,
        folder: Union[str, Path],
        *,
        batch_size: int = 1,
        recursive: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        draw: bool = False,
        save_images: bool = False,
        save_json: bool = False,
        extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS),
        **draw_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Predict all supported images in a folder, in stable path order."""
        if (save_images or save_json) and output_dir is None:
            raise ValueError("output_dir is required when saving results")
        folder = Path(folder)
        files = image_files(folder, recursive=recursive, extensions=extensions)
        results = self.predict_batch(
            files,
            batch_size=batch_size,
            draw=draw,
            **draw_kwargs,
        )
        if save_images:
            for source, result in zip(files, results):
                rendered = result.get("rendered_image")
                if rendered is None:
                    rendered = self.draw(source, result["detections"], **draw_kwargs)
                relative = source.relative_to(folder)
                destination = Path(output_dir) / relative.parent / f"{relative.stem}_pred{relative.suffix}"
                result["output_image"] = str(self.save_image(rendered, destination))
        if save_json:
            self.save_json(results, Path(output_dir) / "predictions.json")
        return results

    def draw(self, image: ImageInput, detections: Sequence[Dict[str, Any]], **kwargs: Any) -> np.ndarray:
        """Draw detections on a copy of an image."""
        image_array, _ = read_image(image)
        return draw_detections(image_array, detections, class_names=self.class_names, **kwargs)

    def save_json(self, predictions: Any, path: Union[str, Path]) -> Path:
        """Serialize predictions, excluding non-JSON rendered images."""
        return write_json(predictions, path)

    def save_image(
        self, image: np.ndarray, output: Union[str, Path], source_name: Optional[str] = None
    ) -> Path:
        """Save an image to a file, or into a directory using ``source_name``."""
        output = Path(output)
        if source_name is not None:
            output = output / f"{Path(source_name).stem}_pred{Path(source_name).suffix or '.jpg'}"
        return write_image(image, output)

    def benchmark(
        self,
        image: Optional[ImageInput] = None,
        *,
        batch_size: int = 1,
        warmup: int = 10,
        iterations: int = 100,
        include_preprocess: bool = False,
    ) -> Dict[str, float]:
        """Benchmark inference and return latency/throughput statistics."""
        if batch_size <= 0 or warmup < 0 or iterations <= 0:
            raise ValueError("batch_size/iterations must be positive and warmup must be non-negative")
        if image is None:
            dtype = torch.float16 if self.fp16 else torch.float32
            tensor = torch.zeros((batch_size, 3, *self.test_size), dtype=dtype)
            inputs = None
        else:
            inputs = [image] * batch_size
            tensor, _ = self.preprocess(inputs)
        tensor = tensor.to(self.device)

        for _ in range(warmup):
            self.infer(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        samples: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            current = self.preprocess(inputs)[0] if include_preprocess and inputs is not None else tensor
            self.infer(current)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            samples.append((time.perf_counter() - t0) * 1000.0)

        values = np.asarray(samples, dtype=np.float64)
        mean_ms = float(values.mean())
        return {
            "batch_size": float(batch_size),
            "iterations": float(iterations),
            "mean_ms": mean_ms,
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "fps": float(batch_size * 1000.0 / mean_ms),
        }


class YOLOXDetector(YOLOXPredictor):
    """Backward-compatible name and positional signature for the old wrapper."""

    def __init__(
        self,
        config_file: Union[str, Path],
        checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        conf_thres: Optional[float] = None,
        nms_thres: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config_file,
            checkpoint,
            device=device,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            **kwargs,
        )


__all__ = ["YOLOXPredictor", "YOLOXDetector"]
