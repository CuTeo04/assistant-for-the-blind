from __future__ import annotations

import ast
import logging
import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
import onnxruntime as ort
import torch

from log_settings import is_enabled

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YOLO_CONFIG_DIR = os.path.join(ROOT_DIR, ".cache", "ultralytics")
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR

from ultralytics import YOLO, YOLOWorld
try:
    from ultralytics.utils.nms import non_max_suppression
except ModuleNotFoundError:
    from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.ops import scale_boxes

logger = logging.getLogger("voice_server.vision")


_AUTO_PROVIDER_PRIORITY = (
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _letterbox_image(img: np.ndarray, new_shape: tuple[int, int], color=(114, 114, 114)) -> np.ndarray:
    shape = img.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])
    ratio = min(new_h / shape[0], new_w / shape[1])
    resized_w, resized_h = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))

    if (shape[1], shape[0]) != (resized_w, resized_h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


class DetectorBackend(ABC):
    def __init__(self, source_name: str, backend_name: str):
        self.source_name = source_name
        self.backend_name = backend_name

    @abstractmethod
    def label_lookup(self) -> dict[int, str]:
        raise NotImplementedError

    @abstractmethod
    def predict_boxes(
        self,
        img,
        conf_threshold: float,
        imgsz=None,
        classes=None,
        device=None,
    ) -> np.ndarray:
        raise NotImplementedError

    def predict_boxes_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float,
        imgsz=None,
        classes=None,
        device=None,
    ) -> list[np.ndarray]:
        return [
            self.predict_boxes(
                img,
                conf_threshold,
                imgsz=imgsz,
                classes=classes,
                device=device,
            )
            for img in images
        ]

    def class_id_for_label(self, target_label: str) -> int | None:
        target_label = _normalize_label(target_label)
        for class_id, label in self.label_lookup().items():
            if label == target_label:
                return int(class_id)
        return None

    def supports_prompt_labels(self) -> bool:
        return False

    def set_prompt_labels(self, labels: list[str]):
        raise NotImplementedError(f"{self.__class__.__name__} does not support prompt labels")

    def reset(self):
        return None

    def tile_batch_size(self) -> int:
        return 1


class GenericModelBackend(DetectorBackend):
    def __init__(self, model, source_name: str, backend_name: str = "external"):
        super().__init__(source_name=source_name, backend_name=backend_name)
        self.model = model

    def label_lookup(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if hasattr(names, "items"):
            return {int(class_id): _normalize_label(label) for class_id, label in names.items()}
        return {int(class_id): _normalize_label(label) for class_id, label in enumerate(names)}

    def predict_boxes(
        self,
        img,
        conf_threshold: float,
        imgsz=None,
        classes=None,
        device=None,
    ) -> np.ndarray:
        kwargs = {
            "conf": conf_threshold,
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        if imgsz:
            kwargs["imgsz"] = imgsz
        if classes is not None:
            kwargs["classes"] = classes

        results = self.model.predict(img, **kwargs)[0]
        return results.boxes.data.cpu().numpy()

    def set_prompt_labels(self, labels: list[str]):
        if not self.supports_prompt_labels():
            super().set_prompt_labels(labels)
        self.model.set_classes(labels)

    def supports_prompt_labels(self) -> bool:
        return hasattr(self.model, "set_classes")

    def reset(self):
        if hasattr(self.model, "predictor"):
            self.model.predictor = None


class UltralyticsDetectorBackend(GenericModelBackend):
    def __init__(self, model_path: str, source_name: str, backend_name: str):
        super().__init__(
            model=YOLO(model_path),
            source_name=source_name,
            backend_name=backend_name,
        )
        self.model_path = model_path


class UltralyticsOpenVocabBackend(GenericModelBackend):
    def __init__(self, model_path: str, source_name: str, backend_name: str):
        super().__init__(
            model=YOLOWorld(model_path),
            source_name=source_name,
            backend_name=backend_name,
        )
        self.model_path = model_path


class OnnxRuntimeDetectorBackend(DetectorBackend):
    def __init__(
        self,
        model_path: str,
        source_name: str,
        backend_name: str,
        execution_provider: str = "cpu",
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ):
        super().__init__(source_name=source_name, backend_name=backend_name)
        self.model_path = model_path
        self.execution_provider = execution_provider
        self.providers = self._resolve_providers(execution_provider)
        self.preferred_providers = list(self.providers)
        self.session_options = self._build_session_options(
            intra_op_num_threads,
            inter_op_num_threads,
        )
        self.session = self._create_session_with_retry(reason="session_init")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.names = self._load_names()
        self.input_hw = self._load_input_hw()
        self.input_batch = self._load_input_batch()
        logger.info(
            "%s ONNX Runtime session ready | providers=%s | input_hw=%s | input_batch=%s",
            self.source_name,
            self.session.get_providers(),
            self.input_hw,
            self.input_batch,
        )

    def _uses_accelerator(self) -> bool:
        return any(provider != "CPUExecutionProvider" for provider in self.preferred_providers)

    def _create_session_with_retry(self, reason: str):
        attempt = 0
        while True:
            attempt += 1
            try:
                logger.info(
                    "%s creating ONNX session | reason=%s | attempt=%d | providers=%s | model=%s",
                    self.source_name,
                    reason,
                    attempt,
                    self.preferred_providers,
                    self.model_path,
                )
                self.providers = list(self.preferred_providers)
                session = ort.InferenceSession(
                    self.model_path,
                    sess_options=self.session_options,
                    providers=self.providers,
                )
                logger.info(
                    "%s ONNX session created | reason=%s | attempt=%d | providers=%s",
                    self.source_name,
                    reason,
                    attempt,
                    session.get_providers(),
                )
                return session
            except Exception:
                if not self._uses_accelerator():
                    raise
                logger.warning(
                    "%s accelerator session failed for %s during %s; retrying on CPU only | attempt=%d providers=%s",
                    self.source_name,
                    self.model_path,
                    reason,
                    attempt,
                    self.preferred_providers,
                    exc_info=True,
                )
                self.preferred_providers = ["CPUExecutionProvider"]
                time.sleep(0.1)

    def _reload_session_with_retry(self, reason: str) -> None:
        self.session = self._create_session_with_retry(reason=reason)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.names = self._load_names()
        self.input_hw = self._load_input_hw()
        self.input_batch = self._load_input_batch()

    def _build_session_options(
        self,
        intra_op_num_threads: int | None,
        inter_op_num_threads: int | None,
    ) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None:
            options.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads is not None:
            options.inter_op_num_threads = int(inter_op_num_threads)
        return options

    def _resolve_providers(self, execution_provider: str) -> list[str]:
        provider_name = str(execution_provider).strip().lower()
        available = set(ort.get_available_providers())
        if provider_name in {"auto", "hardware", "accelerator", "gpu"}:
            providers = [
                provider
                for provider in _AUTO_PROVIDER_PRIORITY
                if provider in available
            ]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            if providers:
                return providers
            raise RuntimeError(
                "No supported ONNX Runtime execution provider is available. "
                "Expected at least CPUExecutionProvider."
            )
        if provider_name in {"directml", "dml"}:
            if "DmlExecutionProvider" not in available:
                raise RuntimeError(
                    "DirectML provider is not available. Install onnxruntime-directml and ensure the iGPU runtime is visible."
                )
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if provider_name == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA provider is not available. Install a CUDA-enabled ONNX Runtime build and verify the GPU runtime."
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_names(self) -> dict[int, str]:
        metadata = self.session.get_modelmeta().custom_metadata_map
        raw_names = metadata.get("names", "")
        if not raw_names:
            return {}
        parsed = ast.literal_eval(raw_names)
        if hasattr(parsed, "items"):
            return {int(class_id): _normalize_label(label) for class_id, label in parsed.items()}
        return {int(class_id): _normalize_label(label) for class_id, label in enumerate(parsed)}

    def _load_input_hw(self) -> tuple[int, int] | None:
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 4 and all(isinstance(dim, int) for dim in input_shape[2:4]):
            return int(input_shape[2]), int(input_shape[3])
        return None

    def _load_input_batch(self) -> int | None:
        input_shape = self.session.get_inputs()[0].shape
        if input_shape and isinstance(input_shape[0], int):
            return int(input_shape[0])
        return None

    def label_lookup(self) -> dict[int, str]:
        return self.names

    def tile_batch_size(self) -> int:
        return max(1, int(self.input_batch or 1))

    def _target_hw(self, imgsz=None) -> tuple[int, int]:
        if self.input_hw is not None:
            return self.input_hw
        if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])
        size = int(imgsz or 640)
        return size, size

    def predict_boxes(
        self,
        img,
        conf_threshold: float,
        imgsz=None,
        classes=None,
        device=None,
    ) -> np.ndarray:
        input_h, input_w = self._target_hw(imgsz)
        return self.predict_boxes_batch(
            [img],
            conf_threshold,
            imgsz=imgsz,
            classes=classes,
            device=device,
        )[0]

    def predict_boxes_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float,
        imgsz=None,
        classes=None,
        device=None,
    ) -> list[np.ndarray]:
        if not images:
            return []

        max_batch = self.input_batch or 1
        if max_batch < len(images):
            return super().predict_boxes_batch(
                images,
                conf_threshold,
                imgsz=imgsz,
                classes=classes,
                device=device,
            )

        original_count = len(images)
        if max_batch > original_count:
            padded_images = list(images)
            pad_image = images[-1]
            while len(padded_images) < max_batch:
                padded_images.append(pad_image)
            images = padded_images

        input_h, input_w = self._target_hw(imgsz)
        if is_enabled("yolo_inference_trace", False):
            logger.info(
                "%s predict_boxes_batch start | images=%d | batch=%d | conf=%.2f | hw=%sx%s",
                self.source_name,
                len(images),
                max_batch,
                conf_threshold,
                input_h,
                input_w,
            )
        tensors = []
        for img in images:
            resized = _letterbox_image(img, (input_h, input_w))
            tensor = resized[:, :, ::-1].transpose(2, 0, 1)
            tensors.append(np.ascontiguousarray(tensor, dtype=np.float32) / 255.0)
        batch_tensor = np.stack(tensors, axis=0)

        attempt = 0
        while True:
            attempt += 1
            try:
                if is_enabled("yolo_inference_trace", False):
                    logger.info(
                        "%s session.run start | attempt=%d | input=%s | batch_shape=%s",
                        self.source_name,
                        attempt,
                        self.input_name,
                        batch_tensor.shape,
                    )
                outputs = self.session.run(self.output_names, {self.input_name: batch_tensor})
                if is_enabled("yolo_inference_trace", False):
                    logger.info(
                        "%s session.run done | attempt=%d",
                        self.source_name,
                        attempt,
                    )
                break
            except Exception:
                if not self._uses_accelerator():
                    raise
                logger.warning(
                    "%s accelerator inference failed for %s; rebuilding same providers and retrying forever | attempt=%d providers=%s",
                    self.source_name,
                    self.model_path,
                    attempt,
                    self.preferred_providers,
                    exc_info=True,
                )
                time.sleep(0.5)
                self._reload_session_with_retry(reason="inference_retry")
        predictions = torch.from_numpy(outputs[0] if isinstance(outputs, list) else outputs)
        detections_batch = non_max_suppression(
            predictions,
            conf_thres=conf_threshold,
            iou_thres=0.45,
            classes=classes,
            nc=len(self.names),
        )

        results: list[np.ndarray] = []
        for img, detections in zip(images, detections_batch):
            if detections.numel() == 0:
                results.append(np.empty((0, 6), dtype=np.float32))
                continue

            detections = detections.clone()
            scale_boxes((input_h, input_w), detections[:, :4], img.shape[:2])
            results.append(detections[:, :6].cpu().numpy().astype(np.float32))
        if is_enabled("yolo_inference_trace", False):
            logger.info("%s predict_boxes_batch done | outputs=%d", self.source_name, len(results))
        return results[:original_count]


def ensure_detector_backend(model, source_name: str) -> DetectorBackend | None:
    if model is None:
        return None
    if isinstance(model, DetectorBackend):
        return model
    return GenericModelBackend(model=model, source_name=source_name)
