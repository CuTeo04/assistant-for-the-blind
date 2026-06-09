from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO, YOLOWorld
try:
    from ultralytics.utils.nms import non_max_suppression
except ModuleNotFoundError:
    from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.ops import scale_boxes

logger = logging.getLogger("voice_server.vision")


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
    ):
        super().__init__(source_name=source_name, backend_name=backend_name)
        self.model_path = model_path
        self.execution_provider = execution_provider
        self.providers = self._resolve_providers(execution_provider)
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.names = self._load_names()
        self.input_hw = self._load_input_hw()
        logger.info(
            "%s ONNX Runtime session ready | providers=%s | input_hw=%s",
            self.source_name,
            self.session.get_providers(),
            self.input_hw,
        )

    def _resolve_providers(self, execution_provider: str) -> list[str]:
        provider_name = str(execution_provider).strip().lower()
        available = set(ort.get_available_providers())
        if provider_name == "directml":
            if "DmlExecutionProvider" not in available:
                raise RuntimeError(
                    "DirectML provider is not available. Install onnxruntime-directml and ensure the iGPU runtime is visible."
                )
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
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

    def label_lookup(self) -> dict[int, str]:
        return self.names

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
        del device
        input_h, input_w = self._target_hw(imgsz)
        resized = _letterbox_image(img, (input_h, input_w))
        tensor = resized[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        predictions = torch.from_numpy(outputs[0] if isinstance(outputs, list) else outputs)
        detections = non_max_suppression(
            predictions,
            conf_thres=conf_threshold,
            iou_thres=0.45,
            classes=classes,
            nc=len(self.names),
        )[0]
        if detections.numel() == 0:
            return np.empty((0, 6), dtype=np.float32)

        detections = detections.clone()
        scale_boxes((input_h, input_w), detections[:, :4], img.shape[:2])
        return detections[:, :6].cpu().numpy().astype(np.float32)


def ensure_detector_backend(model, source_name: str) -> DetectorBackend | None:
    if model is None:
        return None
    if isinstance(model, DetectorBackend):
        return model
    return GenericModelBackend(model=model, source_name=source_name)
