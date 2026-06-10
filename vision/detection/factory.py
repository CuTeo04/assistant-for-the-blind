from __future__ import annotations

import importlib.util
import logging
import os

from vision.config import vision_config as cfg

from .backends import OnnxRuntimeDetectorBackend, UltralyticsDetectorBackend, UltralyticsOpenVocabBackend
from .service import VisionDetectorService

logger = logging.getLogger("voice_server.vision")


def _normalize_backend_name(name: str | None, default: str = "ultralytics") -> str:
    if not name:
        return default
    return str(name).strip().lower()


def create_primary_backend(model_path: str | None = None, backend: str | None = None):
    backend_name = _normalize_backend_name(backend or getattr(cfg, "YOLO_BACKEND", "ultralytics"))
    if model_path is None:
        if backend_name == "onnx":
            model_path = cfg.YOLO_ONNX_PATH
        else:
            model_path = cfg.YOLO_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at: {model_path}")

    if backend_name == "onnx":
        return OnnxRuntimeDetectorBackend(
            model_path=model_path,
            source_name="yolo",
            backend_name=backend_name,
            execution_provider=getattr(cfg, "YOLO_EXECUTION_PROVIDER", "cpu"),
            intra_op_num_threads=getattr(cfg, "YOLO_ONNX_INTRA_OP_THREADS", None),
            inter_op_num_threads=getattr(cfg, "YOLO_ONNX_INTER_OP_THREADS", None),
        )

    return UltralyticsDetectorBackend(
        model_path=model_path,
        source_name="yolo",
        backend_name=backend_name,
    )


def create_primary_full_backend():
    return OnnxRuntimeDetectorBackend(
        model_path=cfg.YOLO_FULL_MODEL_PATH,
        source_name="yolo_full",
        backend_name=_normalize_backend_name(getattr(cfg, "YOLO_BACKEND", "onnx")),
        execution_provider=cfg.YOLO_FULL_EXECUTION_PROVIDER,
        intra_op_num_threads=getattr(cfg, "YOLO_FULL_ONNX_INTRA_OP_THREADS", None),
        inter_op_num_threads=getattr(cfg, "YOLO_FULL_ONNX_INTER_OP_THREADS", None),
    )


def create_primary_tile_backend():
    return OnnxRuntimeDetectorBackend(
        model_path=cfg.YOLO_TILE_MODEL_PATH,
        source_name="yolo_tile",
        backend_name=_normalize_backend_name(getattr(cfg, "YOLO_BACKEND", "onnx")),
        execution_provider=cfg.YOLO_TILE_EXECUTION_PROVIDER,
        intra_op_num_threads=getattr(cfg, "YOLO_TILE_ONNX_INTRA_OP_THREADS", None),
        inter_op_num_threads=getattr(cfg, "YOLO_TILE_ONNX_INTER_OP_THREADS", None),
    )


def create_open_vocab_backend(model_path: str | None = None, backend: str | None = None):
    backend_name = _normalize_backend_name(backend or getattr(cfg, "YOLO_WORLD_BACKEND", "ultralytics"))
    model_path = model_path or cfg.YOLO_WORLD_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO-World model not found at: {model_path}")

    if backend_name == "onnx":
        backend = OnnxRuntimeDetectorBackend(
            model_path=model_path,
            source_name="yolo_world",
            intra_op_num_threads=getattr(cfg, "YOLO_WORLD_ONNX_INTRA_OP_THREADS", None),
            inter_op_num_threads=getattr(cfg, "YOLO_WORLD_ONNX_INTER_OP_THREADS", None),
            backend_name=backend_name,
            execution_provider=getattr(cfg, "YOLO_WORLD_EXECUTION_PROVIDER", "cpu"),
        )
        expected_labels = {label for label in cfg.OPEN_VOCAB_EXTRA_CLASSES if label}
        available_labels = set(backend.label_lookup().values())
        matched_labels = sorted(expected_labels & available_labels)
        missing_labels = sorted(expected_labels - available_labels)
        if expected_labels and not matched_labels:
            raise ValueError(
                "YOLO-World ONNX model does not contain any labels from "
                "vision.open_vocab_extra_classes. Re-export it with "
                "scripts/export_yolo_world_onnx.py so the offline vocabulary is embedded."
            )
        if missing_labels:
            logger.warning(
                "YOLO-World ONNX vocabulary partially matched config labels: matched=%d missing=%d sample_missing=%s",
                len(matched_labels),
                len(missing_labels),
                missing_labels[:10],
            )
        else:
            logger.info(
                "YOLO-World ONNX vocabulary matched all configured open-vocab labels: count=%d",
                len(matched_labels),
            )
        return backend

    if importlib.util.find_spec("clip") is None:
        raise ModuleNotFoundError(
            "Missing dependency 'clip' required by YOLO-World. "
            "Install requirements.txt in the active environment before enabling open_vocab."
        )

    return UltralyticsOpenVocabBackend(
        model_path=model_path,
        source_name="yolo_world",
        backend_name=backend_name,
    )


def create_open_vocab_full_backend():
    return OnnxRuntimeDetectorBackend(
        model_path=cfg.YOLO_WORLD_FULL_MODEL_PATH,
        source_name="yolo_world_full",
        intra_op_num_threads=getattr(cfg, "YOLO_WORLD_FULL_ONNX_INTRA_OP_THREADS", None),
        inter_op_num_threads=getattr(cfg, "YOLO_WORLD_FULL_ONNX_INTER_OP_THREADS", None),
        backend_name=_normalize_backend_name(getattr(cfg, "YOLO_WORLD_BACKEND", "onnx")),
        execution_provider=cfg.YOLO_WORLD_FULL_EXECUTION_PROVIDER,
    )


def create_open_vocab_tile_backend():
    return OnnxRuntimeDetectorBackend(
        intra_op_num_threads=getattr(cfg, "YOLO_WORLD_TILE_ONNX_INTRA_OP_THREADS", None),
        inter_op_num_threads=getattr(cfg, "YOLO_WORLD_TILE_ONNX_INTER_OP_THREADS", None),
        model_path=cfg.YOLO_WORLD_TILE_MODEL_PATH,
        source_name="yolo_world_tile",
        backend_name=_normalize_backend_name(getattr(cfg, "YOLO_WORLD_BACKEND", "onnx")),
        execution_provider=cfg.YOLO_WORLD_TILE_EXECUTION_PROVIDER,
    )


def create_detector_service(primary_backend=None, open_vocab_backend=None) -> VisionDetectorService:
    if cfg.YOLO_SPLIT_WORKERS_ENABLED:
        primary_backend = None
        primary_full_backend = create_primary_full_backend()
        primary_tile_backend = create_primary_tile_backend()
    else:
        primary_backend = primary_backend or create_primary_backend()
        primary_full_backend = None
        primary_tile_backend = None

    if cfg.OPEN_VOCAB_ENABLED and cfg.OPEN_VOCAB_SPLIT_WORKERS_ENABLED:
        open_vocab_backend = None
        open_vocab_full_backend = create_open_vocab_full_backend()
        open_vocab_tile_backend = create_open_vocab_tile_backend()
    elif cfg.OPEN_VOCAB_ENABLED:
        open_vocab_backend = open_vocab_backend or create_open_vocab_backend()
        open_vocab_full_backend = None
        open_vocab_tile_backend = None
    else:
        open_vocab_full_backend = None
        open_vocab_tile_backend = None

    return VisionDetectorService(
        primary_backend,
        open_vocab_backend,
        primary_full_backend=primary_full_backend,
        primary_tile_backend=primary_tile_backend,
        open_vocab_full_backend=open_vocab_full_backend,
        open_vocab_tile_backend=open_vocab_tile_backend,
    )
