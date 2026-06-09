from __future__ import annotations

from vision.detection.factory import create_open_vocab_backend, create_primary_backend
from vision.detection.pipeline import (
    apply_redetect_resolution,
    build_open_vocab_prompts,
    boxes_to_detection_records,
    class_aware_nms,
    filter_allowed_classes,
    find_redetect_candidates,
    make_union_crop_box,
    map_crop_boxes_to_pipeline_coords,
    merge_detection_records,
    redetect_config_from_runtime,
    redetect_for_nested_candidates,
    resolve_redetect_candidate,
    run_detector_pipeline,
    run_redetect_on_crop,
)
from vision.detection.service import VisionDetectorService


def load_yolo(model_path: str):
    backend = "onnx" if str(model_path).strip().lower().endswith(".onnx") else None
    return create_primary_backend(model_path=model_path, backend=backend)


def load_yolo_world(model_path: str):
    return create_open_vocab_backend(model_path=model_path)


def detect_objects(yolo, img, orig_img=None, yolo_world=None):
    detector = yolo if isinstance(yolo, VisionDetectorService) else VisionDetectorService(yolo, yolo_world)
    return detector.detect_objects(img, orig_img=orig_img)


def compute_object_size_cm(box_px, depth_m: float, focal_px: float):
    x1, y1, x2, y2 = box_px
    depth_cm = depth_m * 100.0
    real_w_cm = ((x2 - x1) / focal_px) * depth_cm
    real_h_cm = ((y2 - y1) / focal_px) * depth_cm
    return real_w_cm, real_h_cm


__all__ = [
    "apply_redetect_resolution",
    "boxes_to_detection_records",
    "build_open_vocab_prompts",
    "class_aware_nms",
    "compute_object_size_cm",
    "detect_objects",
    "filter_allowed_classes",
    "find_redetect_candidates",
    "load_yolo",
    "load_yolo_world",
    "make_union_crop_box",
    "map_crop_boxes_to_pipeline_coords",
    "merge_detection_records",
    "redetect_config_from_runtime",
    "redetect_for_nested_candidates",
    "resolve_redetect_candidate",
    "run_detector_pipeline",
    "run_redetect_on_crop",
]
