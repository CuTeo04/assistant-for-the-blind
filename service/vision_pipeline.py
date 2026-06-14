import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from log_settings import is_enabled, print_if_enabled
from vision.config import vision_config as vcfg
from vision.debug_image import save_vision_debug_image
from vision.detection.factory import create_detector_service
from vision.detection.pipeline import boxes_to_detection_records, merge_detection_records
from vision.depth_estimator import calibrate_depth, infer_depth, load_da2_model
from vision.hand_calibrator import (
    compute_focal_length,
    detect_hand_landmarks_full_image,
    init_mediapipe_hands,
)
from vision.label_translation import translate_label
from vision.object_detector import compute_object_size_cm
from vision.object_size_filter import filter_object_data_by_size
from vision.scene_builder import build_dll, build_scene_json, filter_objects, generate_description

logger = logging.getLogger("voice_server.vision")


def _task2_accounted_ms(timings: dict) -> float:
    parallel_block_ms = float(
        timings.get("parallel_block_ms", timings.get("detector_total_ms", 0.0))
    )
    post_parallel_keys = [
        "hand_fallback_ms",
        "focal_fallback_ms",
        "depth_calib_fallback_ms",
        "calib_objects_ms",
        "size_filter_ms",
        "filter_ms",
        "debug_image_ms",
        "scene_ms",
        "distance_desc_ms",
    ]
    if parallel_block_ms <= 0.0:
        parallel_block_ms = sum(
            float(timings.get(key, 0.0))
            for key in (
                "depth_ms",
                "hand_ms",
                "focal_ms",
                "depth_calib_ms",
            )
        )
    elif "parallel_block_ms" not in timings:
        post_parallel_keys.append("depth_calib_ms")
    post_parallel_ms = sum(float(timings.get(key, 0.0)) for key in post_parallel_keys)
    return float(timings.get("load_resize_ms", 0.0)) + parallel_block_ms + post_parallel_ms


def _finalize_task2_timings(timings: dict, total_start: float) -> None:
    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    timings["accounted_ms"] = _task2_accounted_ms(timings)
    timings["unaccounted_ms"] = timings["total_ms"] - timings["accounted_ms"]


def _resolve_request_focal_length(focal_length_px: float | None) -> float:
    if focal_length_px is None:
        return float(vcfg.FIXED_FOCAL_LENGTH_PX)
    try:
        focal = float(focal_length_px)
    except (TypeError, ValueError):
        return float(vcfg.FIXED_FOCAL_LENGTH_PX)
    return focal if focal > 0 else float(vcfg.FIXED_FOCAL_LENGTH_PX)


def _resolve_request_depth_scale(depth_scale: float | None) -> float:
    if depth_scale is None:
        return float(vcfg.DEFAULT_DEPTH_SCALE)
    try:
        scale = float(depth_scale)
    except (TypeError, ValueError):
        return float(vcfg.DEFAULT_DEPTH_SCALE)
    return scale if scale > 0 else float(vcfg.DEFAULT_DEPTH_SCALE)


def _build_empty_timings() -> dict:
    return {
        "load_resize_ms": 0.0,
        "yolo_ms": 0.0,
        "yolo_world_ms": 0.0,
        "detector_total_ms": 0.0,
        "detector_merge_ms": 0.0,
        "detector_overhead_ms": 0.0,
        "yolo_lane_ms": 0.0,
        "yolo_world_lane_ms": 0.0,
        "depth_lane_ms": 0.0,
        "yolo_lane_gap_ms": 0.0,
        "yolo_world_lane_gap_ms": 0.0,
        "depth_lane_gap_ms": 0.0,
        "yolo_combined_pipeline_ms": 0.0,
        "yolo_combined_infer_ms": 0.0,
        "yolo_combined_filter_ms": 0.0,
        "yolo_combined_post_ms": 0.0,
        "yolo_combined_batch_size": 0.0,
        "yolo_combined_image_count": 0.0,
        "yolo_combined_tile_count": 0.0,
        "yolo_world_combined_pipeline_ms": 0.0,
        "yolo_world_combined_infer_ms": 0.0,
        "yolo_world_combined_filter_ms": 0.0,
        "yolo_world_combined_post_ms": 0.0,
        "yolo_world_combined_batch_size": 0.0,
        "yolo_world_combined_image_count": 0.0,
        "yolo_world_combined_tile_count": 0.0,
        "hand_ms": 0.0,
        "hand_full_ms": 0.0,
        "hand_fallback_ms": 0.0,
        "focal_ms": 0.0,
        "focal_fallback_ms": 0.0,
        "depth_ms": 0.0,
        "depth_calib_ms": 0.0,
        "depth_calib_fallback_ms": 0.0,
        "calib_objects_ms": 0.0,
        "size_filter_ms": 0.0,
        "filter_ms": 0.0,
        "scene_ms": 0.0,
        "distance_desc_ms": 0.0,
        "debug_image_ms": 0.0,
        "parallel_block_ms": 0.0,
    }


def _known_hand_distance_cm(hand_distance_cm: float | None) -> float:
    if hand_distance_cm is not None:
        try:
            distance_cm = float(hand_distance_cm)
            if distance_cm > 0:
                return distance_cm
        except (TypeError, ValueError):
            pass
    return float(vcfg.KNOWN_DISTANCE_CM)


def resize_keep_ratio(img, max_size: int):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _format_detection_boxes_for_log(boxes, limit: int = 10) -> str:
    if boxes is None or len(boxes) == 0:
        return "[]"

    sorted_boxes = sorted(
        boxes,
        key=lambda det: float(det.get("conf", 0.0)),
        reverse=True,
    )
    display_boxes = sorted_boxes[:limit] if limit > 0 else sorted_boxes

    parts = []
    for det in display_boxes:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        conf = float(det["conf"])
        label = str(det["label"])
        source = str(det.get("source", "unknown"))
        parts.append(
            f"{label}[{source}](conf={conf:.2f}, box=[{x1},{y1},{x2},{y2}])"
        )

    if len(sorted_boxes) > len(display_boxes):
        parts.append(f"... {len(sorted_boxes) - len(display_boxes)} more")
    return "[" + "; ".join(parts) + "]"


def _format_object_data_for_log(object_data) -> str:
    if not object_data:
        return "[]"

    parts = []
    for det, label, depth_m, real_w_cm, real_h_cm in object_data:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])
        parts.append(
            f"{label}(conf={conf:.2f}, depth={depth_m:.2f}m, "
            f"size={real_w_cm:.1f}x{real_h_cm:.1f}cm, box=[{x1},{y1},{x2},{y2}])"
        )
    return "[" + "; ".join(parts) + "]"


def _format_valid_objects_for_log(objects: list[dict]) -> str:
    if not objects:
        return "[]"

    parts = []
    for obj in objects:
        parts.append(
            f"{obj['label']}(conf={obj['conf']:.2f}, Z={obj['Z']:.2f}m, "
            f"clock={obj['clock_label']}, box=[{obj['x1']},{obj['y1']},{obj['x2']},{obj['y2']}])"
        )
    return "[" + "; ".join(parts) + "]"


def build_distance_description(objects: list[dict]) -> str:
    if not objects:
        return ""

    counts: dict[str, int] = {}
    parts: list[str] = []
    for obj in objects:
        label = translate_label(str(obj.get("label", "object")))
        counts[label] = counts.get(label, 0) + 1
        name = f"{label} {counts[label]}"
        clock_label = str(obj.get("clock_label") or "").strip()
        clock_part = f", {clock_label}" if clock_label else ""
        parts.append(f"{name}: {obj['Z']:.2f}m{clock_part}")

    return "; ".join(parts)


def build_object_brief(objects: list[dict]) -> list[dict]:
    if not objects:
        return []

    counts: dict[str, int] = {}
    items: list[dict] = []
    for obj in objects:
        label = str(obj.get("label", "object"))
        translated_label = translate_label(label)
        counts[translated_label] = counts.get(translated_label, 0) + 1
        items.append(
            {
                "label": label,
                "display_label": f"{translated_label} {counts[translated_label]}",
                "distance_m": float(obj.get("Z", 0.0)),
                "clock_label": str(obj.get("clock_label") or "").strip(),
                "clock_hour": int(obj.get("clock")) if obj.get("clock") is not None else None,
                "confidence": float(obj.get("conf", 0.0)),
                "center_x": int(obj.get("center_x", 0)),
                "center_y": int(obj.get("center_y", 0)),
                "x1": int(obj.get("x1", 0)),
                "y1": int(obj.get("y1", 0)),
                "x2": int(obj.get("x2", 0)),
                "y2": int(obj.get("y2", 0)),
            }
        )
    return items


def init_models():
    device = torch.device(vcfg.DEVICE)
    torch.set_num_threads(vcfg.NUM_THREADS)

    print_if_enabled("startup", f"Đang tải detector chính ({vcfg.YOLO_BACKEND})...")
    detector = create_detector_service()

    print_if_enabled("startup", "Đang tải Depth Anything V2...")
    da_model = load_da2_model(vcfg.DA2_CONFIG, vcfg.DA2_CHECKPOINT, device)
    print_if_enabled("startup", "Đang tải MediaPipe Hands...")
    hands_full, hands_crop = init_mediapipe_hands()

    return detector, da_model, hands_full, hands_crop


def _run_depth_calibration_lane(
    img,
    da_model,
    hands_full,
    hand_distance_cm: float | None = None,
):
    lane_start = time.perf_counter()
    result = {
        "depth_raw": None,
        "new_shape": (0, 0),
        "scale_ratio": 1.0,
        "depth_ms": 0.0,
        "hand_ms": 0.0,
        "hand_full_ms": 0.0,
        "focal_ms": 0.0,
        "depth_calib_ms": 0.0,
        "calibration_info": None,
        "calibration_description": "Không phát hiện được bàn tay để thiết lập camera.",
        "lane_ms": 0.0,
    }

    step_start = time.perf_counter()
    depth_raw, new_shape, scale_ratio = infer_depth(da_model, img, vcfg.INPUT_SIZE_DEPTH)
    result["depth_ms"] = (time.perf_counter() - step_start) * 1000.0
    result["depth_raw"] = depth_raw
    result["new_shape"] = new_shape
    result["scale_ratio"] = scale_ratio

    if hands_full is None:
        result["calibration_description"] = "Không có bộ phát hiện bàn tay để thiết lập camera."
        result["lane_ms"] = (time.perf_counter() - lane_start) * 1000.0
        return result

    step_start = time.perf_counter()
    hand_landmarks, hand_origin, hand_crop_img, hand_full_s = detect_hand_landmarks_full_image(
        img,
        hands_full,
    )
    result["hand_full_ms"] = hand_full_s * 1000.0
    result["hand_ms"] = result["hand_full_ms"]
    if hand_landmarks is None or hand_crop_img is None or hand_origin is None:
        result["lane_ms"] = (time.perf_counter() - lane_start) * 1000.0
        return result

    known_distance_cm = _known_hand_distance_cm(hand_distance_cm)
    focal_length, hand_center, pixel_hand = compute_focal_length(
        hand_landmarks,
        hand_crop_img,
        hand_origin,
        known_distance_cm=known_distance_cm,
        real_hand_length_cm=vcfg.REAL_HAND_LENGTH_CM,
    )
    result["focal_ms"] = (time.perf_counter() - step_start) * 1000.0
    if focal_length is None or hand_center is None:
        result["calibration_description"] = "Không đo được tiêu cự từ bàn tay."
        result["lane_ms"] = (time.perf_counter() - lane_start) * 1000.0
        return result

    h, w = img.shape[:2]
    depth_final = cv2.resize(depth_raw, (w, h))
    hx = int(np.clip(hand_center[0], 0, w - 1))
    hy = int(np.clip(hand_center[1], 0, h - 1))
    raw_hand_depth = float(depth_final[hy, hx])
    if raw_hand_depth <= 0:
        result["calibration_description"] = "Không lấy được độ sâu hợp lệ từ vị trí bàn tay."
        result["lane_ms"] = (time.perf_counter() - lane_start) * 1000.0
        return result

    step_start = time.perf_counter()
    _depth_scaled, depth_scale = calibrate_depth(
        depth_final,
        raw_hand_depth,
        known_distance_cm,
    )
    result["depth_calib_ms"] = (time.perf_counter() - step_start) * 1000.0
    result["calibration_info"] = {
        "focal_length_px": float(focal_length),
        "depth_scale": float(depth_scale),
        "hand_depth_raw_m": raw_hand_depth,
        "known_distance_m": known_distance_cm / 100.0,
        "hand_center": {"x": hx, "y": hy},
        "pixel_hand_span": float(pixel_hand),
        "used_default_hand_distance": hand_distance_cm is None or float(hand_distance_cm) <= 0,
    }
    result["calibration_description"] = (
        "Đã thiết lập camera. "
        f"Tiêu cự sử dụng {result['calibration_info']['focal_length_px']:.1f} pixel. "
        f"Hệ số chuẩn hóa depth {result['calibration_info']['depth_scale']:.3f}."
    )
    result["lane_ms"] = (time.perf_counter() - lane_start) * 1000.0
    return result


def _run_primary_combined_lane(detector, img):
    lane_start = time.perf_counter()
    boxes, timings = detector._run_primary_combined_detection(img)
    return {
        "boxes": boxes,
        "timings": timings,
        "lane_ms": (time.perf_counter() - lane_start) * 1000.0,
    }


def _run_open_vocab_combined_lane(detector, img, candidate_labels: set[str]):
    lane_start = time.perf_counter()
    boxes = np.empty((0, 6), dtype=np.float32)
    timings = {}
    if vcfg.OPEN_VOCAB_ENABLED and candidate_labels:
        boxes, timings = detector._run_open_vocab_combined_detection(
            img,
            candidate_labels,
        )
    return {
        "boxes": boxes,
        "timings": timings,
        "lane_ms": (time.perf_counter() - lane_start) * 1000.0,
    }


def _primary_label_lookup_backend(detector):
    return detector.primary_backend


def _open_vocab_label_lookup_backend(detector):
    return detector.open_vocab_backend


def _open_vocab_records_from_boxes(detector, boxes: np.ndarray, candidate_labels: set[str]) -> list[dict]:
    if boxes.size == 0 or not candidate_labels:
        return []
    label_backend = _open_vocab_label_lookup_backend(detector)
    if label_backend is None:
        return []
    records = boxes_to_detection_records(
        boxes.astype("float32"),
        label_backend.label_lookup(),
        source="yolo_world",
    )
    return [
        record
        for record in records
        if record["label"] in candidate_labels and record["label"] not in vcfg.OPEN_VOCAB_EXCLUDE_LABELS
    ]


def analyze_image_with_calibration(
    image_path: str,
    detector,
    da_model,
    hands_full,
    hands_crop,
    focal_length_px: float | None = None,
    depth_scale: float | None = None,
    hand_distance_cm: float | None = None,
):
    del hands_crop
    resolved_focal = _resolve_request_focal_length(focal_length_px)
    resolved_scale = _resolve_request_depth_scale(depth_scale)

    timings = _build_empty_timings()
    total_start = time.perf_counter()
    logger.info(
        "Task2 calibrated start | image=%s | focal=%.2f | depth_scale=%.4f",
        image_path,
        resolved_focal,
        resolved_scale,
    )

    step_start = time.perf_counter()
    orig = cv2.imread(image_path)
    if orig is None:
        logger.warning("Task2 image read failed | image=%s", image_path)
        _finalize_task2_timings(timings, total_start)
        return {
            "scene_description": "Không đọc được ảnh.",
            "distance_desc": "",
            "object_brief": [],
            "calibration_description": "Không đọc được ảnh.",
            "calibration_info": None,
            "timings": timings,
        }

    img = resize_keep_ratio(orig, vcfg.MAX_SIZE)
    h, w = img.shape[:2]
    timings["load_resize_ms"] = (time.perf_counter() - step_start) * 1000.0
    logger.info("Task2 image ready | image=%s | shape=%sx%s", image_path, h, w)

    prompts, candidate_labels, prompt_ms = detector.prepare_open_vocab_candidates()
    logger.info(
        "Task2 open-vocab candidates ready | prompts=%d | candidates=%d",
        len(prompts),
        len(candidate_labels),
    )

    detector_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vision-task2-flat") as executor:
        primary_future = executor.submit(_run_primary_combined_lane, detector, img)
        open_vocab_future = executor.submit(_run_open_vocab_combined_lane, detector, img, candidate_labels)
        depth_future = executor.submit(
            _run_depth_calibration_lane,
            img,
            da_model,
            hands_full,
            hand_distance_cm,
        )

        primary_result = primary_future.result()
        open_vocab_result = open_vocab_future.result()
        depth_result = depth_future.result()
    primary_timings = primary_result["timings"]
    primary_boxes = primary_result["boxes"]
    open_vocab_timings = open_vocab_result["timings"]
    open_vocab_boxes = open_vocab_result["boxes"]

    primary_label_backend = _primary_label_lookup_backend(detector)
    yolo_records = boxes_to_detection_records(
        primary_boxes,
        primary_label_backend.label_lookup() if primary_label_backend is not None else {},
        source="yolo",
    )
    open_vocab_records = _open_vocab_records_from_boxes(detector, open_vocab_boxes, candidate_labels)

    merge_start = time.perf_counter()
    boxes = merge_detection_records(
        yolo_records,
        open_vocab_records,
        vcfg.OPEN_VOCAB_IOU_THRESHOLD,
    )
    detector_merge_ms = (time.perf_counter() - merge_start) * 1000.0
    detector_total_ms = (time.perf_counter() - detector_start) * 1000.0

    timings["open_vocab_prompt_ms"] = prompt_ms
    timings["yolo_ms"] = float(primary_result["lane_ms"])
    timings["yolo_world_ms"] = float(open_vocab_result["lane_ms"])
    timings["detector_merge_ms"] = detector_merge_ms
    timings["detector_total_ms"] = detector_total_ms
    timings["detector_overhead_ms"] = max(
        0.0,
        detector_total_ms
        - max(float(primary_result["lane_ms"]), float(open_vocab_result["lane_ms"]))
        - detector_merge_ms,
    )
    timings["yolo_lane_ms"] = float(primary_result["lane_ms"])
    timings["yolo_world_lane_ms"] = float(open_vocab_result["lane_ms"])
    timings["depth_lane_ms"] = float(depth_result["lane_ms"])
    timings["parallel_block_ms"] = max(detector_total_ms, timings["depth_lane_ms"])

    timings["yolo_combined_pipeline_ms"] = float(primary_timings.get("pipeline_ms", 0.0))
    timings["yolo_combined_infer_ms"] = float(primary_timings.get("combined_infer_ms", 0.0))
    timings["yolo_combined_filter_ms"] = float(primary_timings.get("combined_filter_ms", 0.0))
    timings["yolo_combined_post_ms"] = float(primary_timings.get("combined_post_ms", 0.0))
    timings["yolo_combined_batch_size"] = float(primary_timings.get("combined_batch_size", 0.0))
    timings["yolo_combined_image_count"] = float(primary_timings.get("combined_image_count", 0.0))
    timings["yolo_combined_tile_count"] = float(primary_timings.get("combined_tile_count", 0.0))
    timings["yolo_redetect_ms"] = float(primary_timings.get("redetect_ms", 0.0))
    timings["yolo_nms_ms"] = float(primary_timings.get("nms_ms", 0.0))
    timings["yolo_lane_gap_ms"] = max(
        0.0,
        timings["yolo_lane_ms"] - timings["yolo_combined_pipeline_ms"],
    )

    timings["yolo_world_combined_pipeline_ms"] = float(open_vocab_timings.get("pipeline_ms", 0.0))
    timings["yolo_world_combined_infer_ms"] = float(open_vocab_timings.get("combined_infer_ms", 0.0))
    timings["yolo_world_combined_filter_ms"] = float(open_vocab_timings.get("combined_filter_ms", 0.0))
    timings["yolo_world_combined_post_ms"] = float(open_vocab_timings.get("combined_post_ms", 0.0))
    timings["yolo_world_combined_batch_size"] = float(open_vocab_timings.get("combined_batch_size", 0.0))
    timings["yolo_world_combined_image_count"] = float(open_vocab_timings.get("combined_image_count", 0.0))
    timings["yolo_world_combined_tile_count"] = float(open_vocab_timings.get("combined_tile_count", 0.0))
    timings["yolo_world_redetect_ms"] = float(open_vocab_timings.get("redetect_ms", 0.0))
    timings["yolo_world_nms_ms"] = float(open_vocab_timings.get("nms_ms", 0.0))
    timings["yolo_world_lane_gap_ms"] = max(
        0.0,
        timings["yolo_world_lane_ms"] - timings["yolo_world_combined_pipeline_ms"],
    )

    timings["depth_ms"] = depth_result["depth_ms"]
    timings["hand_ms"] = depth_result["hand_ms"]
    timings["hand_full_ms"] = depth_result["hand_full_ms"]
    timings["focal_ms"] = depth_result["focal_ms"]
    timings["depth_calib_ms"] = depth_result["depth_calib_ms"]
    timings["depth_lane_gap_ms"] = max(
        0.0,
        timings["depth_lane_ms"]
        - timings["depth_ms"]
        - timings["hand_ms"]
        - timings["focal_ms"]
        - timings["depth_calib_ms"],
    )

    depth_raw = depth_result["depth_raw"]
    depth_resize_start = time.perf_counter()
    depth_final = cv2.resize(depth_raw, (w, h))
    if resolved_scale != 1.0:
        depth_final = depth_final * resolved_scale
    timings["depth_calib_fallback_ms"] = (time.perf_counter() - depth_resize_start) * 1000.0

    step_start = time.perf_counter()
    object_data = []
    for det in boxes:
        if float(det["conf"]) < vcfg.CONF_THRESHOLD:
            continue
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        label = str(det["label"])
        cx = np.clip(int((x1 + x2) / 2), 0, w - 1)
        cy = np.clip(int((y1 + y2) / 2), 0, h - 1)
        depth_m = float(depth_final[cy, cx])
        real_w_cm, real_h_cm = compute_object_size_cm((x1, y1, x2, y2), depth_m, resolved_focal)
        det_box = np.asarray([x1, y1, x2, y2, float(det["conf"]), 0.0], dtype=np.float32)
        object_data.append((det_box, label, depth_m, real_w_cm, real_h_cm))
    timings["calib_objects_ms"] = (time.perf_counter() - step_start) * 1000.0

    step_start = time.perf_counter()
    size_filtered_object_data, relabeled_by_size = filter_object_data_by_size(object_data)
    timings["size_filter_ms"] = (time.perf_counter() - step_start) * 1000.0
    if is_enabled("vision_object_debug", False) and relabeled_by_size:
        logger.info("Size-based relabel applied to %d object(s): %s", len(relabeled_by_size), relabeled_by_size)
    if is_enabled("vision_object_debug", False):
        logger.info("Task2 object_data: %s", _format_object_data_for_log(object_data))
        logger.info("Task2 size_filtered_object_data: %s", _format_object_data_for_log(size_filtered_object_data))

    step_start = time.perf_counter()
    debug_objects = filter_objects(
        size_filtered_object_data,
        resolved_focal,
        h,
        w,
        conf_threshold=vcfg.CONF_THRESHOLD,
        min_depth_m=vcfg.MIN_DEPTH_M,
        max_depth_m=vcfg.MAX_DEPTH_M,
        object_limit=vcfg.DEBUG_IMAGE_MAX_OBJECTS,
    )
    response_objects = filter_objects(
        size_filtered_object_data,
        resolved_focal,
        h,
        w,
        conf_threshold=vcfg.CONF_THRESHOLD,
        min_depth_m=vcfg.MIN_DEPTH_M,
        max_depth_m=vcfg.MAX_DEPTH_M,
        object_limit=max(len(size_filtered_object_data), 1),
    )
    valid_objects = filter_objects(
        size_filtered_object_data,
        resolved_focal,
        h,
        w,
        conf_threshold=vcfg.CONF_THRESHOLD,
        min_depth_m=vcfg.MIN_DEPTH_M,
        max_depth_m=vcfg.MAX_DEPTH_M,
        object_limit=vcfg.DESCRIPTION_MAX_OBJECTS,
    )
    timings["filter_ms"] = (time.perf_counter() - step_start) * 1000.0
    object_brief = build_object_brief(response_objects)
    if is_enabled("vision_object_debug", False):
        logger.info("Task2 valid_objects: %s", _format_valid_objects_for_log(valid_objects))

    if not valid_objects:
        timings["scene_ms"] = 0.0
        timings["distance_desc_ms"] = 0.0
        _finalize_task2_timings(timings, total_start)
        return {
            "scene_description": "Không phát hiện được vật thể hợp lệ để mô tả.",
            "distance_desc": "",
            "object_brief": object_brief,
            "calibration_description": depth_result["calibration_description"],
            "calibration_info": depth_result["calibration_info"],
            "timings": timings,
        }

    if is_enabled("vision_debug_image", True) and debug_objects:
        debug_start = time.perf_counter()
        save_vision_debug_image(image_path, orig, debug_objects, (h, w))
        timings["debug_image_ms"] = (time.perf_counter() - debug_start) * 1000.0

    if len(valid_objects) == 1:
        timings["scene_ms"] = 0.0
        distance_start = time.perf_counter()
        distance_desc = build_distance_description(valid_objects)
        timings["distance_desc_ms"] = (time.perf_counter() - distance_start) * 1000.0
        only_obj = valid_objects[0]
        _finalize_task2_timings(timings, total_start)
        return {
            "scene_description": (
                f"Trước mặt là cái {only_obj['label']}, cách {only_obj['Z']:.1f}m. "
                "Không thấy vật nào khác xung quanh."
            ),
            "distance_desc": distance_desc,
            "object_brief": object_brief,
            "calibration_description": depth_result["calibration_description"],
            "calibration_info": depth_result["calibration_info"],
            "timings": timings,
        }

    scene_start = time.perf_counter()
    dll_head = build_dll(valid_objects)
    scene_json = build_scene_json(dll_head)
    description = generate_description(scene_json)
    timings["scene_ms"] = (time.perf_counter() - scene_start) * 1000.0
    distance_start = time.perf_counter()
    distance_desc = build_distance_description(valid_objects)
    timings["distance_desc_ms"] = (time.perf_counter() - distance_start) * 1000.0
    _finalize_task2_timings(timings, total_start)
    return {
        "scene_description": description,
        "distance_desc": distance_desc,
        "object_brief": object_brief,
        "calibration_description": depth_result["calibration_description"],
        "calibration_info": depth_result["calibration_info"],
        "timings": timings,
    }


def calibrate_camera_with_hand(
    image_path: str,
    da_model,
    hands_full,
    hands_crop,
    hand_distance_cm: float | None = None,
):
    del hands_crop
    timings = _build_empty_timings()
    total_start = time.perf_counter()
    logger.info("Camera setup start | image=%s", image_path)

    step_start = time.perf_counter()
    orig = cv2.imread(image_path)
    if orig is None:
        logger.warning("Camera setup image read failed | image=%s", image_path)
        _finalize_task2_timings(timings, total_start)
        return "Không đọc được ảnh.", timings, None

    img = resize_keep_ratio(orig, vcfg.MAX_SIZE)
    h, w = img.shape[:2]
    timings["load_resize_ms"] = (time.perf_counter() - step_start) * 1000.0

    step_start = time.perf_counter()
    depth_raw, _new_shape, _scale_ratio = infer_depth(da_model, img, vcfg.INPUT_SIZE_DEPTH)
    timings["depth_ms"] = (time.perf_counter() - step_start) * 1000.0
    depth_final = cv2.resize(depth_raw, (w, h))

    step_start = time.perf_counter()
    hand_landmarks, hand_origin, hand_crop_img, hand_full_s = detect_hand_landmarks_full_image(
        img,
        hands_full,
    )
    timings["hand_full_ms"] = hand_full_s * 1000.0
    timings["hand_ms"] = timings["hand_full_ms"]
    if hand_landmarks is None or hand_crop_img is None or hand_origin is None:
        logger.info("Camera setup end | no hand detected")
        _finalize_task2_timings(timings, total_start)
        return "Không phát hiện được bàn tay để thiết lập camera.", timings, None

    known_distance_cm = (
        float(hand_distance_cm)
        if hand_distance_cm is not None and float(hand_distance_cm) > 0
        else float(vcfg.KNOWN_DISTANCE_CM)
    )
    focal_length, hand_center, pixel_hand = compute_focal_length(
        hand_landmarks,
        hand_crop_img,
        hand_origin,
        known_distance_cm=known_distance_cm,
        real_hand_length_cm=vcfg.REAL_HAND_LENGTH_CM,
    )
    if focal_length is None or hand_center is None:
        logger.warning("Camera setup invalid hand span | pixel_hand=%s", pixel_hand)
        _finalize_task2_timings(timings, total_start)
        return "Không đo được tiêu cự từ bàn tay.", timings, None
    timings["focal_ms"] = (time.perf_counter() - step_start) * 1000.0

    hx = int(np.clip(hand_center[0], 0, w - 1))
    hy = int(np.clip(hand_center[1], 0, h - 1))
    raw_hand_depth = float(depth_final[hy, hx])
    if raw_hand_depth <= 0:
        logger.warning("Camera setup invalid hand depth | depth=%s", raw_hand_depth)
        _finalize_task2_timings(timings, total_start)
        return "Không lấy được độ sâu hợp lệ từ vị trí bàn tay.", timings, None

    depth_calib_start = time.perf_counter()
    _depth_scaled, depth_scale = calibrate_depth(
        depth_final,
        raw_hand_depth,
        known_distance_cm,
    )
    timings["depth_calib_ms"] = (time.perf_counter() - depth_calib_start) * 1000.0

    calibration_info = {
        "focal_length_px": float(focal_length),
        "depth_scale": float(depth_scale),
        "hand_depth_raw_m": raw_hand_depth,
        "known_distance_m": known_distance_cm / 100.0,
        "hand_center": {"x": hx, "y": hy},
        "pixel_hand_span": float(pixel_hand),
        "used_default_hand_distance": hand_distance_cm is None or float(hand_distance_cm) <= 0,
    }

    _finalize_task2_timings(timings, total_start)
    logger.info(
        "Camera setup end | focal=%.2f | depth_scale=%.4f | hand_depth_raw=%.3f",
        calibration_info["focal_length_px"],
        calibration_info["depth_scale"],
        calibration_info["hand_depth_raw_m"],
    )
    description = (
        "Đã thiết lập camera. "
        f"Tiêu cự sử dụng {calibration_info['focal_length_px']:.1f} pixel. "
        f"Hệ số chuẩn hóa depth {calibration_info['depth_scale']:.3f}."
    )
    return description, timings, calibration_info


def describe_image_with_models(image_path: str, detector, da_model, hands_full, hands_crop):
    return describe_image_with_calibration(
        image_path,
        detector,
        da_model,
        hands_full,
        hands_crop,
        focal_length_px=vcfg.FIXED_FOCAL_LENGTH_PX,
        depth_scale=vcfg.DEFAULT_DEPTH_SCALE,
    )


def describe_image_with_calibration(
    image_path: str,
    detector,
    da_model,
    hands_full,
    hands_crop,
    focal_length_px: float | None = None,
    depth_scale: float | None = None,
):
    bundle = analyze_image_with_calibration(
        image_path,
        detector,
        da_model,
        hands_full,
        hands_crop,
        focal_length_px=focal_length_px,
        depth_scale=depth_scale,
    )
    return bundle["scene_description"], bundle["timings"], bundle["distance_desc"]
