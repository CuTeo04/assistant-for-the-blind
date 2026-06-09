import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from vision.config import vision_config as vcfg
from vision.detection.factory import create_detector_service
from vision.detection.pipeline import merge_detection_records
from vision.depth_estimator import calibrate_depth, infer_depth, load_da2_model
from vision.hand_calibrator import (
    compute_focal_length,
    detect_hand_landmarks_from_boxes,
    detect_hand_landmarks_full_image,
    init_mediapipe_hands,
)
from vision.object_detector import compute_object_size_cm
from vision.scene_builder import build_dll, build_scene_json, filter_objects, generate_description

logger = logging.getLogger("voice_server.vision")


def _task2_accounted_ms(timings: dict) -> float:
    parallel_block_ms = float(timings.get("detector_total_ms", 0.0))
    post_parallel_ms = sum(
        float(timings.get(key, 0.0))
        for key in (
            "hand_fallback_ms",
            "focal_fallback_ms",
            "depth_calib_fallback_ms",
            "calib_objects_ms",
            "filter_ms",
            "scene_ms",
            "distance_desc_ms",
        )
    )
    return float(timings.get("load_resize_ms", 0.0)) + parallel_block_ms + post_parallel_ms


def _finalize_task2_timings(timings: dict, total_start: float) -> None:
    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    timings["accounted_ms"] = _task2_accounted_ms(timings)
    timings["unaccounted_ms"] = timings["total_ms"] - timings["accounted_ms"]


def resize_keep_ratio(img, max_size: int):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _format_detection_boxes_for_log(boxes) -> str:
    if boxes is None or len(boxes) == 0:
        return "[]"

    parts = []
    for det in boxes:
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
        label = str(obj.get("label", "object"))
        counts[label] = counts.get(label, 0) + 1
        name = f"{label} {counts[label]}"
        clock_label = str(obj.get("clock_label") or "").strip()
        clock_part = f", {clock_label}" if clock_label else ""
        parts.append(f"{name}: {obj['Z']:.2f}m{clock_part}")

    return "; ".join(parts)


def init_models():
    device = torch.device(vcfg.DEVICE)
    torch.set_num_threads(vcfg.NUM_THREADS)

    print(f"Dang tai detector chinh ({vcfg.YOLO_BACKEND})...")
    detector = create_detector_service()

    print("Dang tai Depth Anything V2...")
    da_model = load_da2_model(vcfg.DA2_CONFIG, vcfg.DA2_CHECKPOINT, device)

    hands_full, hands_crop = init_mediapipe_hands()

    return detector, da_model, hands_full, hands_crop


def describe_image_with_models(image_path: str, detector, da_model, hands_full, hands_crop):
    timings = {}
    total_start = time.perf_counter()

    step_start = time.perf_counter()
    orig = cv2.imread(image_path)
    if orig is None:
        _finalize_task2_timings(timings, total_start)
        return "Khong doc duoc anh.", timings, ""

    img = resize_keep_ratio(orig, vcfg.MAX_SIZE)
    h, w = img.shape[:2]
    timings["load_resize_ms"] = (time.perf_counter() - step_start) * 1000.0

    prompts, candidate_labels, prompt_ms = detector.prepare_open_vocab_candidates()

    def run_primary():
        return detector._run_primary_detection(img, orig)

    def run_open_vocab():
        if detector.open_vocab_backend is None or not vcfg.OPEN_VOCAB_ENABLED or not candidate_labels:
            return [], 0.0, {}
        return detector._run_open_vocab_detection(img, candidate_labels, orig)

    def run_depth_hand_full():
        result = {
            "depth_raw": None,
            "new_shape": (0, 0),
            "scale_ratio": 1.0,
            "depth_ms": 0.0,
            "depth_calib_ms": 0.0,
            "depth_final": None,
            "hand_landmarks": None,
            "hand_origin": None,
            "hand_crop_img": None,
            "hand_full_ms": 0.0,
            "focal_length": None,
            "hand_center": None,
            "focal_ms": 0.0,
            "raw_depth_hand": None,
        }

        step_start = time.perf_counter()
        depth_raw, new_shape, scale_ratio = infer_depth(da_model, img, vcfg.INPUT_SIZE_DEPTH)
        result["depth_ms"] = (time.perf_counter() - step_start) * 1000.0
        result["depth_raw"] = depth_raw
        result["new_shape"] = new_shape
        result["scale_ratio"] = scale_ratio

        hand_landmarks, hand_origin, hand_crop_img, hand_full_s = detect_hand_landmarks_full_image(
            img,
            hands_full,
        )
        result["hand_landmarks"] = hand_landmarks
        result["hand_origin"] = hand_origin
        result["hand_crop_img"] = hand_crop_img
        result["hand_full_ms"] = hand_full_s * 1000.0

        if hand_landmarks is not None:
            step_start = time.perf_counter()
            focal_length, hand_center, _pixel_hand = compute_focal_length(
                hand_landmarks,
                hand_crop_img,
                hand_origin,
                vcfg.KNOWN_DISTANCE_CM,
                vcfg.REAL_HAND_LENGTH_CM,
            )
            result["focal_ms"] = (time.perf_counter() - step_start) * 1000.0
            result["focal_length"] = focal_length
            result["hand_center"] = hand_center
            if focal_length is not None:
                new_h, new_w = new_shape
                hx, hy = hand_center
                hx_s = int(np.clip(hx * new_w / w, 0, new_w - 1))
                hy_s = int(np.clip(hy * new_h / h, 0, new_h - 1))
                raw_depth_hand = float(depth_raw[hy_s, hx_s])
                result["raw_depth_hand"] = raw_depth_hand
                if raw_depth_hand >= 0.01:
                    step_start = time.perf_counter()
                    depth_scaled, _depth_scale = calibrate_depth(
                        depth_raw,
                        raw_depth_hand,
                        vcfg.KNOWN_DISTANCE_CM,
                    )
                    result["depth_final"] = cv2.resize(depth_scaled, (w, h))
                    result["depth_calib_ms"] = (time.perf_counter() - step_start) * 1000.0
        return result

    detector_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vision-task2") as executor:
        primary_future = executor.submit(run_primary)
        open_vocab_future = executor.submit(run_open_vocab)
        depth_hand_future = executor.submit(run_depth_hand_full)

        yolo_records, yolo_ms, primary_pipeline_timings = primary_future.result()
        open_vocab_records, yolo_world_ms, open_vocab_pipeline_timings = open_vocab_future.result()
        depth_hand_result = depth_hand_future.result()

    merge_start = time.perf_counter()
    boxes = merge_detection_records(
        yolo_records,
        open_vocab_records,
        vcfg.OPEN_VOCAB_IOU_THRESHOLD,
    )
    detector_merge_ms = (time.perf_counter() - merge_start) * 1000.0
    detector_total_ms = (time.perf_counter() - detector_start) * 1000.0

    timings["open_vocab_prompt_ms"] = prompt_ms
    timings["yolo_ms"] = yolo_ms
    timings["yolo_world_ms"] = yolo_world_ms
    timings["detector_merge_ms"] = detector_merge_ms
    timings["detector_total_ms"] = detector_total_ms
    timings["detector_overhead_ms"] = max(
        0.0,
        detector_total_ms - max(yolo_ms, yolo_world_ms) - detector_merge_ms,
    )
    for key, value in primary_pipeline_timings.items():
        timings[f"yolo_{key}"] = value
    for key, value in open_vocab_pipeline_timings.items():
        timings[f"yolo_world_{key}"] = value
    timings["depth_ms"] = depth_hand_result["depth_ms"]
    timings["hand_full_ms"] = depth_hand_result["hand_full_ms"]
    timings["hand_fallback_ms"] = 0.0
    timings["focal_fallback_ms"] = 0.0
    timings["depth_calib_ms"] = depth_hand_result["depth_calib_ms"]
    timings["depth_calib_fallback_ms"] = 0.0

    if open_vocab_records:
        labels = sorted({record["label"] for record in open_vocab_records})
        logger.info(
            "YOLO-World extra detections outside YOLO labels: labels=%s count=%d merged_total=%d",
            labels,
            len(open_vocab_records),
            len(boxes),
        )
    logger.info(
        "Detector latency breakdown: total=%.1f ms | yolo=%.1f ms | yolo_world=%.1f ms | merge=%.1f ms | overhead=%.1f ms",
        timings["detector_total_ms"],
        timings["yolo_ms"],
        timings["yolo_world_ms"],
        timings["detector_merge_ms"],
        timings["detector_overhead_ms"],
    )
    logger.info(
        "Detections after detector pipeline%s%s merge: %s",
        "+tiled" if vcfg.TILED_ENABLED else "",
        "+YOLO-World" if vcfg.OPEN_VOCAB_ENABLED and detector.open_vocab_backend is not None else "",
        _format_detection_boxes_for_log(boxes),
    )

    hand_landmarks = depth_hand_result["hand_landmarks"]
    hand_origin = depth_hand_result["hand_origin"]
    hand_crop_img = depth_hand_result["hand_crop_img"]
    focal_length = depth_hand_result["focal_length"]
    hand_center = depth_hand_result["hand_center"]
    timings["focal_ms"] = depth_hand_result["focal_ms"]

    if hand_landmarks is None:
        hand_landmarks, hand_origin, hand_crop_img, crop_time_s = detect_hand_landmarks_from_boxes(
            img,
            boxes,
            detector.primary_backend,
            hands_crop,
            conf_threshold=vcfg.CONF_THRESHOLD,
        )
        timings["hand_fallback_ms"] = crop_time_s * 1000.0
        if hand_landmarks is not None:
            focal_step_start = time.perf_counter()
            focal_length, hand_center, _pixel_hand = compute_focal_length(
                hand_landmarks,
                hand_crop_img,
                hand_origin,
                vcfg.KNOWN_DISTANCE_CM,
                vcfg.REAL_HAND_LENGTH_CM,
            )
            timings["focal_fallback_ms"] = (time.perf_counter() - focal_step_start) * 1000.0
            timings["focal_ms"] += timings["focal_fallback_ms"]

    timings["hand_ms"] = timings["hand_full_ms"] + timings["hand_fallback_ms"]

    if hand_landmarks is None:
        _finalize_task2_timings(timings, total_start)
        return "Khong detect duoc tay trong anh.", timings, ""

    if focal_length is None:
        _finalize_task2_timings(timings, total_start)
        return "Khoang cach landmark tay qua nho.", timings, ""

    depth_final = depth_hand_result["depth_final"]
    if depth_final is None:
        depth_raw = depth_hand_result["depth_raw"]
        new_h, new_w = depth_hand_result["new_shape"]

        hx, hy = hand_center
        hx_s = int(np.clip(hx * new_w / w, 0, new_w - 1))
        hy_s = int(np.clip(hy * new_h / h, 0, new_h - 1))

        raw_depth_hand = float(depth_raw[hy_s, hx_s])
        if raw_depth_hand < 0.01:
            _finalize_task2_timings(timings, total_start)
            return "Depth tai tay khong hop le.", timings, ""

        step_start = time.perf_counter()
        depth_scaled, _depth_scale = calibrate_depth(depth_raw, raw_depth_hand, vcfg.KNOWN_DISTANCE_CM)
        depth_final = cv2.resize(depth_scaled, (w, h))
        timings["depth_calib_fallback_ms"] = (time.perf_counter() - step_start) * 1000.0
        timings["depth_calib_ms"] += timings["depth_calib_fallback_ms"]

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
        real_w_cm, real_h_cm = compute_object_size_cm((x1, y1, x2, y2), depth_m, focal_length)
        det_box = np.asarray([x1, y1, x2, y2, float(det["conf"]), 0.0], dtype=np.float32)
        object_data.append((det_box, label, depth_m, real_w_cm, real_h_cm))

    timings["calib_objects_ms"] = (time.perf_counter() - step_start) * 1000.0
    logger.info(
        "YOLO detections after conf>=%.2f and depth calibration: %s",
        vcfg.CONF_THRESHOLD,
        _format_object_data_for_log(object_data),
    )

    step_start = time.perf_counter()
    valid_objects = filter_objects(
        object_data,
        focal_length,
        h,
        w,
        conf_threshold=vcfg.CONF_THRESHOLD,
        min_depth_m=vcfg.MIN_DEPTH_M,
        max_depth_m=vcfg.MAX_DEPTH_M,
        max_objects=vcfg.MAX_OBJECTS,
    )
    timings["filter_ms"] = (time.perf_counter() - step_start) * 1000.0
    logger.info(
        "Valid objects after filter (depth %.2f-%.2fm, max_objects=%d): %s",
        vcfg.MIN_DEPTH_M,
        vcfg.MAX_DEPTH_M,
        vcfg.MAX_OBJECTS,
        _format_valid_objects_for_log(valid_objects),
    )
    if not valid_objects:
        timings["scene_ms"] = 0.0
        timings["distance_desc_ms"] = 0.0
        _finalize_task2_timings(timings, total_start)
        return "Khong phat hien duoc vat the hop le de mo ta.", timings, ""
    if len(valid_objects) == 1:
        timings["scene_ms"] = 0.0
        distance_start = time.perf_counter()
        distance_desc = build_distance_description(valid_objects)
        timings["distance_desc_ms"] = (time.perf_counter() - distance_start) * 1000.0
        _finalize_task2_timings(timings, total_start)
        only_obj = valid_objects[0]
        return (
            f"Truoc mat la cai {only_obj['label']}, cach {only_obj['Z']:.1f}m. "
            "Khong thay vat nao khac xung quanh.",
            timings,
            distance_desc,
        )

    scene_start = time.perf_counter()
    dll_head = build_dll(valid_objects)
    scene_json = build_scene_json(dll_head)
    description = generate_description(scene_json)
    timings["scene_ms"] = (time.perf_counter() - scene_start) * 1000.0
    distance_start = time.perf_counter()
    distance_desc = build_distance_description(valid_objects)
    timings["distance_desc_ms"] = (time.perf_counter() - distance_start) * 1000.0
    _finalize_task2_timings(timings, total_start)
    return description, timings, distance_desc
