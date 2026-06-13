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
from vision.detection.pipeline import merge_detection_records
from vision.depth_estimator import infer_depth, load_da2_model
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
            "debug_image_ms",
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

    print_if_enabled("startup", f"Đang tải detector chính ({vcfg.YOLO_BACKEND})...")
    detector = create_detector_service()

    print_if_enabled("startup", "Đang tải Depth Anything V2...")
    da_model = load_da2_model(vcfg.DA2_CONFIG, vcfg.DA2_CHECKPOINT, device)

    return detector, da_model, None, None


def describe_image_with_models(image_path: str, detector, da_model, hands_full, hands_crop):
    del hands_full, hands_crop
    timings = {}
    total_start = time.perf_counter()
    logger.info("Task2 start | image=%s", image_path)

    step_start = time.perf_counter()
    orig = cv2.imread(image_path)
    if orig is None:
        logger.warning("Task2 image read failed | image=%s", image_path)
        _finalize_task2_timings(timings, total_start)
        return "Không đọc được ảnh.", timings, ""

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

    def run_primary():
        return detector._run_primary_detection(img, orig)

    def run_open_vocab():
        if detector._open_vocab_label_backend() is None or not vcfg.OPEN_VOCAB_ENABLED or not candidate_labels:
            return [], 0.0, {}
        return detector._run_open_vocab_detection(img, candidate_labels, orig)

    def run_depth():
        result = {
            "depth_raw": None,
            "new_shape": (0, 0),
            "scale_ratio": 1.0,
            "depth_ms": 0.0,
        }

        step_start = time.perf_counter()
        depth_raw, new_shape, scale_ratio = infer_depth(da_model, img, vcfg.INPUT_SIZE_DEPTH)
        result["depth_ms"] = (time.perf_counter() - step_start) * 1000.0
        result["depth_raw"] = depth_raw
        result["new_shape"] = new_shape
        result["scale_ratio"] = scale_ratio
        return result

    detector_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vision-task2") as executor:
        logger.info("Task2 submit parallel block | primary=%s | open_vocab=%s | depth=%s", True, bool(candidate_labels), True)
        primary_future = executor.submit(run_primary)
        open_vocab_future = executor.submit(run_open_vocab)
        depth_future = executor.submit(run_depth)

        logger.info("Task2 waiting primary_future")
        yolo_records, yolo_ms, primary_pipeline_timings = primary_future.result()
        logger.info("Task2 primary_future done | detections=%d | yolo_ms=%.1f", len(yolo_records), yolo_ms)
        logger.info("Task2 waiting open_vocab_future")
        open_vocab_records, yolo_world_ms, open_vocab_pipeline_timings = open_vocab_future.result()
        logger.info("Task2 open_vocab_future done | detections=%d | yolo_world_ms=%.1f", len(open_vocab_records), yolo_world_ms)
        logger.info("Task2 waiting depth_future")
        depth_result = depth_future.result()
        logger.info("Task2 depth_future done | depth_ms=%.1f | new_shape=%s", depth_result["depth_ms"], depth_result["new_shape"])

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
    timings["depth_ms"] = depth_result["depth_ms"]
    timings["hand_full_ms"] = 0.0
    timings["hand_fallback_ms"] = 0.0
    timings["focal_fallback_ms"] = 0.0
    timings["depth_calib_ms"] = 0.0
    timings["depth_calib_fallback_ms"] = 0.0
    timings["focal_ms"] = 0.0
    timings["debug_image_ms"] = 0.0

    if open_vocab_records and is_enabled("vision_detector_debug", True):
        labels = sorted({record["label"] for record in open_vocab_records})
        logger.info(
            "YOLO-World extra detections outside YOLO labels: labels=%s count=%d merged_total=%d",
            labels,
            len(open_vocab_records),
            len(boxes),
        )
    if is_enabled("vision_detector_debug", True):
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
            "+YOLO-World" if vcfg.OPEN_VOCAB_ENABLED and detector._open_vocab_label_backend() is not None else "",
            _format_detection_boxes_for_log(boxes),
        )

    timings["hand_ms"] = 0.0
    focal_length = vcfg.FIXED_FOCAL_LENGTH_PX

    depth_raw = depth_result["depth_raw"]
    depth_final = cv2.resize(depth_raw, (w, h))

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
    if is_enabled("vision_object_debug", True):
        logger.info(
            "YOLO detections after conf>=%.2f and depth calibration: %s",
            vcfg.CONF_THRESHOLD,
            _format_object_data_for_log(object_data),
        )

    step_start = time.perf_counter()
    debug_objects = filter_objects(
        object_data,
        focal_length,
        h,
        w,
        conf_threshold=vcfg.CONF_THRESHOLD,
        min_depth_m=vcfg.MIN_DEPTH_M,
        max_depth_m=vcfg.MAX_DEPTH_M,
        object_limit=vcfg.DEBUG_IMAGE_MAX_OBJECTS,
    )
    valid_objects = filter_objects(
        object_data,
        focal_length,
        h,
        w,
        conf_threshold=vcfg.CONF_THRESHOLD,
        min_depth_m=vcfg.MIN_DEPTH_M,
        max_depth_m=vcfg.MAX_DEPTH_M,
        object_limit=vcfg.DESCRIPTION_MAX_OBJECTS,
    )
    timings["filter_ms"] = (time.perf_counter() - step_start) * 1000.0
    if is_enabled("vision_object_debug", True):
        logger.info(
            "Debug objects after filter (depth %.2f-%.2fm, max_objects=%d): %s",
            vcfg.MIN_DEPTH_M,
            vcfg.MAX_DEPTH_M,
            vcfg.DEBUG_IMAGE_MAX_OBJECTS,
            _format_valid_objects_for_log(debug_objects),
        )
        logger.info(
            "Description objects after filter (depth %.2f-%.2fm, max_objects=%d): %s",
            vcfg.MIN_DEPTH_M,
            vcfg.MAX_DEPTH_M,
            vcfg.DESCRIPTION_MAX_OBJECTS,
            _format_valid_objects_for_log(valid_objects),
        )
    if not valid_objects:
        timings["scene_ms"] = 0.0
        timings["distance_desc_ms"] = 0.0
        logger.info("Task2 end | no valid objects")
        _finalize_task2_timings(timings, total_start)
        return "Không phát hiện được vật thể hợp lệ để mô tả.", timings, ""
    if is_enabled("vision_debug_image", True) and debug_objects:
        debug_start = time.perf_counter()
        debug_image_path = save_vision_debug_image(image_path, orig, debug_objects, (h, w))
        timings["debug_image_ms"] = (time.perf_counter() - debug_start) * 1000.0
        if debug_image_path:
            logger.info("Vision debug image saved: %s", debug_image_path)
        else:
            logger.warning("Vision debug image could not be created for %s", image_path)
    if len(valid_objects) == 1:
        timings["scene_ms"] = 0.0
        distance_start = time.perf_counter()
        distance_desc = build_distance_description(valid_objects)
        timings["distance_desc_ms"] = (time.perf_counter() - distance_start) * 1000.0
        only_obj = valid_objects[0]
        _finalize_task2_timings(timings, total_start)
        logger.info("Task2 end | single valid object | label=%s", only_obj["label"])
        return (
            f"Truoc mat la cai {only_obj['label']}, cach {only_obj['Z']:.1f}m. "
            "Không thấy vật nào khác xung quanh.",
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
    logger.info("Task2 end | multi object | valid_objects=%d", len(valid_objects))
    return description, timings, distance_desc
