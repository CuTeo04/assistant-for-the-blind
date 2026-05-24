import time

import cv2
import numpy as np
import torch

from vision.config import vision_config as vcfg
from vision.depth_estimator import calibrate_depth, infer_depth, load_da2_model
from vision.hand_calibrator import compute_focal_length, detect_hand_landmarks, init_mediapipe_hands
from vision.object_detector import compute_object_size_cm, detect_objects, load_yolo
from vision.scene_builder import build_dll, build_scene_json, filter_objects, generate_description


def resize_keep_ratio(img, max_size: int):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def build_distance_description(objects: list[dict]) -> str:
    if not objects:
        return ""

    counts: dict[str, int] = {}
    parts: list[str] = []
    for obj in objects:
        label = str(obj.get("label", "object"))
        counts[label] = counts.get(label, 0) + 1
        name = f"{label} {counts[label]}"
        parts.append(f"{name}: {obj['Z']:.2f}m")

    return "; ".join(parts)


def init_models():
    device = torch.device(vcfg.DEVICE)
    torch.set_num_threads(vcfg.NUM_THREADS)

    print("Dang tai YOLO...")
    yolo = load_yolo(vcfg.YOLO_MODEL_PATH)

    print("Dang tai Depth Anything V2...")
    da_model = load_da2_model(vcfg.DA2_CONFIG, vcfg.DA2_CHECKPOINT, device)

    hands_full, hands_crop = init_mediapipe_hands()

    return yolo, da_model, hands_full, hands_crop


def describe_image_with_models(image_path: str, yolo, da_model, hands_full, hands_crop):
    timings = {}
    total_start = time.perf_counter()

    step_start = time.perf_counter()
    orig = cv2.imread(image_path)
    if orig is None:
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return "Khong doc duoc anh.", timings, ""

    img = resize_keep_ratio(orig, vcfg.MAX_SIZE)
    h, w = img.shape[:2]
    timings["load_resize_ms"] = (time.perf_counter() - step_start) * 1000.0

    step_start = time.perf_counter()
    boxes = detect_objects(yolo, img)
    timings["yolo_ms"] = (time.perf_counter() - step_start) * 1000.0

    step_start = time.perf_counter()
    hand_landmarks, hand_origin, hand_crop_img = detect_hand_landmarks(
        img, boxes, yolo, hands_full, hands_crop, conf_threshold=vcfg.CONF_THRESHOLD
    )
    timings["hand_ms"] = (time.perf_counter() - step_start) * 1000.0

    if hand_landmarks is None:
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return "Khong detect duoc tay trong anh.", timings, ""

    step_start = time.perf_counter()
    focal_length, hand_center, _pixel_hand = compute_focal_length(
        hand_landmarks,
        hand_crop_img,
        hand_origin,
        vcfg.KNOWN_DISTANCE_CM,
        vcfg.REAL_HAND_LENGTH_CM,
    )
    timings["focal_ms"] = (time.perf_counter() - step_start) * 1000.0

    if focal_length is None:
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return "Khoang cach landmark tay qua nho.", timings, ""

    step_start = time.perf_counter()
    depth_raw, (new_h, new_w), _scale_ratio = infer_depth(da_model, img, vcfg.INPUT_SIZE_DEPTH)
    timings["depth_ms"] = (time.perf_counter() - step_start) * 1000.0

    hx, hy = hand_center
    hx_s = int(np.clip(hx * new_w / w, 0, new_w - 1))
    hy_s = int(np.clip(hy * new_h / h, 0, new_h - 1))

    raw_depth_hand = float(depth_raw[hy_s, hx_s])
    if raw_depth_hand < 0.01:
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return "Depth tai tay khong hop le.", timings, ""

    step_start = time.perf_counter()
    depth_scaled, _depth_scale = calibrate_depth(depth_raw, raw_depth_hand, vcfg.KNOWN_DISTANCE_CM)
    depth_final = cv2.resize(depth_scaled, (w, h))

    object_data = []
    for det in boxes:
        if det[4] < vcfg.CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, det[:4])
        label = yolo.names[int(det[5])]

        cx = np.clip(int((x1 + x2) / 2), 0, w - 1)
        cy = np.clip(int((y1 + y2) / 2), 0, h - 1)

        depth_m = float(depth_final[cy, cx])
        real_w_cm, real_h_cm = compute_object_size_cm((x1, y1, x2, y2), depth_m, focal_length)
        object_data.append((det, label, depth_m, real_w_cm, real_h_cm))

    timings["calib_objects_ms"] = (time.perf_counter() - step_start) * 1000.0

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
    if not valid_objects:
        timings["scene_ms"] = (time.perf_counter() - step_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return "Khong phat hien duoc vat the hop le de mo ta.", timings, ""
    if len(valid_objects) == 1:
        timings["scene_ms"] = (time.perf_counter() - step_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        only_obj = valid_objects[0]
        distance_desc = build_distance_description(valid_objects)
        return (
            f"Truoc mat la cai {only_obj['label']}, cach {only_obj['Z']:.1f}m. "
            "Khong thay vat nao khac xung quanh.",
            timings,
            distance_desc,
        )

    dll_head = build_dll(valid_objects)
    scene_json = build_scene_json(dll_head)
    description = generate_description(scene_json)
    distance_desc = build_distance_description(valid_objects)
    timings["scene_ms"] = (time.perf_counter() - step_start) * 1000.0
    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return description, timings, distance_desc
