import math
import time

import cv2
import mediapipe as mp

from log_settings import print_if_enabled


def init_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands_full = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    hands_crop = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )
    return hands_full, hands_crop


def detect_hand_landmarks_full_image(img_bgr, hands_full):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    result = hands_full.process(img_rgb)
    full_time = time.perf_counter() - start

    if result.multi_hand_landmarks:
        print_if_enabled("hand_debug", f"MediaPipe detect tay tren toan anh ({full_time:.3f}s)")
        return result.multi_hand_landmarks[0], (0, 0), img_bgr, full_time

        print_if_enabled("hand_debug", f"Không thấy tay trên toàn ảnh ({full_time:.3f}s), thử crop person...")
    return None, None, None, full_time


def detect_hand_landmarks_from_boxes(
    img_bgr,
    boxes,
    yolo,
    hands_crop,
    conf_threshold: float = 0.5,
):
    h, w = img_bgr.shape[:2]

    for det in boxes:
        conf = float(det["conf"]) if isinstance(det, dict) else float(det[4])
        if conf < conf_threshold:
            continue
        label = (
            str(det["label"]).strip().lower()
            if isinstance(det, dict)
            else str(yolo.names[int(det[5])]).strip().lower()
        )
        if label != "person":
            continue

        if isinstance(det, dict):
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])
        else:
            x1, y1, x2, y2 = map(int, det[:4])
        box_h = y2 - y1
        box_w = x2 - x1

        crop_y1 = max(0, y1)
        crop_y2 = min(h, y1 + int(box_h * 0.60))
        pad_x = int(box_w * 0.20)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(w, x2 + pad_x)

        hand_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if hand_crop.size == 0:
            continue

        start = time.perf_counter()
        crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        result = hands_crop.process(crop_rgb)
        crop_time = time.perf_counter() - start

        if result.multi_hand_landmarks:
            print_if_enabled("hand_debug", f"MediaPipe detect tay trong crop ({crop_time:.3f}s)")
            return result.multi_hand_landmarks[0], (crop_x1, crop_y1), hand_crop, crop_time

    return None, None, None, 0.0


def detect_hand_landmarks(
    img_bgr,
    boxes,
    yolo,
    hands_full,
    hands_crop,
    conf_threshold: float = 0.5,
):
    hand_landmarks, hand_origin, hand_crop_img, _full_time = detect_hand_landmarks_full_image(
        img_bgr,
        hands_full,
    )
    if hand_landmarks is not None:
        return hand_landmarks, hand_origin, hand_crop_img

    hand_landmarks, hand_origin, hand_crop_img, _crop_time = detect_hand_landmarks_from_boxes(
        img_bgr,
        boxes,
        yolo,
        hands_crop,
        conf_threshold=conf_threshold,
    )
    return hand_landmarks, hand_origin, hand_crop_img


def compute_focal_length(
    landmarks,
    crop_img,
    origin,
    known_distance_cm: float,
    real_hand_length_cm: float,
):
    hand_center, pixel_hand = compute_hand_reference(landmarks, crop_img, origin)
    if pixel_hand < 5:
        return None, None, pixel_hand

    focal_length = (pixel_hand * known_distance_cm) / real_hand_length_cm
    return focal_length, hand_center, pixel_hand


def compute_hand_reference(landmarks, crop_img, origin):
    ch, cw = crop_img.shape[:2]
    ox, oy = origin

    p5 = landmarks.landmark[5]
    p17 = landmarks.landmark[17]

    x5 = int(p5.x * cw) + ox
    y5 = int(p5.y * ch) + oy
    x17 = int(p17.x * cw) + ox
    y17 = int(p17.y * ch) + oy

    pixel_hand = math.sqrt((x5 - x17) ** 2 + (y5 - y17) ** 2)
    hand_center = ((x5 + x17) // 2, (y5 + y17) // 2)
    return hand_center, pixel_hand
