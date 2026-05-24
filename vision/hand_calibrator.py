import math
import time

import cv2
import mediapipe as mp


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


def detect_hand_landmarks(
    img_bgr,
    boxes,
    yolo,
    hands_full,
    hands_crop,
    conf_threshold: float = 0.5,
):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    result = hands_full.process(img_rgb)
    full_time = time.perf_counter() - start

    if result.multi_hand_landmarks:
        print(f"MediaPipe detect tay tren toan anh ({full_time:.3f}s)")
        return result.multi_hand_landmarks[0], (0, 0), img_bgr

    print(f"Khong thay tay tren toan anh ({full_time:.3f}s), thu crop person...")

    for det in boxes:
        if det[4] < conf_threshold:
            continue
        label = yolo.names[int(det[5])]
        if label != "person":
            continue

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
            print(f"MediaPipe detect tay trong crop ({crop_time:.3f}s)")
            return result.multi_hand_landmarks[0], (crop_x1, crop_y1), hand_crop

    return None, None, None


def compute_focal_length(
    landmarks,
    crop_img,
    origin,
    known_distance_cm: float,
    real_hand_length_cm: float,
):
    ch, cw = crop_img.shape[:2]
    ox, oy = origin

    p5 = landmarks.landmark[5]
    p17 = landmarks.landmark[17]

    x5 = int(p5.x * cw) + ox
    y5 = int(p5.y * ch) + oy
    x17 = int(p17.x * cw) + ox
    y17 = int(p17.y * ch) + oy

    pixel_hand = math.sqrt((x5 - x17) ** 2 + (y5 - y17) ** 2)
    if pixel_hand < 5:
        return None, None, pixel_hand

    hand_center = ((x5 + x17) // 2, (y5 + y17) // 2)
    focal_length = (pixel_hand * known_distance_cm) / real_hand_length_cm
    return focal_length, hand_center, pixel_hand
