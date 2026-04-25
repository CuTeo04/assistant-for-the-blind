import os
import sys
import math
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import mediapipe as mp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DA2_DIR = os.path.join(ROOT_DIR, "Depth-Anything-V2")
DA2_METRIC_DIR = os.path.join(DA2_DIR, "metric_depth")

preferred_paths = [DA2_METRIC_DIR, DA2_DIR]
for path in reversed(preferred_paths):
    if path in sys.path:
        sys.path.remove(path)
for path in reversed(preferred_paths):
    sys.path.insert(0, path)

from depth_anything_v2.dpt import DepthAnythingV2

# ====================== CONFIG ======================
REAL_HAND_LENGTH_CM = 7.0
KNOWN_DISTANCE_CM   = 45.0
MAX_SIZE            = 1280

device = torch.device("cpu")
torch.set_num_threads(12)

CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth")

# ====================== MODELS ======================
print("✅ Đang tải YOLO11n...")
start_time = time.perf_counter()
yolo = YOLO("yolo11n.pt")
print(f"   → Load YOLO: {time.perf_counter() - start_time:.3f} giây\n")

print("Đang tải Depth Anything V2 Metric Indoor...")
start_time = time.perf_counter()
da_config = {
    'encoder': 'vits',
    'features': 64,
    'out_channels': [48, 96, 192, 384],
    'max_depth': 20.0
}
da_model = DepthAnythingV2(**da_config)
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError("❌ Không tìm thấy checkpoint Depth Anything")

da_model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
da_model.to(device).eval()
print(f"   → Load DA2 Metric: {time.perf_counter() - start_time:.3f} giây\n")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands_full = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
hands_crop = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# ====================== UTILS ======================
def resize_keep_ratio(img, max_size=MAX_SIZE):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def compute_object_size_cm(box_px, depth_m, focal_px):
    x1, y1, x2, y2 = box_px
    depth_cm = depth_m * 100.0
    real_w_cm = ((x2 - x1) / focal_px) * depth_cm
    real_h_cm = ((y2 - y1) / focal_px) * depth_cm
    return real_w_cm, real_h_cm


def detect_hand_landmarks(img_bgr, boxes):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    result = hands_full.process(img_rgb)
    full_time = time.perf_counter() - start

    if result.multi_hand_landmarks:
        print(f"✅ MediaPipe detect tay trên toàn ảnh ({full_time:.3f}s)")
        return result.multi_hand_landmarks[0], (0, 0), img_bgr

    print(f"⚠️ Không thấy tay trên toàn ảnh ({full_time:.3f}s), thử crop person...")

    for det in boxes:
        if det[4] < 0.5:
            continue
        label = yolo.names[int(det[5])]
        if label != "person":
            continue

        x1, y1, x2, y2 = map(int, det[:4])
        box_h = y2 - y1
        box_w = x2 - x1

        crop_y1 = max(0, y1)
        crop_y2 = min(h, y1 + int(box_h * 0.60))
        pad_x   = int(box_w * 0.20)
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
            print(f"✅ MediaPipe detect tay trong crop ({crop_time:.3f}s)")
            return result.multi_hand_landmarks[0], (crop_x1, crop_y1), hand_crop

    return None, None, None


# ====================== MAIN ======================
def process_image(image_path="test_room.jpg"):
    total_start = time.perf_counter()

    # ---- LOAD & RESIZE ----
    start = time.perf_counter()
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError("❌ Không đọc được ảnh")
    load_time = time.perf_counter() - start

    orig_h, orig_w = orig.shape[:2]
    print(f"📷 Original: {orig_w}x{orig_h} | Load ảnh: {load_time:.3f}s")

    start = time.perf_counter()
    img = resize_keep_ratio(orig, MAX_SIZE)
    resize_time = time.perf_counter() - start
    h, w = img.shape[:2]
    print(f"📐 Resized: {w}x{h} | Resize: {resize_time:.3f}s")

    # ---- YOLO detect objects ----
    start = time.perf_counter()
    results = yolo(img, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()
    yolo_time = time.perf_counter() - start
    print(f"🔍 YOLO detection: {yolo_time:.3f}s")

    # ---- Detect hand ----
    start = time.perf_counter()
    hand_landmarks, hand_origin, hand_crop_img = detect_hand_landmarks(img, boxes)
    hand_time = time.perf_counter() - start

    if hand_landmarks is None:
        print("❌ Không detect được tay trong ảnh")
        return

    # ---- Process landmarks & focal length ----
    start = time.perf_counter()
    ch, cw = hand_crop_img.shape[:2]
    ox, oy = hand_origin

    p5  = hand_landmarks.landmark[5]
    p17 = hand_landmarks.landmark[17]

    x5  = int(p5.x  * cw) + ox
    y5  = int(p5.y  * ch) + oy
    x17 = int(p17.x * cw) + ox
    y17 = int(p17.y * ch) + oy

    pixel_hand  = math.sqrt((x5 - x17)**2 + (y5 - y17)**2)
    hand_center = ((x5 + x17) // 2, (y5 + y17) // 2)
    landmark_time = time.perf_counter() - start

    print(f"📏 Hand pixel (5→17): {pixel_hand:.2f}px | Xử lý landmark: {landmark_time:.3f}s")

    if pixel_hand < 5:
        print("❌ Khoảng cách landmark tay quá nhỏ")
        return

    start = time.perf_counter()
    focal_length = (pixel_hand * KNOWN_DISTANCE_CM) / REAL_HAND_LENGTH_CM
    focal_time = time.perf_counter() - start
    print(f"📸 Focal length ≈ {focal_length:.2f}px | Tính focal: {focal_time:.4f}s")

    # ---- Depth Anything ----
    start = time.perf_counter()
    input_size = 518
    scale_ratio = input_size / max(h, w)
    new_h = int(h * scale_ratio)
    new_w = int(w * scale_ratio)

    img_small = cv2.resize(img, (new_w, new_h))
    depth_raw = da_model.infer_image(img_small)

    hx, hy = hand_center
    hx_s = np.clip(int(hx * new_w / w), 0, new_w - 1)
    hy_s = np.clip(int(hy * new_h / h), 0, new_h - 1)

    raw_depth_hand = float(depth_raw[hy_s, hx_s])
    depth_infer_time = time.perf_counter() - start

    print(f"🌊 Depth Anything infer: {depth_infer_time:.3f}s | Raw depth tay: {raw_depth_hand:.4f}m")

    if raw_depth_hand < 0.01:
        print("❌ Depth tại tay không hợp lệ")
        return

    # ---- Calibrate depth ----
    start = time.perf_counter()
    depth_scale = (KNOWN_DISTANCE_CM / 100.0) / raw_depth_hand
    depth_scaled = depth_raw * depth_scale
    depth_final = cv2.resize(depth_scaled, (w, h))
    calib_time = time.perf_counter() - start

    print(f"⚖️  Calibrate depth (scale): {calib_time:.3f}s | Scale = {depth_scale:.4f}")

    # ---- Object depth & size ----
    start = time.perf_counter()
    print("\n=== KẾT QUẢ DETECT ===")
    print(f"{'Vật thể':<14} | {'Depth':>7} | {'Rộng thật':>10} | {'Cao thật':>10}")
    print("-" * 52)

    object_data = []
    for det in boxes:
        if det[4] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, det[:4])
        label = yolo.names[int(det[5])]

        cx = np.clip(int((x1 + x2) / 2), 0, w - 1)
        cy = np.clip(int((y1 + y2) / 2), 0, h - 1)

        depth_m = float(depth_final[cy, cx])
        real_w_cm, real_h_cm = compute_object_size_cm((x1, y1, x2, y2), depth_m, focal_length)

        print(f"{label:<14} | {depth_m:>6.2f}m | {real_w_cm:>9.1f}cm | {real_h_cm:>9.1f}cm")
        object_data.append((det, label, depth_m, real_w_cm, real_h_cm))

    object_time = time.perf_counter() - start

    # ---- Tổng thời gian ----
    total_time = time.perf_counter() - total_start

    print(f"\n{'='*65}")
    print(f"⏱️  TỔNG THỜI GIAN XỬ LÝ: {total_time:.3f} giây")
    print(f"   • Load + Resize : {load_time + resize_time:.3f}s")
    print(f"   • YOLO          : {yolo_time:.3f}s")
    print(f"   • MediaPipe     : {hand_time:.3f}s")
    print(f"   • Landmark + Focal: {landmark_time + focal_time:.3f}s")
    print(f"   • Depth Anything: {depth_infer_time:.3f}s")
    print(f"   • Calibrate     : {calib_time:.3f}s")
    print(f"   • Object post-process: {object_time:.3f}s")
    print(f"{'='*65}\n")

    # ---- DRAW (phần visualize) ----
    img_draw = img.copy()

    for det, label, depth_m, real_w_cm, real_h_cm in object_data:
        x1, y1, x2, y2 = map(int, det[:4])
        ratio = min(depth_m / 5.0, 1.0)
        color = (int(255 * (1 - ratio)), int(255 * ratio), 0)  # BGR

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        line1 = f"{label} | {depth_m:.2f}m"
        line2 = f"{real_w_cm:.0f}cm x {real_h_cm:.0f}cm"
        cv2.putText(img_draw, line1, (x1, max(y1 - 22, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(img_draw, line2, (x1, max(y1 - 5, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    # Vẽ tay
    hand_overlay = img_draw[oy:oy+ch, ox:ox+cw].copy()
    mp_draw.draw_landmarks(hand_overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    img_draw[oy:oy+ch, ox:ox+cw] = hand_overlay

    cv2.circle(img_draw, hand_center, 8, (255, 0, 0), -1)
    cv2.putText(img_draw, f"Hand ref ({KNOWN_DISTANCE_CM:.0f}cm | f={focal_length:.0f}px)",
                (hand_center[0] - 80, hand_center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

    # ---- SHOW ----
    plt.figure(figsize=(15, 9))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Blind Assist | Total time: {total_time:.2f}s | Focal={focal_length:.0f}px")
    plt.tight_layout()
    plt.show()


# ====================== RUN ======================
if __name__ == "__main__":
    process_image("test_room.jpg")