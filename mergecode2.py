import os
import sys
import math
import cv2
import torch
import numpy as np
from PIL import Image
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
REAL_HAND_LENGTH_CM  = 7.0     # khoảng cách thật landmark 5→17 (cm)
KNOWN_DISTANCE_M     = 0.45    # tay cách camera (mét) — cố định theo quy ước hệ thống
MAX_SIZE             = 1280

device = torch.device("cpu")
torch.set_num_threads(12)

CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth")

# ====================== MODELS ======================
print("✅ YOLO loaded")
yolo = YOLO("yolo11n.pt")

print("Đang tải Depth Anything V2 Metric Indoor...")
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
print("✅ Depth Anything loaded\n")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands_full = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
hands_crop = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils


# ====================== DEPTH CALIBRATION ======================
def calibrate_depth_affine(depth_raw, raw_at_ref, known_dist_m):
    """
    Affine calibration đúng cho DA2 Metric:

        depth_calibrated = depth_raw + b        (a=1, chỉ bù offset)

    Tại sao KHÔNG dùng depth_raw * scale?
    ─────────────────────────────────────
    DA2 Metric Hypersim đã được train supervised để output mét thực tế.
    Nhân toàn bộ depth map với scalar k = known/raw_at_hand sẽ:
      • Kéo stretch tuyến tính tất cả giá trị → vật ở 3m bị đẩy thành k*3m
      • Phá hủy tính tương đối metric đã học
      • Chỉ đúng nếu model bị lỗi scale đều (affine với b=0) — rất hiếm

    Chỉ bù offset (b) vì:
      • Model thường đúng về relative depth (tỉ lệ giữa các vật)
      • Chỉ có thể bị drift tuyệt đối nhỏ (bias) do domain gap
      • b = known - raw_at_ref bù chính xác drift đó tại điểm tham chiếu

    Nếu có ≥2 điểm tham chiếu đã biết khoảng cách thực:
        a, b = np.polyfit(raw_samples, real_samples, deg=1)  ← fit cả a lẫn b

    Args:
        depth_raw    : depth map từ DA2 (mét), shape (H, W)
        raw_at_ref   : depth tại điểm tham chiếu (tay) trước calibration
        known_dist_m : khoảng cách thực tại điểm tham chiếu (mét)
    Returns:
        depth_calibrated : depth đã bù offset (mét)
        bias             : giá trị b đã áp dụng
    """
    bias = known_dist_m - raw_at_ref
    depth_calibrated = np.clip(depth_raw + bias, 0.05, 20.0)
    return depth_calibrated, bias


# ====================== UTILS ======================
def resize_keep_ratio(img, max_size=MAX_SIZE):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def detect_hand_landmarks(img_bgr, boxes):
    """
    Tầng 1: MediaPipe trên toàn ảnh (confidence 0.5).
    Tầng 2: Crop person box — 60% chiều cao phía trên,
             mở rộng 20% sang 2 bên để không cắt mất tay.
    """
    h, w = img_bgr.shape[:2]

    # Tầng 1: toàn ảnh
    result = hands_full.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        print("✅ MediaPipe detect tay — toàn ảnh")
        return result.multi_hand_landmarks[0], (0, 0), img_bgr

    print("⚠️  Không thấy tay trên toàn ảnh, thử crop person...")

    # Tầng 2: crop person
    for det in boxes:
        if det[4] < 0.5 or yolo.names[int(det[5])] != "person":
            continue

        x1, y1, x2, y2 = map(int, det[:4])
        bh, bw = y2 - y1, x2 - x1
        crop_y1 = max(0, y1)
        crop_y2 = min(h, y1 + int(bh * 0.60))
        pad_x   = int(bw * 0.20)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(w, x2 + pad_x)

        crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        result = hands_crop.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            print(f"✅ MediaPipe detect tay — crop person ({crop_x1},{crop_y1})→({crop_x2},{crop_y2})")
            return result.multi_hand_landmarks[0], (crop_x1, crop_y1), crop

    return None, None, None


def compute_object_size_cm(box_px, depth_m, focal_px):
    """
    Pinhole camera model:  real_size = (pixel_size / focal_px) × depth_cm
    """
    x1, y1, x2, y2 = box_px
    depth_cm = depth_m * 100.0
    return ((x2 - x1) / focal_px) * depth_cm, ((y2 - y1) / focal_px) * depth_cm


# ====================== MAIN ======================
def process_image(image_path="test_room.jpg"):
    # ── Load & resize ──────────────────────────────────────────────
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError("❌ Không đọc được ảnh")
    print(f"📷 Original : {orig.shape[1]}x{orig.shape[0]}")

    img = resize_keep_ratio(orig, MAX_SIZE)
    h, w = img.shape[:2]
    print(f"📐 Resized  : {w}x{h}")

    # ── YOLO ───────────────────────────────────────────────────────
    boxes = yolo(img, verbose=False)[0].boxes.data.cpu().numpy()

    # ── Hand detection ─────────────────────────────────────────────
    hand_lm, hand_origin, hand_crop_img = detect_hand_landmarks(img, boxes)
    if hand_lm is None:
        print("❌ Không detect được tay")
        return

    ch, cw = hand_crop_img.shape[:2]
    ox, oy = hand_origin

    p5  = hand_lm.landmark[5];  x5  = int(p5.x  * cw) + ox;  y5  = int(p5.y  * ch) + oy
    p17 = hand_lm.landmark[17]; x17 = int(p17.x * cw) + ox;  y17 = int(p17.y * ch) + oy

    pixel_hand  = math.sqrt((x5-x17)**2 + (y5-y17)**2)
    hand_center = ((x5+x17)//2, (y5+y17)//2)
    print(f"📏 Hand pixel (5→17): {pixel_hand:.2f}px")

    if pixel_hand < 5:
        print("❌ Landmark tay lỗi (khoảng cách pixel quá nhỏ)")
        return

    # ── Focal length ───────────────────────────────────────────────
    focal_px = (pixel_hand * (KNOWN_DISTANCE_M * 100)) / REAL_HAND_LENGTH_CM
    print(f"📸 Focal length : {focal_px:.2f}px")

    # ── Depth Anything V2 (metric) ─────────────────────────────────
    scale_r = 518 / max(h, w)
    new_h, new_w = int(h * scale_r), int(w * scale_r)
    depth_raw = da_model.infer_image(cv2.resize(img, (new_w, new_h)))
    # depth_raw: đã là mét thực tế — KHÔNG nhân thêm bất kỳ scalar nào

    hx_s = np.clip(int(hand_center[0] * new_w / w), 0, new_w-1)
    hy_s = np.clip(int(hand_center[1] * new_h / h), 0, new_h-1)
    raw_at_hand = float(depth_raw[hy_s, hx_s])

    print(f"\n📊 Depth tại tay:")
    print(f"   DA2 raw       : {raw_at_hand:.4f} m")
    print(f"   Thực tế       : {KNOWN_DISTANCE_M:.4f} m")

    if raw_at_hand < 0.01:
        print("❌ Depth tại tay không hợp lệ")
        return

    # ── Affine calibration (chỉ bù offset b, giữ a=1) ─────────────
    depth_cal, bias = calibrate_depth_affine(depth_raw, raw_at_hand, KNOWN_DISTANCE_M)
    print(f"   Bias (offset) : {bias:+.4f} m", end="  ")
    print("⚠️  DA2 có drift đáng kể" if abs(bias) > 0.10 else "✅ DA2 khá chính xác")

    depth_final = cv2.resize(depth_cal, (w, h))

    # ── Object depth & size ────────────────────────────────────────
    print(f"\n{'Vật thể':<14} | {'Depth':>7} | {'Rộng':>8} | {'Cao':>8}")
    print("─" * 46)

    object_data = []
    for det in boxes:
        if det[4] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, det[:4])
        label = yolo.names[int(det[5])]
        cx = np.clip((x1+x2)//2, 0, w-1)
        cy = np.clip((y1+y2)//2, 0, h-1)
        dm = float(depth_final[cy, cx])
        rw, rh = compute_object_size_cm((x1,y1,x2,y2), dm, focal_px)
        print(f"{label:<14} | {dm:>6.3f}m | {rw:>7.1f}cm | {rh:>7.1f}cm")
        object_data.append((det, label, dm, rw, rh))

    # ── Draw ───────────────────────────────────────────────────────
    img_draw = img.copy()

    for det, label, dm, rw, rh in object_data:
        x1, y1, x2, y2 = map(int, det[:4])
        ratio = min(dm / 5.0, 1.0)
        col   = (int(255*(1-ratio)), int(255*ratio), 0)   # đỏ=gần, xanh=xa (BGR)
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), col, 2)
        cv2.putText(img_draw, f"{label} {dm:.2f}m",
                    (x1, max(y1-22, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(img_draw, f"{rw:.0f}cm x {rh:.0f}cm",
                    (x1, max(y1-5, 30)),  cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

    # Hand overlay (landmarks trong crop → paste về ảnh đầy đủ)
    overlay = img_draw[oy:oy+ch, ox:ox+cw].copy()
    mp_draw.draw_landmarks(overlay, hand_lm, mp_hands.HAND_CONNECTIONS)
    img_draw[oy:oy+ch, ox:ox+cw] = overlay

    cv2.circle(img_draw, hand_center, 8, (255,100,0), -1)
    cv2.putText(img_draw,
                f"Hand ref={KNOWN_DISTANCE_M*100:.0f}cm | f={focal_px:.0f}px | bias={bias:+.3f}m",
                (hand_center[0]-110, hand_center[1]-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,100,0), 2)

    plt.figure(figsize=(15, 9))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Blind Assist | f={focal_px:.0f}px | bias={bias:+.3f}m | DA2 Metric affine-calibrated")
    plt.tight_layout()
    plt.show()


# ====================== RUN ======================
if __name__ == "__main__":
    process_image("test_room.jpg")