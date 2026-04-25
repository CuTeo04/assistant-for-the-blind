import mediapipe as mp

print("mediapipe path =", mp.__file__)
print("mediapipe content =", dir(mp))
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH ======================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METRIC_DIR = os.path.join(ROOT_DIR, "Depth-Anything-V2", "metric_depth")
if METRIC_DIR not in sys.path:
    sys.path.insert(0, METRIC_DIR)

torch.set_num_threads(12)
device = torch.device("cpu")

print(f"🚀 Đang chạy trên CPU | Threads: 12\n")

# ====================== LOAD MODELS ======================
print("✅ YOLO11n (object) loaded")
yolo_object = YOLO("yolo11n.pt")

# Hand Detection Model (dùng YOLO11n-pose hoặc YOLO11n nếu bạn train riêng)
print("✅ YOLO11n Hand Detection loaded")
yolo_hand = YOLO("yolo11n.pt")   # ← Bạn có thể thay bằng model hand chuyên dụng sau

# UniDepthV2
print("Đang tải UniDepthV2...")
from unidepth.models import UniDepthV2
unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
unidepth_model = unidepth_model.to(device)
unidepth_model.eval()
unidepth_model.resolution_level = 4
print("✅ UniDepthV2 loaded\n")

# Depth Anything V2 Metric
print("Đang tải Depth Anything V2 Metric Indoor...")
from depth_anything_v2.dpt import DepthAnythingV2

da_config = {
    'encoder': 'vits',
    'features': 64,
    'out_channels': [48, 96, 192, 384],
    'max_depth': 20.0
}

da_model = DepthAnythingV2(**da_config)

checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth")

if os.path.exists(checkpoint_path):
    da_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    da_model = da_model.to(device)
    da_model.eval()
    print("✅ Depth Anything V2 Metric Indoor loaded\n")
else:
    print("❌ Không tìm thấy checkpoint Depth Anything!")
    da_model = None

print("=" * 95)

def process_image(image_path="test_room.jpg"):
    rgb = np.array(Image.open(image_path).convert("RGB"))
    orig_h, orig_w = rgb.shape[:2]

    # ====================== DETECT OBJECTS ======================
    results = yolo_object(rgb, verbose=False)[0]
    boxes = results.boxes.data.cpu().numpy()

    # ====================== DETECT HAND (calibration) ======================
    hand_results = yolo_hand(rgb, verbose=False)[0]
    hand_boxes = hand_results.boxes.data.cpu().numpy()

    hand_center = None
    hand_conf = 0.0

    for det in hand_boxes:
        if det[4] < 0.5: continue  # confidence threshold
        # Lấy trung tâm bàn tay
        cx = int((det[0] + det[2]) / 2)
        cy = int((det[1] + det[3]) / 2)
        conf = det[4]
        if conf > hand_conf:
            hand_conf = conf
            hand_center = (cx, cy)

    # ====================== UniDepthV2 (tham chiếu) ======================
    rgb_resized_uni = cv2.resize(rgb, (768, 576), interpolation=cv2.INTER_AREA)
    tensor_uni = torch.from_numpy(rgb_resized_uni).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor_uni = tensor_uni.to(device)

    with torch.no_grad():
        pred = unidepth_model.infer(tensor_uni)
    depth_uni = pred["depth"].squeeze().cpu().numpy()

    print(f"\n=== SO SÁNH ĐỘ SÂU TRÊN ẢNH: {image_path} ===\n")
    print("=== UniDepthV2 ===")
    for det in boxes:
        if det[4] < 0.5: continue
        label = yolo_object.names[int(det[5])]
        cx = int((det[0] + det[2]) / 2 * 768 / orig_w)
        cy = int((det[1] + det[3]) / 2 * 576 / orig_h)
        z = float(depth_uni[cy, cx])
        print(f"{label:12} | Độ sâu: {z:6.2f} m")

    if da_model is None:
        return

    # ====================== Depth Anything V2 Metric ======================
    print("\n=== Depth Anything V2 Metric Indoor ===")
    
    input_size = 518
    h, w = rgb.shape[:2]
    scale_ratio = input_size / max(h, w)
    new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)
    
    rgb_resized_da = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    depth_da_raw = da_model.infer_image(rgb_resized_da)
    
    print(f"   Raw depth min/max: {depth_da_raw.min():.4f} / {depth_da_raw.max():.4f}")

    # ====================== TÍNH SCALE TỰ ĐỘNG TỪ BÀN TAY ======================
    scale = 1.0
    if hand_center is not None:
        hx, hy = hand_center
        # Lấy raw depth tại vị trí bàn tay
        raw_at_hand = float(depth_da_raw[int(hy * new_h / orig_h), int(hx * new_w / orig_w)])
        
        if raw_at_hand > 0.01:  # tránh chia cho 0
            scale = 0.5 / raw_at_hand   # 50cm = 0.5m
            print(f"   ✅ Phát hiện bàn tay tại ({hx}, {hy}) | Raw depth = {raw_at_hand:.4f}")
            print(f"   🎯 Scale được tính từ bàn tay (0.5m): {scale:.4f}\n")
        else:
            print("   ⚠️ Raw depth tại bàn tay quá nhỏ, dùng fallback scale.\n")
    else:
        print("   ⚠️ Không phát hiện bàn tay → dùng fallback tự động tìm scale.\n")

    # Nếu không có tay, fallback về cách cũ (so với UniDepth)
    if scale == 1.0 and hand_center is None:
        # ... (có thể giữ phần test scale cũ nếu bạn muốn)
        pass

    # ====================== ÁP DỤNG SCALE ======================
    depth_da_final = cv2.resize(depth_da_raw * scale, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    print("=== Độ sâu sau khi áp dụng scale ===")
    for det in boxes:
        if det[4] < 0.5: continue
        label = yolo_object.names[int(det[5])]
        cx = int((det[0] + det[2]) / 2)
        cy = int((det[1] + det[3]) / 2)
        z = float(depth_da_final[cy, cx])
        print(f"{label:12} | Độ sâu: {z:6.2f} m")

    # ====================== VẼ KẾT QUẢ ======================
    img_draw = rgb.copy()

    # Vẽ object
    for det in boxes:
        if det[4] < 0.5: continue
        x1, y1, x2, y2 = map(int, det[:4])
        label = yolo_object.names[int(det[5])]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        z_uni = float(depth_uni[int(cy * 576 / orig_h), int(cx * 768 / orig_w)])
        z_da  = float(depth_da_final[cy, cx])
        
        text = f"{label}  U:{z_uni:.1f}m  DA:{z_da:.1f}m"
        cv2.putText(img_draw, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Vẽ bàn tay (nếu có) và ghi chú scale
    if hand_center is not None:
        hx, hy = hand_center
        cv2.circle(img_draw, (hx, hy), 10, (255, 0, 0), -1)
        cv2.putText(img_draw, f"Hand (0.5m) - Scale={scale:.3f}", 
                    (hx-80, hy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    plt.figure(figsize=(16, 10))
    plt.imshow(img_draw)
    plt.axis("off")
    plt.title(f"Depth với calibrate tay 50cm | Scale = {scale:.4f}")
    plt.show()


if __name__ == "__main__":
    process_image("test_room.jpg")