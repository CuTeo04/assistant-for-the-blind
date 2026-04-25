import os
import sys
import math
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
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

BG_CLASSES = {"wall", "ceiling", "floor", "window"}

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


# ====================== SCENE DESCRIPTION MODULE ======================

class Node:
    """Node trong Doubly Linked List đại diện cho một vật thể trong không gian."""
    def __init__(self, label, X, Y, Z, cx, cy, x1, y1, x2, y2):
        self.label = label
        self.X = X          # ngang: trái(-) / phải(+), đơn vị mét
        self.Y = Y          # dọc: dưới(-) / trên(+), đơn vị mét (đã flip cy)
        self.Z = Z          # sâu: khoảng cách từ camera, đơn vị mét
        self.cx = cx        # tọa độ pixel trung tâm box (ngang)
        self.cy = cy        # tọa độ pixel trung tâm box (dọc)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # DLL pointers
        self.prev = None
        self.next = None
        self.dist_to_next = 0.0   # dist_ngang đến node kế (mét)

        # Quan hệ dọc & sâu
        self.above  = []   # các vật nằm trên vật này
        self.below  = []   # các vật nằm dưới vật này
        self.behind = []   # các vật nằm sau vật này


def priority_score(obj):
    """Tính điểm ưu tiên: ưu tiên vật gần, conf cao, không phải background."""
    bg_penalty = 0.5 if obj["label"] in BG_CLASSES else 1.0
    return bg_penalty * obj["conf"] / (obj["Z"] + 0.1)


def filter_objects(object_data, depth_final, focal_length, img_h, img_w):
    """
    Lọc và chuẩn hóa danh sách vật thể:
    - Lọc theo conf > 0.5 và 0.3 <= depth_m <= 8.0
    - Tính tọa độ 3D thực (X_real, Y_real, Z_real)
    - Giữ top-5 theo priority_score
    Trả về list[dict].
    """
    valid = []
    for det, label, depth_m, real_w_cm, real_h_cm in object_data:
        conf = float(det[4])
        if conf <= 0.5:
            continue
        if not (0.3 <= depth_m <= 8.0):
            continue

        x1, y1, x2, y2 = map(int, det[:4])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Tính tọa độ 3D thực — flip trục Y để phù hợp không gian thực
        X_real =  cx              * depth_m / focal_length
        Y_real = (img_h - cy)     * depth_m / focal_length
        Z_real =  depth_m

        valid.append({
            "label":  label,
            "conf":   conf,
            "X":      X_real,
            "Y":      Y_real,
            "Z":      Z_real,
            "cx":     cx,
            "cy":     cy,
            "x1":     x1,
            "y1":     y1,
            "x2":     x2,
            "y2":     y2,
        })

    # Sắp xếp và giữ top-5
    valid.sort(key=priority_score, reverse=True)
    return valid[:5]


def dist_ngang(a, b):
    """Khoảng cách trên mặt phẳng ngang (XZ) giữa hai vật."""
    return math.sqrt((b["X"] - a["X"])**2 + (b["Z"] - a["Z"])**2)


def dist_ngang_nodes(a: Node, b: Node):
    """Khoảng cách ngang giữa hai Node."""
    return math.sqrt((b.X - a.X)**2 + (b.Z - a.Z)**2)


def x_overlap(a, b):
    """Kiểm tra hai bounding box có chồng lấp theo trục X không."""
    return not (a["x2"] < b["x1"] or b["x2"] < a["x1"])


def compute_thresholds(objects):
    """Tính ngưỡng phân loại động dựa trên phân phối thực tế của cảnh."""
    depths = [obj["Z"] for obj in objects]
    xs     = [obj["X"] for obj in objects]

    q25, q75 = np.percentile(depths, [25, 75])

    THRESH_DEEP  = max(0.40, (q75 - q25) * 0.30)
    THRESH_HORIZ = max(0.15, np.std(xs)   * 0.25)
    THRESH_VERT  = 0.30

    return THRESH_DEEP, THRESH_HORIZ, THRESH_VERT


def build_dll(objects):
    """
    Xây dựng Doubly Linked List từ danh sách vật thể đã lọc.
    Trả về (head_node, all_nodes, all_same_plane).
    """
    if not objects:
        return None, [], False

    if len(objects) == 1:
        node = Node(
            objects[0]["label"],
            objects[0]["X"], objects[0]["Y"], objects[0]["Z"],
            objects[0]["cx"], objects[0]["cy"],
            objects[0]["x1"], objects[0]["y1"], objects[0]["x2"], objects[0]["y2"],
        )
        return node, [node], True

    THRESH_DEEP, THRESH_HORIZ, THRESH_VERT = compute_thresholds(objects)

    depths = [obj["Z"] for obj in objects]
    all_same_plane = (max(depths) - min(depths)) < THRESH_DEEP

    # --- Bước 1: Tách nhóm ngang ---
    # Các vật có |dZ| < THRESH_DEEP với nhau → cùng nhóm ngang.
    # Tìm nhóm chứa vật gần nhất (Z nhỏ nhất).
    nearest = min(objects, key=lambda o: o["Z"])

    if all_same_plane:
        group_main = objects[:]
        group_rest = []
    else:
        group_main = [o for o in objects if abs(o["Z"] - nearest["Z"]) < THRESH_DEEP]
        group_rest = [o for o in objects if o not in group_main]

    # --- Bước 2: Sắp xếp nhóm ngang theo X_real tăng dần ---
    group_main.sort(key=lambda o: o["X"])

    # --- Bước 3: Tạo Node và nối thành DLL ---
    nodes_main = [
        Node(o["label"], o["X"], o["Y"], o["Z"],
             o["cx"], o["cy"], o["x1"], o["y1"], o["x2"], o["y2"])
        for o in group_main
    ]

    for i in range(len(nodes_main)):
        if i > 0:
            nodes_main[i].prev = nodes_main[i - 1]
            nodes_main[i - 1].next = nodes_main[i]
            nodes_main[i - 1].dist_to_next = dist_ngang_nodes(nodes_main[i - 1], nodes_main[i])

    # Tạo node cho nhóm còn lại (phía sau)
    nodes_rest = [
        Node(o["label"], o["X"], o["Y"], o["Z"],
             o["cx"], o["cy"], o["x1"], o["y1"], o["x2"], o["y2"])
        for o in group_rest
    ]

    all_nodes = nodes_main + nodes_rest

    # --- Bước 4: Gắn quan hệ dọc và sâu ---
    # Tạo dict để tra cứu object data theo label (dùng cho x_overlap)
    obj_by_label = {o["label"]: o for o in objects}

    for node_a in nodes_main:
        obj_a = obj_by_label[node_a.label]
        for node_b in all_nodes:
            if node_a is node_b:
                continue
            obj_b = obj_by_label[node_b.label]

            dZ = node_b.Z - node_a.Z
            dY = node_b.Y - node_a.Y

            # Quan hệ sâu
            if not all_same_plane and dZ > THRESH_DEEP:
                node_a.behind.append(node_b)

            # Quan hệ dọc (chỉ khi hai box chồng lấp theo X)
            if x_overlap(obj_a, obj_b) and abs(dY) > THRESH_VERT:
                if dY > 0:
                    # node_b cao hơn node_a
                    node_a.above.append(node_b)
                else:
                    node_a.below.append(node_b)

    head = nodes_main[0]
    return head, all_nodes, all_same_plane


def choose_neo(nodes_main, img_w):
    """
    Chọn điểm neo: vật gần nhất (Z nhỏ nhất),
    tiebreak bằng khoảng cách đến tâm ảnh.
    """
    def neo_score(node):
        center_dist = abs(node.cx - img_w / 2) / img_w
        return node.Z + center_dist * 0.1

    return min(nodes_main, key=neo_score)


def build_scene_json(head: Node, all_nodes: list, all_same_plane: bool, img_w: int) -> dict:
    """
    Từ DLL, sinh cấu trúc JSON trung gian hoàn toàn deterministic.
    """
    # Thu thập tất cả node trong DLL chính (từ head đến đuôi)
    dll_nodes = []
    cur = head
    while cur:
        dll_nodes.append(cur)
        cur = cur.next

    # Chọn điểm neo
    neo = choose_neo(dll_nodes, img_w)

    scene_json = {
        "neo": neo.label,
        "neo_depth": round(neo.Z, 1),
        "chuoi_ngang": [],
        "tren_duoi": [],
        "phia_sau": []
    }

    # Duyệt sang phải từ neo
    cur = neo.next
    prev_label = neo.label
    while cur:
        scene_json["chuoi_ngang"].append({
            "từ": cur.label,
            "hướng": "phải",
            "của": prev_label,
            "dist": round(cur.prev.dist_to_next, 1)
        })
        prev_label = cur.label
        cur = cur.next

    # Duyệt sang trái từ neo
    cur = neo.prev
    prev_label = neo.label
    while cur:
        scene_json["chuoi_ngang"].append({
            "từ": cur.label,
            "hướng": "trái",
            "của": prev_label,
            "dist": round(cur.next.dist_to_next, 1)
        })
        prev_label = cur.label
        cur = cur.prev

    # Quan hệ trên/dưới và sau từ tất cả node
    for node in dll_nodes:
        for above_node in node.above:
            scene_json["tren_duoi"].append({
                "vật": above_node.label,
                "quan_hệ": "trên",
                "của": node.label
            })
        for below_node in node.below:
            scene_json["tren_duoi"].append({
                "vật": below_node.label,
                "quan_hệ": "dưới",
                "của": node.label
            })
        if not all_same_plane:
            for behind_node in node.behind:
                scene_json["phia_sau"].append({
                    "vật": behind_node.label,
                    "sau": node.label,
                    "dist": round(dist_ngang_nodes(node, behind_node), 1)
                })

    return scene_json


def generate_description(scene_json: dict) -> str:
    """
    Từ JSON trung gian, sinh câu mô tả tiếng Việt tự nhiên.
    """
    # Edge case: chỉ 1 vật (không có chuỗi ngang, không có quan hệ khác)
    if not scene_json["chuoi_ngang"] and not scene_json["tren_duoi"] and not scene_json["phia_sau"]:
        return (
            f"Trước mặt là cái {scene_json['neo']}, "
            f"cách {scene_json['neo_depth']:.1f}m. "
            f"Không thấy vật nào khác xung quanh."
        )

    parts = []

    # Câu 1: điểm neo
    parts.append(
        f"Trước mặt là cái {scene_json['neo']}, "
        f"cách {scene_json['neo_depth']:.1f}m."
    )

    # Câu 2+: quan hệ ngang
    for rel in scene_json["chuoi_ngang"]:
        parts.append(
            f"Bên {rel['hướng']} cái {rel['của']} "
            f"{rel['dist']:.1f}m là cái {rel['từ']}."
        )

    # Quan hệ trên/dưới
    for rel in scene_json["tren_duoi"]:
        parts.append(
            f"Bên {rel['quan_hệ']} cái {rel['của']} là cái {rel['vật']}."
        )

    # Quan hệ sau
    for rel in scene_json["phia_sau"]:
        parts.append(
            f"Phía sau cái {rel['sau']} "
            f"{rel['dist']:.1f}m là cái {rel['vật']}."
        )

    return " ".join(parts)


def describe_scene(object_data, depth_final, focal_length, img_h, img_w):
    """
    Pipeline tổng thể: từ raw object_data → câu mô tả tiếng Việt.
    Trả về (description_str, scene_json).
    """
    # 1. Lọc và chuẩn hóa
    valid_objects = filter_objects(object_data, depth_final, focal_length, img_h, img_w)

    if not valid_objects:
        return "Không phát hiện vật thể nào trong tầm nhìn.", {}

    # 2. Xây dựng DLL
    head, all_nodes, all_same_plane = build_dll(valid_objects)

    if head is None:
        return "Không phát hiện vật thể nào hợp lệ.", {}

    # 3. Sinh JSON trung gian
    scene_json = build_scene_json(head, all_nodes, all_same_plane, img_w)

    # 4. Sinh câu tiếng Việt
    description = generate_description(scene_json)

    return description, scene_json


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

    # ---- BUILD SCENE & DESCRIBE ----
    start = time.perf_counter()
    description, scene_json = describe_scene(object_data, depth_final, focal_length, h, w)
    describe_time = time.perf_counter() - start

    # ---- Tổng thời gian ----
    total_time = time.perf_counter() - total_start

    print(f"\n{'='*65}")
    print(f"⏱️  TỔNG THỜI GIAN XỬ LÝ: {total_time:.3f} giây")
    print(f"   • Load + Resize      : {load_time + resize_time:.3f}s")
    print(f"   • YOLO               : {yolo_time:.3f}s")
    print(f"   • MediaPipe          : {hand_time:.3f}s")
    print(f"   • Landmark + Focal   : {landmark_time + focal_time:.3f}s")
    print(f"   • Depth Anything     : {depth_infer_time:.3f}s")
    print(f"   • Calibrate          : {calib_time:.3f}s")
    print(f"   • Object post-process: {object_time:.3f}s")
    print(f"   • Sinh mô tả         : {describe_time:.3f}s")
    print(f"{'='*65}")

    print(f"\n{'='*65}")
    print("🗣️  MÔ TẢ KHÔNG GIAN:")
    print(description)
    print(f"\n⏱️  Sinh mô tả: {describe_time:.3f}s")
    print(f"{'='*65}\n")

    # In JSON trung gian để debug (tuỳ chọn)
    if scene_json:
        import json
        print("📋 JSON trung gian (debug):")
        print(json.dumps(scene_json, ensure_ascii=False, indent=2))


# ====================== RUN ======================
if __name__ == "__main__":
    process_image("test_room.jpg")