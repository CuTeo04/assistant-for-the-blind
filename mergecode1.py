import os
import sys
import math
from dataclasses import dataclass, field

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


# ====================== SCENE DESCRIPTION (DLL) ======================
BG_CLASSES = {"wall", "ceiling", "floor", "window"}


@dataclass
class SceneNode:
    label: str
    X: float
    X_left: float
    X_right: float
    Y: float
    Z: float
    cx: int
    cy: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    img_w: int

    prev: "SceneNode | None" = None
    next: "SceneNode | None" = None
    dist_to_next: float = 0.0

    above: list["SceneNode"] = field(default_factory=list)
    below: list["SceneNode"] = field(default_factory=list)
    behind: list["SceneNode"] = field(default_factory=list)


def _x_overlap(a: SceneNode, b: SceneNode) -> bool:
    return not (a.x2 < b.x1 or b.x2 < a.x1)


def _dist_ngang(a: SceneNode, b: SceneNode) -> float:
    # Khoảng cách "ngang trên mặt sàn" tốt hơn dùng tâm:
    # lấy khoảng cách giữa 2 đoạn [X_left, X_right] (cạnh gần nhất), rồi kết hợp với dZ.
    a_min = float(min(a.X_left, a.X_right))
    a_max = float(max(a.X_left, a.X_right))
    b_min = float(min(b.X_left, b.X_right))
    b_max = float(max(b.X_left, b.X_right))

    if a_max < b_min:
        dX = b_min - a_max
    elif b_max < a_min:
        dX = a_min - b_max
    else:
        dX = 0.0

    dZ = float(b.Z - a.Z)
    return float(math.sqrt(dX * dX + dZ * dZ))


def _horiz_dir(ref: SceneNode, other: SceneNode) -> str:
    # Quan hệ trái/phải dựa trên mép bbox (đoạn [X_left, X_right])
    ref_min = float(min(ref.X_left, ref.X_right))
    ref_max = float(max(ref.X_left, ref.X_right))
    oth_min = float(min(other.X_left, other.X_right))
    oth_max = float(max(other.X_left, other.X_right))

    if oth_min > ref_max:
        return "phải"
    if oth_max < ref_min:
        return "trái"

    # Nếu chồng lấp theo trục X:
    # - Nếu lệch độ sâu > 10cm: mô tả trước/sau (tránh gọi "ngay cạnh" sai ngữ cảnh)
    # - Nếu lệch độ sâu nhỏ: mới gọi "ngay cạnh"
    dz = float(other.Z - ref.Z)
    if abs(dz) > 0.10:
        return "sau" if dz > 0 else "trước"

    return "ngay cạnh"


def _priority_score(obj: dict) -> float:
    bg_penalty = 0.5 if obj["label"] in BG_CLASSES else 1.0
    return float(bg_penalty * obj["conf"] / (obj["Z"] + 0.1))


def filter_objects(object_data, focal_length: float, img_h: int, img_w: int):
    objects = []
    for det, label, depth_m, _real_w_cm, _real_h_cm in object_data:
        conf = float(det[4])
        if conf <= 0.5:
            continue
        if not (0.3 <= float(depth_m) <= 8.0):
            continue

        x1, y1, x2, y2 = map(int, det[:4])
        cx = int(np.clip((x1 + x2) / 2, 0, img_w - 1))
        cy = int(np.clip((y1 + y2) / 2, 0, img_h - 1))

        Z = float(depth_m)
        # Theo plan: flip trục Y bằng (img_h - cy).
        # Với trục X, dùng tâm ảnh để phân biệt trái(-)/phải(+).
        X = float((cx - (img_w / 2.0)) * Z / focal_length)
        Y = float((img_h - cy) * Z / focal_length)

        # Với vật thể lớn, dùng 2 mép bounding box để ước lượng "đoạn chiếm chỗ" theo trục X.
        X_left = float((x1 - (img_w / 2.0)) * Z / focal_length)
        X_right = float((x2 - (img_w / 2.0)) * Z / focal_length)
        if X_left > X_right:
            X_left, X_right = X_right, X_left

        objects.append({
            "label": str(label),
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "X": X,
            "X_left": X_left,
            "X_right": X_right,
            "Y": Y,
            "Z": Z,
            "img_w": img_w,
        })

    objects = sorted(objects, key=_priority_score, reverse=True)[:5]
    return objects


def _compute_thresholds(objects: list[dict]):
    depths = [o["Z"] for o in objects]
    xs = [o["X"] for o in objects]

    if len(depths) >= 2:
        q25, q75 = np.percentile(depths, [25, 75])
        iqr = float(q75 - q25)
    else:
        iqr = 0.0

    std_x = float(np.std(xs)) if len(xs) >= 2 else 0.0

    thresh_deep = max(0.40, iqr * 0.30)
    thresh_horiz = max(0.15, std_x * 0.25)
    thresh_vert = 0.30
    return float(thresh_deep), float(thresh_horiz), float(thresh_vert)


def _choose_neo(objects: list[dict]):
    if not objects:
        return None
    img_w = int(objects[0]["img_w"])

    def neo_score(o: dict) -> float:
        center_dist = abs(o["cx"] - img_w / 2) / img_w
        return float(o["Z"] + center_dist * 0.1)

    return min(objects, key=neo_score)


def build_dll(objects: list[dict]):
    if not objects:
        return None

    neo_obj = _choose_neo(objects)
    thresh_deep, _thresh_horiz, thresh_vert = _compute_thresholds(objects)

    # Group ngang chính: các vật có |dZ| < THRESH_DEEP so với neo
    group_main = [o for o in objects if abs(o["Z"] - neo_obj["Z"]) < thresh_deep]
    if not group_main:
        group_main = [neo_obj]

    # Tạo node cho tất cả objects (để tham chiếu above/behind),
    # nhưng chỉ nối prev/next trong group_main.
    nodes: dict[int, SceneNode] = {}
    for idx, o in enumerate(objects):
        nodes[idx] = SceneNode(
            label=o["label"],
            X=float(o["X"]),
            X_left=float(o["X_left"]),
            X_right=float(o["X_right"]),
            Y=float(o["Y"]),
            Z=float(o["Z"]),
            cx=int(o["cx"]),
            cy=int(o["cy"]),
            x1=int(o["x1"]),
            y1=int(o["y1"]),
            x2=int(o["x2"]),
            y2=int(o["y2"]),
            conf=float(o["conf"]),
            img_w=int(o["img_w"]),
        )
        o["_node_id"] = idx

    # Sắp xếp theo mép trái bbox (edge-based) để trục DLL phản ánh "vật nằm bên trái/phải" chính xác hơn.
    group_main_sorted = sorted(group_main, key=lambda o: (float(o["X_left"]), float(o["X_right"]), float(o["Z"])))
    dll_nodes = [nodes[o["_node_id"]] for o in group_main_sorted]

    for i in range(1, len(dll_nodes)):
        left = dll_nodes[i - 1]
        right = dll_nodes[i]
        left.next = right
        right.prev = left
        left.dist_to_next = _dist_ngang(left, right)

    head = dll_nodes[0]
    while head.prev is not None:
        head = head.prev

    # Gắn quan hệ dọc và sâu
    all_same_plane = (max(o["Z"] for o in objects) - min(o["Z"] for o in objects)) < thresh_deep

    for a in dll_nodes:
        for o in objects:
            b = nodes[o["_node_id"]]
            if b is a:
                continue

            dY = b.Y - a.Y
            dZ = b.Z - a.Z

            # Tránh nhầm "ở phía sau" thành "ở phía trên":
            # chỉ kết luận trên/dưới khi hai vật tương đối gần nhau theo trục sâu.
            if _x_overlap(a, b) and abs(dY) > thresh_vert and abs(dZ) <= thresh_deep:
                if dY > 0:
                    a.above.append(b)
                else:
                    a.below.append(b)

            if not all_same_plane and dZ > thresh_deep:
                a.behind.append(b)

    return head


def _dll_to_list(head: SceneNode):
    out = []
    cur = head
    while cur is not None:
        out.append(cur)
        cur = cur.next
    return out


def _node_id(n: SceneNode) -> tuple[str, int, int, int, int]:
    # Dùng bbox + label làm "ID" ổn định để chống trùng quan hệ.
    return (n.label, n.x1, n.y1, n.x2, n.y2)


def _build_display_name_map(dll: list[SceneNode]) -> dict[tuple[str, int, int, int, int], str]:
    # Nếu có nhiều vật trùng label (vd: person/person), thêm số thứ tự để mô tả không bị "trùng".
    all_nodes: dict[tuple[str, int, int, int, int], SceneNode] = {}

    for n in dll:
        all_nodes[_node_id(n)] = n
        for a in n.above:
            all_nodes[_node_id(a)] = a
        for b in n.below:
            all_nodes[_node_id(b)] = b
        for h in n.behind:
            all_nodes[_node_id(h)] = h

    by_label: dict[str, list[SceneNode]] = {}
    for n in all_nodes.values():
        by_label.setdefault(n.label, []).append(n)

    out: dict[tuple[str, int, int, int, int], str] = {}
    for label, nodes in by_label.items():
        # Luôn gắn ID cho *mỗi* vật thể (kể cả chỉ có 1 instance) để không bị mơ hồ khi mô tả.
        # Sắp theo gần -> xa, trái -> phải để đánh số deterministic.
        nodes_sorted = sorted(nodes, key=lambda n: (float(n.Z), float(n.X), float(n.Y), int(n.cx), int(n.cy)))
        for i, n in enumerate(nodes_sorted, start=1):
            out[_node_id(n)] = f"{label} {i}"

    return out


def build_scene_json(dll_head: SceneNode):
    if dll_head is None:
        return None

    dll = _dll_to_list(dll_head)
    if not dll:
        return None

    name_map = _build_display_name_map(dll)

    def name(n: SceneNode) -> str:
        return name_map.get(_node_id(n), n.label)

    img_w = int(dll[0].img_w)

    def neo_score(n: SceneNode) -> float:
        center_dist = abs(n.cx - img_w / 2) / img_w
        return float(n.Z + center_dist * 0.1)

    neo = min(dll, key=neo_score)

    scene_json = {
        "neo": name(neo),
        "neo_depth": float(neo.Z),
        "chuoi_ngang": [],
        "tren_duoi": [],
        "phia_sau": [],
    }

    seen_ngang: set[tuple[str, str, str]] = set()
    seen_tren_duoi: set[tuple[str, str, str]] = set()
    seen_sau: set[tuple[str, str]] = set()

    # Mỗi vật thể chỉ "được mô tả" 1 lần (tránh trùng câu).
    described_targets: set[str] = {name(neo)}

    # Duyệt sang phải từ neo
    cur = neo.next
    prev_name = name(neo)
    while cur:
        cur_name = name(cur)
        direction = _horiz_dir(cur.prev, cur)
        key = (direction, prev_name, cur_name)
        if key not in seen_ngang:
            seen_ngang.add(key)
            scene_json["chuoi_ngang"].append({
                "từ": cur_name,
                "hướng": direction,
                "của": prev_name,
                "dist": float(cur.prev.dist_to_next),
            })
            described_targets.add(cur_name)
        prev_name = cur_name
        cur = cur.next

    # Duyệt sang trái từ neo
    cur = neo.prev
    prev_name = name(neo)
    while cur:
        cur_name = name(cur)
        direction = _horiz_dir(cur.next, cur)
        key = (direction, prev_name, cur_name)
        if key not in seen_ngang:
            seen_ngang.add(key)
            scene_json["chuoi_ngang"].append({
                "từ": cur_name,
                "hướng": direction,
                "của": prev_name,
                "dist": float(cur.next.dist_to_next),
            })
            described_targets.add(cur_name)
        prev_name = cur_name
        cur = cur.prev

    for node in dll:
        node_name = name(node)

        # Ưu tiên mô tả "phía sau" trước, để tránh 1 vật bị mô tả vừa "trên" vừa "sau".
        for behind_node in sorted(node.behind, key=lambda b: float(_dist_ngang(node, b))):
            behind_name = name(behind_node)
            if behind_name in described_targets:
                continue
            key = (node_name, behind_name)
            if key not in seen_sau:
                seen_sau.add(key)
                scene_json["phia_sau"].append({
                    "vật": behind_name,
                    "sau": node_name,
                    "dist": float(_dist_ngang(node, behind_node)),
                })
                described_targets.add(behind_name)

        for above_node in sorted(node.above, key=lambda b: (float(-b.Y), float(b.Z), float(b.X))):
            above_name = name(above_node)
            if above_name in described_targets:
                continue
            key = ("trên", node_name, above_name)
            if key not in seen_tren_duoi:
                seen_tren_duoi.add(key)
                scene_json["tren_duoi"].append({
                    "vật": above_name,
                    "quan_hệ": "trên",
                    "của": node_name,
                })
                described_targets.add(above_name)

        for below_node in sorted(node.below, key=lambda b: (float(b.Y), float(b.Z), float(b.X))):
            below_name = name(below_node)
            if below_name in described_targets:
                continue
            key = ("dưới", node_name, below_name)
            if key not in seen_tren_duoi:
                seen_tren_duoi.add(key)
                scene_json["tren_duoi"].append({
                    "vật": below_name,
                    "quan_hệ": "dưới",
                    "của": node_name,
                })
                described_targets.add(below_name)

    return scene_json


def _fmt_distance(d_m: float) -> str:
    d = float(d_m)
    if d < 0:
        d = -d

    # < 0.5m: đọc chi tiết hơn (cm) để tránh bị làm tròn thành 0.0m
    if d < 0.5:
        cm = int(round(d * 100.0))
        if cm <= 0:
            return f"{d:.2f}m"
        return f"{cm}cm"

    # >= 0.5m: đọc theo 0.1m để câu ngắn, dễ nghe
    return f"{d:.1f}m"


def generate_description(scene_json: dict) -> str:
    if not scene_json:
        return "Không phát hiện được vật thể hợp lệ để mô tả."

    parts = []
    parts.append(
        f"Trước mặt là cái {scene_json['neo']}, "
        f"cách {_fmt_distance(scene_json['neo_depth'])}."
    )

    for rel in scene_json.get("chuoi_ngang", []):
        if rel.get("hướng") == "ngay cạnh":
            parts.append(
                f"Ngay cạnh cái {rel['của']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['từ']}."
            )
        elif rel.get("hướng") == "sau":
            parts.append(
                f"Phía sau cái {rel['của']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['từ']}."
            )
        elif rel.get("hướng") == "trước":
            parts.append(
                f"Phía trước cái {rel['của']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['từ']}."
            )
        else:
            parts.append(
                f"Bên {rel['hướng']} cái {rel['của']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['từ']}."
            )

    for rel in scene_json.get("tren_duoi", []):
        if rel.get("quan_hệ") == "dưới":
            parts.append(f"Bên dưới cái {rel['của']} là cái {rel['vật']}." )
        else:
            parts.append(f"Bên trên cái {rel['của']} là cái {rel['vật']}." )

    for rel in scene_json.get("phia_sau", []):
        parts.append(
            f"Phía sau cái {rel['sau']} "
            f"{_fmt_distance(rel['dist'])} là cái {rel['vật']}."
        )

    return " ".join(parts)


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

    # ---- BUILD SCENE & DESCRIBE ----
    start = time.perf_counter()

    valid_objects = filter_objects(object_data, focal_length, h, w)
    if not valid_objects:
        description = "Không phát hiện được vật thể hợp lệ để mô tả."
    elif len(valid_objects) == 1:
        description = f"Trước mặt là cái {valid_objects[0]['label']}, cách {valid_objects[0]['Z']:.1f}m. Không thấy vật nào khác xung quanh."
    else:
        dll_head = build_dll(valid_objects)
        scene_json = build_scene_json(dll_head)
        description = generate_description(scene_json)

    describe_time = time.perf_counter() - start

    print(f"\n{'='*65}")
    print("🗣️  MÔ TẢ KHÔNG GIAN:")
    print(description)
    print(f"\n⏱️  Sinh mô tả: {describe_time:.3f}s")
    print(f"{'='*65}")


# ====================== RUN ======================
if __name__ == "__main__":
    process_image("test_room.jpg")