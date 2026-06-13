import math
from dataclasses import dataclass, field

import numpy as np

from vision.clock_direction import compute_clock_direction, format_clock_label

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
    ref_min = float(min(ref.X_left, ref.X_right))
    ref_max = float(max(ref.X_left, ref.X_right))
    oth_min = float(min(other.X_left, other.X_right))
    oth_max = float(max(other.X_left, other.X_right))

    if oth_min > ref_max:
        return "phải"
    if oth_max < ref_min:
        return "trái"

    dz = float(other.Z - ref.Z)
    if abs(dz) > 0.10:
        return "sau" if dz > 0 else "trước"

    return "ngay cạnh"


def _priority_score(obj: dict) -> float:
    bg_penalty = 0.5 if obj["label"] in BG_CLASSES else 1.0
    return float(bg_penalty * obj["conf"] / (obj["Z"] + 0.1))


def _selection_score(obj: dict) -> tuple[float, float]:
    return -float(obj["Z"]), float(obj["conf"]) + _priority_score(obj)


def filter_objects(
    object_data,
    focal_length: float,
    img_h: int,
    img_w: int,
    conf_threshold: float,
    min_depth_m: float,
    max_depth_m: float,
    object_limit: int,
):
    objects = []
    for det, label, depth_m, _real_w_cm, _real_h_cm in object_data:
        conf = float(det[4])
        if conf <= conf_threshold:
            continue
        if not (min_depth_m <= float(depth_m) <= max_depth_m):
            continue

        x1, y1, x2, y2 = map(int, det[:4])
        cx = int(np.clip((x1 + x2) / 2, 0, img_w - 1))
        cy = int(np.clip((y1 + y2) / 2, 0, img_h - 1))

        Z = float(depth_m)
        X = float((cx - (img_w / 2.0)) * Z / focal_length)
        Y = float((img_h - cy) * Z / focal_length)

        X_left = float((x1 - (img_w / 2.0)) * Z / focal_length)
        X_right = float((x2 - (img_w / 2.0)) * Z / focal_length)
        if X_left > X_right:
            X_left, X_right = X_right, X_left
        clock = compute_clock_direction(
            cx,
            cy,
            image_width=img_w,
            image_height=img_h,
            dead_zone_px=min(img_w, img_h) * 0.04,
        )

        objects.append(
            {
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
                "img_h": img_h,
                "clock": clock,
                "clock_label": format_clock_label(clock),
            }
        )

    objects = sorted(objects, key=_selection_score, reverse=True)[:object_limit]
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

    group_main = [o for o in objects if abs(o["Z"] - neo_obj["Z"]) < thresh_deep]
    if not group_main:
        group_main = [neo_obj]

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

    group_main_sorted = sorted(
        group_main, key=lambda o: (float(o["X_left"]), float(o["X_right"]), float(o["Z"]))
    )
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

    all_same_plane = (max(o["Z"] for o in objects) - min(o["Z"] for o in objects)) < thresh_deep

    for a in dll_nodes:
        for o in objects:
            b = nodes[o["_node_id"]]
            if b is a:
                continue

            dY = b.Y - a.Y
            dZ = b.Z - a.Z

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
    return (n.label, n.x1, n.y1, n.x2, n.y2)


def _build_display_name_map(dll: list[SceneNode]) -> dict[tuple[str, int, int, int, int], str]:
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
        nodes_sorted = sorted(
            nodes, key=lambda n: (float(n.Z), float(n.X), float(n.Y), int(n.cx), int(n.cy))
        )
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

    described_targets: set[str] = {name(neo)}

    cur = neo.next
    prev_name = name(neo)
    while cur:
        cur_name = name(cur)
        direction = _horiz_dir(cur.prev, cur)
        key = (direction, prev_name, cur_name)
        if key not in seen_ngang:
            seen_ngang.add(key)
            scene_json["chuoi_ngang"].append(
                {
                    "tu": cur_name,
                    "huong": direction,
                    "cua": prev_name,
                    "dist": float(cur.prev.dist_to_next),
                }
            )
            described_targets.add(cur_name)
        prev_name = cur_name
        cur = cur.next

    cur = neo.prev
    prev_name = name(neo)
    while cur:
        cur_name = name(cur)
        direction = _horiz_dir(cur.next, cur)
        key = (direction, prev_name, cur_name)
        if key not in seen_ngang:
            seen_ngang.add(key)
            scene_json["chuoi_ngang"].append(
                {
                    "tu": cur_name,
                    "huong": direction,
                    "cua": prev_name,
                    "dist": float(cur.next.dist_to_next),
                }
            )
            described_targets.add(cur_name)
        prev_name = cur_name
        cur = cur.prev

    for node in dll:
        node_name = name(node)

        for behind_node in sorted(node.behind, key=lambda b: float(_dist_ngang(node, b))):
            behind_name = name(behind_node)
            if behind_name in described_targets:
                continue
            key = (node_name, behind_name)
            if key not in seen_sau:
                seen_sau.add(key)
                scene_json["phia_sau"].append(
                    {
                        "vat": behind_name,
                        "sau": node_name,
                        "dist": float(_dist_ngang(node, behind_node)),
                    }
                )
                described_targets.add(behind_name)

        for above_node in sorted(node.above, key=lambda b: (float(-b.Y), float(b.Z), float(b.X))):
            above_name = name(above_node)
            if above_name in described_targets:
                continue
            key = ("tren", node_name, above_name)
            if key not in seen_tren_duoi:
                seen_tren_duoi.add(key)
                scene_json["tren_duoi"].append(
                    {
                        "vat": above_name,
                        "quan_he": "tren",
                        "cua": node_name,
                    }
                )
                described_targets.add(above_name)

        for below_node in sorted(node.below, key=lambda b: (float(b.Y), float(b.Z), float(b.X))):
            below_name = name(below_node)
            if below_name in described_targets:
                continue
            key = ("duoi", node_name, below_name)
            if key not in seen_tren_duoi:
                seen_tren_duoi.add(key)
                scene_json["tren_duoi"].append(
                    {
                        "vat": below_name,
                        "quan_he": "duoi",
                        "cua": node_name,
                    }
                )
                described_targets.add(below_name)

    return scene_json


def _fmt_distance(d_m: float) -> str:
    d = float(d_m)
    if d < 0:
        d = -d

    if d < 0.5:
        cm = int(round(d * 100.0))
        if cm <= 0:
            return f"{d:.2f}m"
        return f"{cm}cm"

    return f"{d:.1f}m"


def generate_description(scene_json: dict) -> str:
    if not scene_json:
        return "Không phát hiện được vật thể hợp lệ để mô tả."

    direction_map = {
        "phải": "phải",
        "trái": "trái",
        "trước": "trước",
        "sau": "sau",
        "ngay cạnh": "ngay cạnh",
    }

    parts = []
    parts.append(
        f"Trước mặt là cái {scene_json['neo']}, "
        f"cách {_fmt_distance(scene_json['neo_depth'])}."
    )

    for rel in scene_json.get("chuoi_ngang", []):
        direction = direction_map.get(rel.get("huong"), rel.get("huong"))
        if rel.get("huong") == "ngay cạnh":
            parts.append(
                f"Ngay cạnh cái {rel['cua']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['tu']}."
            )
        elif rel.get("huong") == "sau":
            parts.append(
                f"Phía sau cái {rel['cua']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['tu']}."
            )
        elif rel.get("huong") == "trước":
            parts.append(
                f"Phía trước cái {rel['cua']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['tu']}."
            )
        else:
            parts.append(
                f"Bên {direction} cái {rel['cua']} "
                f"{_fmt_distance(rel['dist'])} là cái {rel['tu']}."
            )

    for rel in scene_json.get("tren_duoi", []):
        if rel.get("quan_he") == "duoi":
            parts.append(f"Bên dưới cái {rel['cua']} là cái {rel['vat']}.")
        else:
            parts.append(f"Bên trên cái {rel['cua']} là cái {rel['vat']}.")

    for rel in scene_json.get("phia_sau", []):
        parts.append(
            f"Phía sau cái {rel['sau']} "
            f"{_fmt_distance(rel['dist'])} là cái {rel['vat']}."
        )

    return " ".join(parts)
