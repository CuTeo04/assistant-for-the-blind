import logging
import os
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np


logger = logging.getLogger("voice_server.vision")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBUG_IMAGE_DIR = os.path.join(ROOT_DIR, "debug_outputs", "vision")
SIDEBAR_WIDTH = 420
BOX_COLORS = [
    (60, 220, 60),
    (80, 180, 255),
    (255, 180, 60),
    (220, 90, 220),
    (80, 255, 220),
]


def _build_display_names(objects: list[dict]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    names: list[str] = []
    for obj in objects:
        label = str(obj.get("label", "object"))
        counts[label] += 1
        names.append(f"{label} {counts[label]}")
    return names


def _distance_between_objects(obj_a: dict, obj_b: dict) -> float:
    dx = float(obj_a["X"] - obj_b["X"])
    dy = float(obj_a["Y"] - obj_b["Y"])
    dz = float(obj_a["Z"] - obj_b["Z"])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _nearest_neighbors(objects: list[dict], names: list[str], index: int, limit: int = 2) -> str:
    source = objects[index]
    distances: list[tuple[float, str]] = []
    for other_index, other in enumerate(objects):
        if other_index == index:
            continue
        distances.append((_distance_between_objects(source, other), names[other_index]))

    distances.sort(key=lambda item: item[0])
    nearby = [f"{name} {distance:.2f}m" for distance, name in distances[:limit]]
    return ", ".join(nearby)


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    anchor_x: int,
    anchor_y: int,
    color: tuple[int, int, int],
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 18
    max_width = 0

    for line in lines:
        (width, _height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, width)

    block_height = line_height * len(lines) + 6
    x = max(4, anchor_x)
    y = max(block_height + 4, anchor_y)
    x2 = min(image.shape[1] - 4, x + max_width + 10)
    y2 = min(image.shape[0] - 4, y + 6)
    x = max(4, x2 - max_width - 10)
    y = max(block_height + 4, y2 - 6)

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - block_height), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    text_y = y - block_height + 16
    for line in lines:
        cv2.putText(image, line, (x + 5, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        text_y += line_height


def _draw_box_badge(
    image: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.25
    thickness = 6
    padding_x = 24
    padding_y = 18
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    left = max(4, x1)
    top = max(4, y1 - height - baseline - padding_y * 2 - 6)
    right = min(image.shape[1] - 4, left + width + padding_x * 2)
    bottom = top + height + baseline + padding_y * 2

    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    cv2.putText(
        image,
        text,
        (left + padding_x, bottom - baseline - padding_y),
        font,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def _build_sidebar(canvas: np.ndarray, objects: list[dict], names: list[str]) -> np.ndarray:
    image_h, image_w = canvas.shape[:2]
    sidebar_width = SIDEBAR_WIDTH * 2
    sidebar = np.full((image_h, sidebar_width, 3), 24, dtype=np.uint8)

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    body_font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        sidebar,
        "VISION DEBUG",
        (20, 42),
        title_font,
        2.4,
        (240, 240, 240),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        sidebar,
        f"objects: {len(objects)}",
        (20, 96),
        body_font,
        1.6,
        (180, 180, 180),
        3,
        cv2.LINE_AA,
    )

    top = 130
    bottom_margin = 18
    row_gap = 12
    available_height = max(120, image_h - top - bottom_margin)
    block_height = max(120, int((available_height - row_gap * max(0, len(objects) - 1)) / max(1, len(objects))))
    for index, obj in enumerate(objects):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        block_y1 = top + index * (block_height + row_gap)
        block_y2 = min(image_h - 12, block_y1 + block_height)
        if block_y1 >= image_h - 20:
            break

        cv2.rectangle(sidebar, (14, block_y1), (sidebar_width - 14, block_y2), (45, 45, 45), -1)
        cv2.rectangle(sidebar, (14, block_y1), (sidebar_width - 14, block_y2), color, 3)
        badge_bottom = min(block_y2 - 12, block_y1 + 52)
        cv2.rectangle(sidebar, (24, block_y1 + 12), (66, badge_bottom), color, -1)
        cv2.putText(
            sidebar,
            str(index + 1),
            (32, min(block_y2 - 16, block_y1 + 40)),
            body_font,
            1.0,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )

        lines = [
            names[index],
            f"conf: {float(obj['conf']):.2f}",
            f"camera: {float(obj['Z']):.2f}m",
        ]
        nearby = _nearest_neighbors(objects, names, index)
        if nearby:
            lines.append(f"near: {nearby}")

        text_y = block_y1 + 34
        for line_index, line in enumerate(lines):
            scale = 1.95 if line_index == 0 else 1.56
            color_text = (245, 245, 245) if line_index == 0 else (215, 215, 215)
            cv2.putText(
                sidebar,
                line,
                (84, text_y),
                body_font,
                scale,
                color_text,
                4,
                cv2.LINE_AA,
            )
            text_y += max(34, int(block_height * 0.26))

    combined = np.concatenate([canvas, sidebar], axis=1)
    divider_x = image_w
    cv2.line(combined, (divider_x, 0), (divider_x, image_h), (90, 90, 90), 2)
    return combined


def save_vision_debug_image(
    image_path: str,
    original_image: np.ndarray,
    objects: list[dict],
    working_shape: tuple[int, int],
) -> str | None:
    if original_image is None or len(objects) == 0:
        return None

    canvas = original_image.copy()
    names = _build_display_names(objects)
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
    working_h, working_w = working_shape
    scale_x = float(canvas.shape[1]) / float(max(working_w, 1))
    scale_y = float(canvas.shape[0]) / float(max(working_h, 1))

    for index, obj in enumerate(objects):
        x1 = int(round(float(obj["x1"]) * scale_x))
        y1 = int(round(float(obj["y1"]) * scale_y))
        x2 = int(round(float(obj["x2"]) * scale_x))
        y2 = int(round(float(obj["y2"]) * scale_y))
        cx = int(round(float(obj["cx"]) * scale_x))
        cy = int(round(float(obj["cy"]) * scale_y))
        color = BOX_COLORS[index % len(BOX_COLORS)]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, (cx, cy), 5, color, -1)
        _draw_box_badge(canvas, f"{index + 1}. {names[index]}", x1, y1, color)

    canvas = _build_sidebar(canvas, objects, names)

    image_name = os.path.splitext(os.path.basename(image_path))[0] or "image"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(DEBUG_IMAGE_DIR, f"{image_name}_debug_{timestamp}.jpg")
    if not cv2.imwrite(output_path, canvas):
        logger.warning("Could not write vision debug image to %s", output_path)
        return None
    return output_path
