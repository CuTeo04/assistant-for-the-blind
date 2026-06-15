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
BASE_LAYOUT_SIZE = 960.0


def _compute_debug_scale(image: np.ndarray) -> float:
    image_h, image_w = image.shape[:2]
    raw_scale = min(image_h, image_w) / BASE_LAYOUT_SIZE
    return float(np.clip(raw_scale, 0.55, 1.25))


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


def _nearest_neighbors(objects: list[dict], names: list[str], index: int, limit: int = 4) -> list[str]:
    source = objects[index]
    distances: list[tuple[float, str]] = []
    for other_index, other in enumerate(objects):
        if other_index == index:
            continue
        distances.append((_distance_between_objects(source, other), names[other_index]))

    distances.sort(key=lambda item: item[0])
    return [f"{name} {distance:.2f}m" for distance, name in distances[:limit]]


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    anchor_x: int,
    anchor_y: int,
    color: tuple[int, int, int],
) -> None:
    if not lines:
        return

    scale = _compute_debug_scale(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale
    thickness = max(1, int(round(1 * scale)))
    line_height = max(14, int(round(18 * scale)))
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
    scale = _compute_debug_scale(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.25 * scale
    thickness = max(2, int(round(6 * scale)))
    padding_x = max(8, int(round(24 * scale)))
    padding_y = max(6, int(round(18 * scale)))
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
    scale = _compute_debug_scale(canvas)
    body_font = cv2.FONT_HERSHEY_SIMPLEX

    top = max(24, int(round(36 * scale)))
    bottom_margin = max(12, int(round(18 * scale)))
    row_gap = max(8, int(round(12 * scale)))
    available_height = max(120, image_h - top - bottom_margin)
    block_height = max(120, int((available_height - row_gap * max(0, len(objects) - 1)) / max(1, len(objects))))
    line_thickness = max(1, int(round(3 * scale)))
    line_margin_x = max(18, int(round(18 * scale)))
    line_margin_y = max(10, int(round(14 * scale)))
    def _prepare_layout(sidebar_width: int) -> tuple[list[dict], int]:
        layouts: list[dict] = []
        needed_width = sidebar_width

        for index, obj in enumerate(objects):
            block_y1 = top + index * (block_height + row_gap)
            block_y2 = min(image_h - 12, block_y1 + block_height)
            if block_y1 >= image_h - 20:
                break

            left_x = 108

            left_lines = [
                names[index],
                f"conf: {float(obj['conf']):.2f}",
                f"camera: {float(obj['Z']):.2f}m",
            ]
            near_lines = _nearest_neighbors(objects, names, index, limit=4)

            card_font_scale = 2.05 * scale
            fit_gap = max(2, int(round(2 * scale)))
            min_font_scale = 0.75 * scale
            while card_font_scale > min_font_scale:
                (_sample_w, sample_h), sample_base = cv2.getTextSize("Ag", body_font, card_font_scale, line_thickness)
                text_block_h = sample_h + sample_base
                left_total_h = len(left_lines) * text_block_h + max(0, len(left_lines) - 1) * fit_gap
                right_total_h = len(near_lines) * text_block_h + max(0, len(near_lines) - 1) * fit_gap
                if left_total_h <= block_height - 2 * line_margin_y and right_total_h <= block_height - 2 * line_margin_y:
                    break
                card_font_scale -= 0.05 * scale

            (_sample_w, sample_h), sample_base = cv2.getTextSize("Ag", body_font, card_font_scale, line_thickness)
            text_block_h = sample_h + sample_base
            line_pitch = text_block_h + fit_gap

            left_width = max(cv2.getTextSize(line, body_font, card_font_scale, line_thickness)[0][0] for line in left_lines)
            right_width = max(cv2.getTextSize(line, body_font, card_font_scale, line_thickness)[0][0] for line in near_lines) if near_lines else 0
            one_char_gap = cv2.getTextSize("0", body_font, card_font_scale, line_thickness)[0][0]
            right_x = left_x + left_width + int(round(one_char_gap * 1.0))
            badge_font_scale = 1.15 * scale
            badge_thickness = max(1, int(round(3 * scale)))
            badge_padding_x = max(10, int(round(10 * scale)))
            badge_padding_y = max(8, int(round(8 * scale)))
            (badge_text_w, badge_text_h), badge_base = cv2.getTextSize(
                str(index + 1),
                body_font,
                badge_font_scale,
                badge_thickness,
            )
            badge_width = badge_text_w + badge_padding_x * 2
            badge_height = badge_text_h + badge_base + badge_padding_y * 2
            card_needed_width = max(
                24 + badge_width,
                right_x + right_width + line_margin_x,
            )
            needed_width = max(needed_width, int(round(card_needed_width + 14)))

            left_usable_h = max(1, block_height - 2 * line_margin_y)
            right_usable_h = max(1, block_height - 2 * line_margin_y)
            left_first_y = block_y1 + line_margin_y + sample_h
            right_first_y = block_y1 + line_margin_y + sample_h
            if len(left_lines) > 1:
                left_step = max(1.0, float(left_usable_h - text_block_h) / float(len(left_lines) - 1))
                left_y_positions = [int(round(left_first_y + i * left_step)) for i in range(len(left_lines))]
            else:
                left_y_positions = [int(round(block_y1 + block_height / 2.0 + sample_h / 2.0))]

            if len(near_lines) > 1:
                right_step = max(1.0, float(right_usable_h - text_block_h) / float(len(near_lines) - 1))
                right_y_positions = [int(round(right_first_y + i * right_step)) for i in range(len(near_lines))]
            else:
                right_y_positions = [int(round(block_y1 + block_height / 2.0 + sample_h / 2.0))]

            layouts.append(
                {
                    "index": index,
                    "obj": obj,
                    "color": BOX_COLORS[index % len(BOX_COLORS)],
                    "block_y1": block_y1,
                    "block_y2": block_y2,
                    "badge_right": 24 + badge_width,
                    "badge_bottom": block_y1 + 12 + badge_height,
                    "badge_font_scale": badge_font_scale,
                    "badge_thickness": badge_thickness,
                    "badge_padding_x": badge_padding_x,
                    "badge_padding_y": badge_padding_y,
                    "left_x": left_x,
                    "right_x": right_x,
                    "fit_font_scale": card_font_scale,
                    "line_thickness": line_thickness,
                    "left_lines": left_lines,
                    "near_lines": near_lines,
                    "left_y_positions": left_y_positions,
                    "right_y_positions": right_y_positions,
                }
            )

        return layouts, needed_width

    sidebar_width = max(
        420,
        int(round(image_w * 0.42)),
        int(round(SIDEBAR_WIDTH * 1.5 * scale)),
    )
    layouts, needed_width = _prepare_layout(sidebar_width)
    sidebar_width = max(sidebar_width, int(round(needed_width)))
    layouts, _ = _prepare_layout(sidebar_width)

    sidebar = np.full((image_h, sidebar_width, 3), 24, dtype=np.uint8)
    for layout in layouts:
        block_y1 = layout["block_y1"]
        block_y2 = layout["block_y2"]
        color = layout["color"]
        cv2.rectangle(sidebar, (14, block_y1), (sidebar_width - 14, block_y2), (45, 45, 45), -1)
        cv2.rectangle(sidebar, (14, block_y1), (sidebar_width - 14, block_y2), color, 3)
        badge_bottom = min(block_y2 - 12, layout["badge_bottom"])
        badge_right = layout["badge_right"]
        cv2.rectangle(sidebar, (24, block_y1 + 12), (badge_right, badge_bottom), color, -1)
        badge_text = str(layout["index"] + 1)
        (badge_text_w, badge_text_h), badge_base = cv2.getTextSize(
            badge_text,
            body_font,
            layout["badge_font_scale"],
            layout["badge_thickness"],
        )
        badge_text_x = 24 + layout["badge_padding_x"] + max(0, (badge_right - 24 - 2 * layout["badge_padding_x"] - badge_text_w) // 2)
        badge_text_y = block_y1 + 12 + layout["badge_padding_y"] + badge_text_h
        cv2.putText(
            sidebar,
            badge_text,
            (badge_text_x, min(block_y2 - 16, badge_text_y)),
            body_font,
            layout["badge_font_scale"],
            (20, 20, 20),
            layout["badge_thickness"],
            cv2.LINE_AA,
        )

        for line, y_pos in zip(layout["left_lines"], layout["left_y_positions"]):
            cv2.putText(
                sidebar,
                line,
                (layout["left_x"], y_pos),
                body_font,
                layout["fit_font_scale"],
                (235, 235, 235),
                layout["line_thickness"],
                cv2.LINE_AA,
            )

        for line, y_pos in zip(layout["near_lines"], layout["right_y_positions"]):
            cv2.putText(
                sidebar,
                line,
                (layout["right_x"], y_pos),
                body_font,
                layout["fit_font_scale"],
                (215, 215, 215),
                layout["line_thickness"],
                cv2.LINE_AA,
            )

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
    box_thickness = max(3, int(round(4 * _compute_debug_scale(canvas))))
    center_radius = max(4, int(round(6 * _compute_debug_scale(canvas))))

    for index, obj in enumerate(objects):
        x1 = int(round(float(obj["x1"]) * scale_x))
        y1 = int(round(float(obj["y1"]) * scale_y))
        x2 = int(round(float(obj["x2"]) * scale_x))
        y2 = int(round(float(obj["y2"]) * scale_y))
        cx = int(round(float(obj["cx"]) * scale_x))
        cy = int(round(float(obj["cy"]) * scale_y))
        color = BOX_COLORS[index % len(BOX_COLORS)]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
        cv2.circle(canvas, (cx, cy), center_radius, color, -1)
        _draw_box_badge(canvas, str(index + 1), x1, y1, color)

    canvas = _build_sidebar(canvas, objects, names)

    image_name = os.path.splitext(os.path.basename(image_path))[0] or "image"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(DEBUG_IMAGE_DIR, f"{image_name}_debug_{timestamp}.jpg")
    if not cv2.imwrite(output_path, canvas):
        logger.warning("Could not write vision debug image to %s", output_path)
        return None
    return output_path
