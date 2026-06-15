from __future__ import annotations

import logging
import time

import numpy as np

from log_settings import is_enabled
from vision.config import vision_config as cfg

from .backends import ensure_detector_backend

logger = logging.getLogger("voice_server.vision")
SEMANTIC_MERGE_LABEL_GROUPS = (
    frozenset({"electric kettle", "kettle", "cooking pot"}),
    frozenset({"cabinet", "shelf"}),
    frozenset({"backpack", "shopping bag"}),
    frozenset({"desk", "table", "dining table"}),
)
SEMANTIC_MERGE_LABEL_PRIORITY = {
    "cabinet": 1,
    "shelf": 0,
    "desk": 2,
    "table": 1,
    "dining table": 0,
}
SEMANTIC_MERGE_IOU_THRESHOLD = 0.65
SEMANTIC_MERGE_CONTAINMENT_THRESHOLD = 0.80


def boxes_to_detection_records(boxes: np.ndarray, label_lookup: dict[int, str], source: str) -> list[dict]:
    if boxes.size == 0:
        return []

    records = []
    for det in boxes:
        class_id = int(det[5])
        label = label_lookup.get(class_id)
        if not label:
            continue
        records.append(
            {
                "x1": int(det[0]),
                "y1": int(det[1]),
                "x2": int(det[2]),
                "y2": int(det[3]),
                "conf": float(det[4]),
                "label": label,
                "class_id": class_id,
                "source": source,
            }
        )
    return records


def filter_allowed_classes(boxes: np.ndarray, yolo) -> np.ndarray:
    if boxes.size == 0 or not cfg.ALLOWED_CLASSES:
        return boxes

    backend = ensure_detector_backend(yolo, source_name="yolo")
    label_lookup = backend.label_lookup()
    keep_indexes = []
    for idx, det in enumerate(boxes):
        label = label_lookup.get(int(det[5]), "")
        if label in cfg.ALLOWED_CLASSES:
            keep_indexes.append(idx)

    if not keep_indexes:
        return np.empty((0, 6), dtype=np.float32)
    return boxes[keep_indexes]


def clip_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes

    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h - 1)
    return boxes


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _centered_start(center: float, tile_size: int, max_length: int) -> int:
    if max_length <= tile_size:
        return 0
    start = int(round(center - tile_size / 2.0))
    return max(0, min(start, max_length - tile_size))


def build_center_tile_meta(
    orig_h: int,
    orig_w: int,
    tile_size: int,
    overlap: float,
    center_tile_count: int,
) -> list[tuple[int, int, int, int]]:
    if orig_h <= tile_size and orig_w <= tile_size:
        return [(0, 0, orig_w, orig_h)]

    center_tile_count = 4 if int(center_tile_count) >= 4 else 2
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    cx = orig_w / 2.0
    cy = orig_h / 2.0

    tiles: list[tuple[int, int, int, int]] = []
    if center_tile_count == 2:
        if orig_h >= orig_w:
            centers = [(cx, cy - stride / 2.0), (cx, cy + stride / 2.0)]
        else:
            centers = [(cx - stride / 2.0, cy), (cx + stride / 2.0, cy)]
    else:
        half_stride = stride / 2.0
        centers = [
            (cx - half_stride, cy - half_stride),
            (cx + half_stride, cy - half_stride),
            (cx - half_stride, cy + half_stride),
            (cx + half_stride, cy + half_stride),
        ]

    seen: set[tuple[int, int, int, int]] = set()
    for tile_cx, tile_cy in centers:
        x1 = _centered_start(tile_cx, tile_size, orig_w)
        y1 = _centered_start(tile_cy, tile_size, orig_h)
        x2 = min(x1 + tile_size, orig_w)
        y2 = min(y1 + tile_size, orig_h)
        tile = (x1, y1, x2, y2)
        if tile not in seen:
            seen.add(tile)
            tiles.append(tile)

    if not tiles:
        return [(0, 0, min(tile_size, orig_w), min(tile_size, orig_h))]
    return tiles


def _build_full_and_tile_inputs(
    img: np.ndarray,
    *,
    include_full: bool,
    tiled_enabled: bool,
    tile_size: int,
    overlap: float,
    tile_mode: str,
    center_tile_count: int,
) -> list[dict]:
    img_h, img_w = img.shape[:2]
    batch_items: list[dict] = []
    if include_full:
        batch_items.append({"kind": "full", "image": img})

    if not tiled_enabled:
        return batch_items

    if tile_mode == "center_batch":
        tile_meta = build_center_tile_meta(
            img_h,
            img_w,
            tile_size,
            overlap,
            center_tile_count,
        )
    else:
        tile_meta = []
        for y1 in tile_starts(img_h, tile_size, overlap):
            for x1 in tile_starts(img_w, tile_size, overlap):
                x2 = min(x1 + tile_size, img_w)
                y2 = min(y1 + tile_size, img_h)
                tile_meta.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in tile_meta:
        batch_items.append(
            {
                "kind": "tile",
                "image": img[y1:y2, x1:x2],
                "meta": (x1, y1, x2, y2),
            }
        )
    return batch_items


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    smaller_area = min(box_area(box_a), box_area(box_b))
    if smaller_area <= 0:
        return 0.0
    return intersection_area(box_a, box_b) / smaller_area


def center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax = (float(box_a[0]) + float(box_a[2])) / 2.0
    ay = (float(box_a[1]) + float(box_a[3])) / 2.0
    bx = (float(box_b[0]) + float(box_b[2])) / 2.0
    by = (float(box_b[1]) + float(box_b[3])) / 2.0

    union_w = max(float(box_a[2]), float(box_b[2])) - min(float(box_a[0]), float(box_b[0]))
    union_h = max(float(box_a[3]), float(box_b[3])) - min(float(box_a[1]), float(box_b[1]))
    union_diag = max(1.0, float(np.hypot(union_w, union_h)))
    return float(np.hypot(ax - bx, ay - by) / union_diag)


def label_for_det(det: np.ndarray, yolo) -> str:
    backend = ensure_detector_backend(yolo, source_name="yolo")
    return backend.label_lookup().get(int(det[5]), "")


def redetect_config_from_runtime(classes=None, conf_thr=None, enabled=None) -> dict:
    return {
        "enabled": cfg.REDETECT_ENABLED if enabled is None else bool(enabled),
        "classes": cfg.REDETECT_CLASSES if classes is None else classes,
        "containment_thr": cfg.REDETECT_CONTAINMENT_THR,
        "iou_thr": cfg.REDETECT_IOU_THR,
        "center_dist_ratio": cfg.REDETECT_CENTER_DIST_RATIO,
        "padding_ratio": cfg.REDETECT_PADDING_RATIO,
        "imgsz": cfg.REDETECT_IMGSZ,
        "conf_thr": cfg.REDETECT_CONF_THR if conf_thr is None else conf_thr,
        "max_candidates": cfg.REDETECT_MAX_CANDIDATES,
        "fallback": cfg.REDETECT_FALLBACK,
        "min_box_size": cfg.REDETECT_MIN_BOX_SIZE,
        "min_crop_size": cfg.REDETECT_MIN_CROP_SIZE,
    }


def redetect_conf_for_label(label: str, config: dict) -> float:
    conf_thr = config.get("conf_thr", {})
    if isinstance(conf_thr, dict):
        return float(conf_thr.get(label, conf_thr.get("default", 0.25)))
    return float(conf_thr)


def find_redetect_candidates(detections: np.ndarray, yolo, config: dict) -> list[dict]:
    if detections.size == 0 or not config.get("enabled", False):
        return []

    classes = {str(label).strip().lower() for label in config.get("classes", [])}
    min_box_size = float(config.get("min_box_size", 16))
    candidates = []
    for idx_a in range(len(detections)):
        det_a = detections[idx_a]
        label_a = label_for_det(det_a, yolo)
        if classes and label_a not in classes:
            continue
        if min(det_a[2] - det_a[0], det_a[3] - det_a[1]) < min_box_size:
            continue

        for idx_b in range(idx_a + 1, len(detections)):
            det_b = detections[idx_b]
            if int(det_a[5]) != int(det_b[5]):
                continue
            if min(det_b[2] - det_b[0], det_b[3] - det_b[1]) < min_box_size:
                continue

            containment = containment_ratio(det_a[:4], det_b[:4])
            iou_value = float(iou(det_a[:4], det_b[None, :4])[0])
            center_ratio = center_distance_ratio(det_a[:4], det_b[:4])
            suspicious_overlap = (
                containment >= float(config.get("containment_thr", 0.75))
                or iou_value >= float(config.get("iou_thr", 0.45))
            )
            if not suspicious_overlap:
                continue
            if center_ratio > float(config.get("center_dist_ratio", 0.35)):
                continue

            candidates.append(
                {
                    "id_a": idx_a,
                    "id_b": idx_b,
                    "label": label_a,
                    "class_id": int(det_a[5]),
                    "containment": containment,
                    "iou": iou_value,
                    "center_dist_ratio": center_ratio,
                    "score": max(containment, iou_value) * max(float(det_a[4]), float(det_b[4])),
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[: int(config.get("max_candidates", 5))]


def make_union_crop_box(
    box_a: np.ndarray,
    box_b: np.ndarray,
    image_shape,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    x1 = min(float(box_a[0]), float(box_b[0]))
    y1 = min(float(box_a[1]), float(box_b[1]))
    x2 = max(float(box_a[2]), float(box_b[2]))
    y2 = max(float(box_a[3]), float(box_b[3]))

    pad_x = (x2 - x1) * float(padding_ratio)
    pad_y = (y2 - y1) * float(padding_ratio)
    crop_x1 = int(max(0, np.floor(x1 - pad_x)))
    crop_y1 = int(max(0, np.floor(y1 - pad_y)))
    crop_x2 = int(min(img_w, np.ceil(x2 + pad_x)))
    crop_y2 = int(min(img_h, np.ceil(y2 + pad_y)))
    return crop_x1, crop_y1, crop_x2, crop_y2


def record_to_box(record: dict) -> np.ndarray:
    return np.asarray(
        [record["x1"], record["y1"], record["x2"], record["y2"]],
        dtype=np.float32,
    )


def _semantic_group_for_label(label: str) -> str:
    normalized = str(label).strip().lower()
    for index, group in enumerate(SEMANTIC_MERGE_LABEL_GROUPS):
        if normalized in group:
            return f"semantic:{index}"
    return normalized


def _semantic_label_priority(label: str) -> int:
    return SEMANTIC_MERGE_LABEL_PRIORITY.get(str(label).strip().lower(), 0)


def _should_merge_records(current: dict, candidate: dict, iou_threshold: float) -> bool:
    current_box = record_to_box(current)
    candidate_box = record_to_box(candidate)
    overlap = float(iou(current_box, candidate_box[None, :])[0])
    if current["label"] == candidate["label"]:
        return overlap > iou_threshold

    current_group = _semantic_group_for_label(current["label"])
    candidate_group = _semantic_group_for_label(candidate["label"])
    if current_group != candidate_group or not current_group.startswith("semantic:"):
        return False

    containment = containment_ratio(current_box, candidate_box)
    return overlap > SEMANTIC_MERGE_IOU_THRESHOLD or containment > SEMANTIC_MERGE_CONTAINMENT_THRESHOLD


def merge_detection_records(
    yolo_records: list[dict],
    open_vocab_records: list[dict],
    iou_threshold: float,
) -> list[dict]:
    if not open_vocab_records:
        return sorted(yolo_records, key=lambda item: item["conf"], reverse=True)
    if not yolo_records:
        return sorted(open_vocab_records, key=lambda item: item["conf"], reverse=True)

    grouped: dict[str, list[dict]] = {}
    for record in [*yolo_records, *open_vocab_records]:
        grouped.setdefault(_semantic_group_for_label(record["label"]), []).append(record)

    merged: list[dict] = []
    for group_key, records in grouped.items():
        records = sorted(
            records,
            key=lambda item: (
                _semantic_label_priority(item["label"]),
                item["conf"],
                1 if item.get("source") == "yolo" else 0,
            ),
            reverse=True,
        )
        kept: list[dict] = []
        while records:
            current = records.pop(0)
            kept.append(current)
            filtered = []
            for candidate in records:
                if not _should_merge_records(current, candidate, iou_threshold):
                    filtered.append(candidate)
                elif is_enabled("vision_detector_debug", True) and current["label"] != candidate["label"]:
                    logger.info(
                        "Semantic detector merge: keep=%s[%.2f,%s] drop=%s[%.2f,%s] group=%s",
                        current["label"],
                        float(current["conf"]),
                        current.get("source", "unknown"),
                        candidate["label"],
                        float(candidate["conf"]),
                        candidate.get("source", "unknown"),
                        group_key,
                    )
            records = filtered
        merged.extend(kept)

    return sorted(merged, key=lambda item: item["conf"], reverse=True)


def build_open_vocab_prompts(yolo) -> list[str]:
    if not cfg.OPEN_VOCAB_EXTRA_CLASSES:
        return []

    backend = ensure_detector_backend(yolo, source_name="yolo")
    yolo_labels = set(backend.label_lookup().values())
    blocked_labels = set(cfg.ALLOWED_CLASSES) | yolo_labels
    prompts = []
    seen = set()
    for label in cfg.OPEN_VOCAB_EXTRA_CLASSES:
        if label in blocked_labels or label in seen:
            continue
        prompts.append(label)
        seen.add(label)
    return prompts


def run_redetect_on_crop(model, crop, target_class: str, config: dict) -> np.ndarray:
    backend = ensure_detector_backend(model, source_name="yolo")
    class_id = backend.class_id_for_label(target_class)
    conf_thr = redetect_conf_for_label(target_class, config)
    classes = [class_id] if class_id is not None else None
    boxes = backend.predict_boxes(
        crop,
        conf_thr,
        imgsz=int(config.get("imgsz", 960)),
        classes=classes,
    )
    if boxes.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if class_id is not None:
        boxes = boxes[boxes[:, 5].astype(int) == class_id]
    return boxes.astype(np.float32)


def map_crop_boxes_to_pipeline_coords(
    crop_detections: np.ndarray,
    crop_box: tuple[int, int, int, int],
    resize_scale=(1.0, 1.0),
) -> np.ndarray:
    if crop_detections.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    scale_x, scale_y = resize_scale
    mapped = crop_detections.astype(np.float32).copy()
    x1, y1, _x2, _y2 = crop_box
    mapped[:, [0, 2]] = mapped[:, [0, 2]] / float(scale_x) + x1
    mapped[:, [1, 3]] = mapped[:, [1, 3]] / float(scale_y) + y1
    return mapped


def overlap_with_union(det: np.ndarray, union_box: np.ndarray) -> float:
    det_area = box_area(det[:4])
    if det_area <= 0:
        return 0.0
    return intersection_area(det[:4], union_box) / det_area


def resolve_redetect_candidate(
    original_a: np.ndarray,
    original_b: np.ndarray,
    redetect_results: np.ndarray,
    config: dict,
    candidate: dict | None = None,
) -> dict:
    fallback = str(config.get("fallback", "keep_both"))
    label = str(candidate.get("label", "")) if candidate else ""
    union_box = np.array(
        [
            min(original_a[0], original_b[0]),
            min(original_a[1], original_b[1]),
            max(original_a[2], original_b[2]),
            max(original_a[3], original_b[3]),
        ],
        dtype=np.float32,
    )

    if redetect_results.size == 0:
        return {"action": fallback, "metadata": {"redetect_reason": "nested_candidate"}}

    conf_thr = redetect_conf_for_label(label, config) if label else 0.0
    good = [
        det
        for det in redetect_results
        if float(det[4]) >= conf_thr and overlap_with_union(det, union_box) >= 0.60
    ]
    if len(good) == 1:
        merged = np.asarray(good[0], dtype=np.float32).copy()
        merged[4] = max(float(merged[4]), float(original_a[4]), float(original_b[4]))
        metadata = {
            "source": "redetect",
            "redetect_from": [candidate["id_a"], candidate["id_b"]] if candidate else [],
            "redetect_reason": "nested_candidate",
        }
        return {"action": "merge", "box": merged, "metadata": metadata}

    if len(good) >= 2:
        return {"action": "keep_both", "metadata": {"redetect_reason": "nested_candidate"}}

    return {"action": fallback, "metadata": {"redetect_reason": "nested_candidate"}}


def apply_redetect_resolution(
    detections: np.ndarray,
    candidates: list[dict],
    resolutions: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    if detections.size == 0 or not candidates:
        return detections, []

    removed: set[int] = set()
    additions = []
    metadata = []
    for candidate, resolution in zip(candidates, resolutions):
        idx_a = int(candidate["id_a"])
        idx_b = int(candidate["id_b"])
        if idx_a in removed or idx_b in removed:
            continue

        action = resolution.get("action", "keep_both")
        if action == "merge" and "box" in resolution:
            removed.update({idx_a, idx_b})
            additions.append(np.asarray(resolution["box"], dtype=np.float32))
            metadata.append(resolution.get("metadata", {}))
        elif action == "keep_higher_conf":
            drop_idx = idx_a if float(detections[idx_a, 4]) < float(detections[idx_b, 4]) else idx_b
            removed.add(drop_idx)
            metadata.append(resolution.get("metadata", {}))
        else:
            metadata.append(resolution.get("metadata", {}))

    kept = [det for idx, det in enumerate(detections) if idx not in removed]
    if additions:
        kept.extend(additions)
    if not kept:
        return np.empty((0, 6), dtype=np.float32), metadata
    return np.asarray(kept, dtype=np.float32), metadata


def redetect_for_nested_candidates(
    yolo,
    img,
    detections: np.ndarray,
    config: dict | None = None,
    log_prefix: str = "YOLO",
) -> np.ndarray:
    backend = ensure_detector_backend(yolo, source_name="yolo")
    config = config or redetect_config_from_runtime()
    if detections.size == 0 or not config.get("enabled", False):
        return detections

    start = time.perf_counter()
    candidates = find_redetect_candidates(detections, backend, config)
    if is_enabled("vision_detector_debug", True):
        logger.info("%s redetect nested candidates: count=%d", log_prefix, len(candidates))
    if not candidates:
        return detections

    resolutions = []
    min_crop_size = int(config.get("min_crop_size", 32))
    for candidate in candidates:
        det_a = detections[candidate["id_a"]]
        det_b = detections[candidate["id_b"]]
        crop_box = make_union_crop_box(det_a[:4], det_b[:4], img.shape, config.get("padding_ratio", 0.15))
        x1, y1, x2, y2 = crop_box
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < min_crop_size or crop_h < min_crop_size:
            resolution = {
                "action": config.get("fallback", "keep_both"),
                "metadata": {"redetect_reason": "nested_candidate"},
            }
            redetect_results = np.empty((0, 6), dtype=np.float32)
        else:
            crop = img[y1:y2, x1:x2]
            crop_results = run_redetect_on_crop(backend, crop, candidate["label"], config)
            redetect_results = map_crop_boxes_to_pipeline_coords(crop_results, crop_box)
            resolution = resolve_redetect_candidate(det_a, det_b, redetect_results, config, candidate)

        if is_enabled("vision_detector_debug", True):
            logger.info(
                "%s redetect candidate | class=%s ids=[%d,%d] crop=%s results=%d decision=%s",
                log_prefix,
                candidate["label"],
                candidate["id_a"],
                candidate["id_b"],
                crop_box,
                len(redetect_results),
                resolution.get("action", "keep_both"),
            )
        resolutions.append(resolution)

    final_detections, metadata = apply_redetect_resolution(detections, candidates, resolutions)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if is_enabled("vision_detector_debug", True):
        logger.info(
            "%s redetect complete: before=%d after=%d metadata=%s latency=%.1f ms",
            log_prefix,
            len(detections),
            len(final_detections),
            metadata,
            elapsed_ms,
        )
    return final_detections


def class_aware_nms(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 6)

    kept = []
    for class_id in np.unique(boxes[:, 5].astype(int)):
        class_boxes = boxes[boxes[:, 5].astype(int) == class_id]
        order = np.argsort(class_boxes[:, 4])[::-1]
        class_boxes = class_boxes[order]

        while len(class_boxes) > 0:
            current = class_boxes[0]
            kept.append(current)
            if len(class_boxes) == 1:
                break

            overlaps = iou(current[:4], class_boxes[1:, :4])
            class_boxes = class_boxes[1:][overlaps <= iou_threshold]

    if not kept:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def detect_tiled_on_original(
    yolo,
    orig_img,
    target_w: int,
    target_h: int,
    conf_threshold: float,
    imgsz=None,
    device=None,
    filter_fn=None,
    tile_size: int | None = None,
    overlap: float | None = None,
    tile_mode: str | None = None,
    center_tile_count: int | None = None,
) -> np.ndarray:
    backend = ensure_detector_backend(yolo, source_name="yolo")
    orig_h, orig_w = orig_img.shape[:2]
    tile_size = max(1, int(cfg.TILE_SIZE if tile_size is None else tile_size))
    overlap = min(max(float(cfg.TILE_OVERLAP if overlap is None else overlap), 0.0), 0.95)
    tile_mode = str(cfg.TILE_MODE if tile_mode is None else tile_mode).strip().lower()
    center_tile_count = int(cfg.CENTER_TILE_COUNT if center_tile_count is None else center_tile_count)
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)
    batch_size = max(1, int(backend.tile_batch_size()))

    boxes = []
    tile_meta: list[tuple[int, int, int, int]] = []
    tile_images: list[np.ndarray] = []
    if tile_mode == "center_batch":
        tile_meta = build_center_tile_meta(
            orig_h,
            orig_w,
            tile_size,
            overlap,
            center_tile_count,
        )
        tile_images = [orig_img[y1:y2, x1:x2] for x1, y1, x2, y2 in tile_meta]
    else:
        for y1 in tile_starts(orig_h, tile_size, overlap):
            for x1 in tile_starts(orig_w, tile_size, overlap):
                x2 = min(x1 + tile_size, orig_w)
                y2 = min(y1 + tile_size, orig_h)
                tile_meta.append((x1, y1, x2, y2))
                tile_images.append(orig_img[y1:y2, x1:x2])

    for start in range(0, len(tile_images), batch_size):
        batch_images = tile_images[start:start + batch_size]
        batch_meta = tile_meta[start:start + batch_size]
        batch_boxes_list = backend.predict_boxes_batch(
            batch_images,
            conf_threshold,
            imgsz=imgsz,
            device=device,
        )
        for (x1, y1, _x2, _y2), tile_boxes in zip(batch_meta, batch_boxes_list):
            if filter_fn is not None:
                tile_boxes = filter_fn(tile_boxes, backend)
            if tile_boxes.size == 0:
                continue

            tile_boxes[:, [0, 2]] = (tile_boxes[:, [0, 2]] + x1) * scale_x
            tile_boxes[:, [1, 3]] = (tile_boxes[:, [1, 3]] + y1) * scale_y
            boxes.append(tile_boxes)

    if not boxes:
        return np.empty((0, 6), dtype=np.float32)

    tiled_boxes = np.vstack(boxes).astype(np.float32)
    return clip_boxes(tiled_boxes, target_w, target_h)


def run_batched_full_tile_pipeline(
    model,
    img,
    conf_threshold: float,
    *,
    imgsz=None,
    device=None,
    filter_fn=None,
    redetect_config: dict | None = None,
    log_prefix: str = "YOLO",
    return_timings: bool = False,
    tiled_enabled: bool | None = None,
    tile_imgsz=None,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
    tile_mode: str | None = None,
    center_tile_count: int | None = None,
    max_batch: int | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    backend = ensure_detector_backend(model, source_name=log_prefix.lower())
    tiled_enabled = cfg.TILED_ENABLED if tiled_enabled is None else bool(tiled_enabled)
    tile_size = max(1, int(cfg.TILE_SIZE if tile_size is None else tile_size))
    tile_overlap = min(max(float(cfg.TILE_OVERLAP if tile_overlap is None else tile_overlap), 0.0), 0.95)
    tile_mode = str(cfg.TILE_MODE if tile_mode is None else tile_mode).strip().lower()
    center_tile_count = int(cfg.CENTER_TILE_COUNT if center_tile_count is None else center_tile_count)
    target_imgsz = imgsz if tile_imgsz is None else tile_imgsz
    backend_batch = max(1, int(backend.tile_batch_size()))
    batch_limit = backend_batch if max_batch is None else max(1, min(backend_batch, int(max_batch)))
    timings: dict[str, float] = {
        "full_infer_ms": 0.0,
        "full_filter_ms": 0.0,
        "tiled_ms": 0.0,
        "nms_ms": 0.0,
        "redetect_ms": 0.0,
        "pipeline_ms": 0.0,
        "combined_infer_ms": 0.0,
        "combined_filter_ms": 0.0,
        "combined_post_ms": 0.0,
        "combined_batch_size": float(batch_limit),
        "combined_image_count": 0.0,
        "combined_tile_count": 0.0,
    }
    pipeline_start = time.perf_counter()
    batch_items = _build_full_and_tile_inputs(
        img,
        include_full=True,
        tiled_enabled=tiled_enabled,
        tile_size=tile_size,
        overlap=tile_overlap,
        tile_mode=tile_mode,
        center_tile_count=center_tile_count,
    )
    timings["combined_image_count"] = float(len(batch_items))
    timings["combined_tile_count"] = float(max(0, len(batch_items) - 1))

    img_h, img_w = img.shape[:2]
    full_boxes = np.empty((0, 6), dtype=np.float32)
    tiled_boxes_list: list[np.ndarray] = []
    full_total = 0
    tile_total = 0

    infer_start = time.perf_counter()
    for start in range(0, len(batch_items), batch_limit):
        chunk = batch_items[start:start + batch_limit]
        chunk_images = [item["image"] for item in chunk]
        chunk_boxes_list = backend.predict_boxes_batch(
            chunk_images,
            conf_threshold,
            imgsz=target_imgsz,
            device=device,
        )
        for item, raw_boxes in zip(chunk, chunk_boxes_list):
            if item["kind"] == "full":
                full_total = len(raw_boxes)
            else:
                tile_total += len(raw_boxes)

            if filter_fn is not None:
                raw_boxes = filter_fn(raw_boxes, backend)
            boxes = raw_boxes.astype(np.float32)

            if item["kind"] == "full":
                full_boxes = boxes
                continue

            if boxes.size == 0:
                continue
            x1, y1, _x2, _y2 = item["meta"]
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            tiled_boxes_list.append(clip_boxes(boxes, img_w, img_h))
    timings["combined_infer_ms"] = (time.perf_counter() - infer_start) * 1000.0
    timings["full_infer_ms"] = timings["combined_infer_ms"]
    timings["tiled_ms"] = timings["combined_infer_ms"]

    filter_start = time.perf_counter()
    tiled_boxes = (
        np.vstack(tiled_boxes_list).astype(np.float32)
        if tiled_boxes_list
        else np.empty((0, 6), dtype=np.float32)
    )
    timings["combined_filter_ms"] = (time.perf_counter() - filter_start) * 1000.0
    timings["full_filter_ms"] = timings["combined_filter_ms"]

    combined = tiled_boxes if full_boxes.size == 0 else np.vstack([full_boxes, tiled_boxes]).astype(np.float32)
    if combined.size == 0:
        step_start = time.perf_counter()
        final_boxes = redetect_for_nested_candidates(
            backend,
            img,
            combined,
            config=redetect_config,
            log_prefix=log_prefix,
        )
        timings["redetect_ms"] = (time.perf_counter() - step_start) * 1000.0
        timings["combined_post_ms"] = timings["redetect_ms"]
        timings["pipeline_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
        if is_enabled("vision_detector_debug", True):
            logger.info(
                "%s combined full+tile detections: full=%d tiled=%d merged=%d batch_images=%d",
                log_prefix,
                full_total,
                tile_total,
                len(final_boxes),
                len(batch_items),
            )
        if return_timings:
            return final_boxes, timings
        return final_boxes

    step_start = time.perf_counter()
    merged_boxes = class_aware_nms(combined, cfg.NMS_IOU_THRESHOLD)
    timings["nms_ms"] = (time.perf_counter() - step_start) * 1000.0
    step_start = time.perf_counter()
    final_boxes = redetect_for_nested_candidates(
        backend,
        img,
        merged_boxes,
        config=redetect_config,
        log_prefix=log_prefix,
    )
    timings["redetect_ms"] = (time.perf_counter() - step_start) * 1000.0
    timings["combined_post_ms"] = timings["nms_ms"] + timings["redetect_ms"]
    timings["pipeline_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
    if is_enabled("vision_detector_debug", True):
        logger.info(
            "%s combined full+tile detections: full=%d tiled=%d merged=%d batch_images=%d",
            log_prefix,
            full_total,
            tile_total,
            len(final_boxes),
            len(batch_items),
        )
    if return_timings:
        return final_boxes, timings
    return final_boxes


def run_detector_pipeline(
    model,
    img,
    orig_img,
    conf_threshold: float,
    imgsz=None,
    device=None,
    filter_fn=None,
    redetect_config: dict | None = None,
    log_prefix: str = "YOLO",
    return_timings: bool = False,
    tiled_enabled: bool | None = None,
    full_image_enabled: bool = True,
    tile_imgsz=None,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
    tile_mode: str | None = None,
    center_tile_count: int | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    backend = ensure_detector_backend(model, source_name=log_prefix.lower())
    tiled_enabled = cfg.TILED_ENABLED if tiled_enabled is None else bool(tiled_enabled)
    timings: dict[str, float] = {
        "full_infer_ms": 0.0,
        "full_filter_ms": 0.0,
        "tiled_ms": 0.0,
        "nms_ms": 0.0,
        "redetect_ms": 0.0,
        "pipeline_ms": 0.0,
    }
    pipeline_start = time.perf_counter()

    full_boxes = np.empty((0, 6), dtype=np.float32)
    full_total = 0
    if full_image_enabled:
        step_start = time.perf_counter()
        full_boxes = backend.predict_boxes(img, conf_threshold, imgsz=imgsz, device=device)
        timings["full_infer_ms"] = (time.perf_counter() - step_start) * 1000.0
        full_total = len(full_boxes)
        step_start = time.perf_counter()
        if filter_fn is not None:
            full_boxes = filter_fn(full_boxes, backend)
        full_boxes = full_boxes.astype(np.float32)
        timings["full_filter_ms"] = (time.perf_counter() - step_start) * 1000.0

    if not tiled_enabled or img is None:
        step_start = time.perf_counter()
        final_boxes = redetect_for_nested_candidates(
            backend,
            img,
            full_boxes,
            config=redetect_config,
            log_prefix=log_prefix,
        )
        timings["redetect_ms"] = (time.perf_counter() - step_start) * 1000.0
        timings["pipeline_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
        if is_enabled("vision_detector_debug", True):
            logger.info(
                "%s full-image detections: full=%d filtered=%d tiled=0 merged=%d",
                log_prefix,
                full_total,
                len(full_boxes),
                len(final_boxes),
            )
        if return_timings:
            return final_boxes, timings
        return final_boxes

    img_h, img_w = img.shape[:2]
    step_start = time.perf_counter()
    tiled_boxes = detect_tiled_on_original(
        backend,
        img,
        img_w,
        img_h,
        conf_threshold,
        imgsz=imgsz if tile_imgsz is None else tile_imgsz,
        device=device,
        filter_fn=filter_fn,
        tile_size=tile_size,
        overlap=tile_overlap,
        tile_mode=tile_mode,
        center_tile_count=center_tile_count,
    )
    timings["tiled_ms"] = (time.perf_counter() - step_start) * 1000.0
    if tiled_boxes.size == 0:
        step_start = time.perf_counter()
        final_boxes = redetect_for_nested_candidates(
            backend,
            img,
            full_boxes,
            config=redetect_config,
            log_prefix=log_prefix,
        )
        timings["redetect_ms"] = (time.perf_counter() - step_start) * 1000.0
        timings["pipeline_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
        if is_enabled("vision_detector_debug", True):
            logger.info(
                "%s full+tiled detections: full=%d filtered_full=%d tiled=0 merged=%d",
                log_prefix,
                full_total,
                len(full_boxes),
                len(final_boxes),
            )
        if return_timings:
            return final_boxes, timings
        return final_boxes

    combined = tiled_boxes if full_boxes.size == 0 else np.vstack([full_boxes, tiled_boxes]).astype(np.float32)
    step_start = time.perf_counter()
    merged_boxes = class_aware_nms(combined, cfg.NMS_IOU_THRESHOLD)
    timings["nms_ms"] = (time.perf_counter() - step_start) * 1000.0
    step_start = time.perf_counter()
    final_boxes = redetect_for_nested_candidates(
        backend,
        img,
        merged_boxes,
        config=redetect_config,
        log_prefix=log_prefix,
    )
    timings["redetect_ms"] = (time.perf_counter() - step_start) * 1000.0
    timings["pipeline_ms"] = (time.perf_counter() - pipeline_start) * 1000.0
    if is_enabled("vision_detector_debug", True):
        logger.info(
            "%s full+tiled detections: full=%d filtered_full=%d tiled=%d merged=%d",
            log_prefix,
            full_total,
            len(full_boxes),
            len(tiled_boxes),
            len(final_boxes),
        )
    if return_timings:
        return final_boxes, timings
    return final_boxes
