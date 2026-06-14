from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


SIZE_REFERENCE_PATH = Path(__file__).with_name("object_size_reference.json")


def _normalize_label(label: str) -> str:
    return str(label or "").strip().lower()


def _as_range(rule: dict, key: str) -> tuple[float, float] | None:
    value = rule.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        return None
    if low > high:
        low, high = high, low
    return low, high


@lru_cache(maxsize=1)
def load_size_reference() -> dict[str, dict]:
    if not SIZE_REFERENCE_PATH.exists():
        return {"classes": {}, "similar_shape_groups": []}

    with SIZE_REFERENCE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    classes = payload.get("classes", {})
    groups = payload.get("similar_shape_groups", [])
    if not isinstance(classes, dict):
        classes = {}
    if not isinstance(groups, list):
        groups = []
    return {
        "classes": {
            _normalize_label(label): rule
            for label, rule in classes.items()
            if isinstance(rule, dict)
        },
        "similar_shape_groups": [
            [_normalize_label(label) for label in group if _normalize_label(label)]
            for group in groups
            if isinstance(group, list)
        ],
    }


def _matches_range(real_w_cm: float, real_h_cm: float, rule: dict) -> bool:
    short_side_range = _as_range(rule, "short_side_cm")
    long_side_range = _as_range(rule, "long_side_cm")
    if short_side_range is not None and long_side_range is not None:
        short_side = min(abs(float(real_w_cm)), abs(float(real_h_cm)))
        long_side = max(abs(float(real_w_cm)), abs(float(real_h_cm)))
        return short_side_range[0] <= short_side <= short_side_range[1] and long_side_range[0] <= long_side <= long_side_range[1]

    width_range = _as_range(rule, "width_cm")
    height_range = _as_range(rule, "height_cm")
    if width_range is None or height_range is None:
        return True

    width = abs(float(real_w_cm))
    height = abs(float(real_h_cm))

    def in_range(value: float, rng: tuple[float, float]) -> bool:
        return rng[0] <= value <= rng[1]

    direct = in_range(width, width_range) and in_range(height, height_range)
    swapped = in_range(height, width_range) and in_range(width, height_range)
    return direct or swapped


def _range_center(rng: tuple[float, float]) -> float:
    return (rng[0] + rng[1]) / 2.0


def _candidate_match_score(real_w_cm: float, real_h_cm: float, rule: dict) -> float:
    short_side_range = _as_range(rule, "short_side_cm")
    long_side_range = _as_range(rule, "long_side_cm")
    width = abs(float(real_w_cm))
    height = abs(float(real_h_cm))

    if short_side_range is not None and long_side_range is not None:
        short_side = min(width, height)
        long_side = max(width, height)
        short_span = max(1.0, short_side_range[1] - short_side_range[0])
        long_span = max(1.0, long_side_range[1] - long_side_range[0])
        return abs(short_side - _range_center(short_side_range)) / short_span + abs(long_side - _range_center(long_side_range)) / long_span

    width_range = _as_range(rule, "width_cm")
    height_range = _as_range(rule, "height_cm")
    if width_range is None or height_range is None:
        return float("inf")

    width_span = max(1.0, width_range[1] - width_range[0])
    height_span = max(1.0, height_range[1] - height_range[0])
    direct_score = abs(width - _range_center(width_range)) / width_span + abs(height - _range_center(height_range)) / height_span
    swapped_score = abs(height - _range_center(width_range)) / width_span + abs(width - _range_center(height_range)) / height_span
    return min(direct_score, swapped_score)


def _normalize_reference_payload(size_reference) -> dict:
    if size_reference is None:
        return load_size_reference()
    classes = size_reference.get("classes", size_reference)
    groups = size_reference.get("similar_shape_groups", [])
    return {
        "classes": {
            _normalize_label(label): rule
            for label, rule in classes.items()
        },
        "similar_shape_groups": [
            [_normalize_label(label) for label in group if _normalize_label(label)]
            for group in groups
            if isinstance(group, list)
        ],
    }


def _find_similar_group(label: str, groups: list[list[str]]) -> list[str] | None:
    for group in groups:
        if label in group:
            return group
    return None


def filter_object_data_by_size(object_data: list[tuple], size_reference=None):
    reference_payload = _normalize_reference_payload(size_reference)
    reference = reference_payload["classes"]
    groups = reference_payload["similar_shape_groups"]

    normalized_object_data: list[tuple] = []
    relabeled: list[dict] = []

    for item in object_data:
        det, label, depth_m, real_w_cm, real_h_cm = item
        normalized = _normalize_label(label)
        group = _find_similar_group(normalized, groups)
        if group is None:
            normalized_object_data.append(item)
            continue

        candidates = []
        for candidate_label in group:
            candidate_rule = reference.get(candidate_label)
            if candidate_rule is None:
                continue
            if _matches_range(real_w_cm, real_h_cm, candidate_rule):
                candidates.append(candidate_label)

        target_label = normalized
        if normalized in candidates:
            target_label = normalized
        elif candidates:
            target_label = min(
                candidates,
                key=lambda candidate_label: _candidate_match_score(
                    real_w_cm,
                    real_h_cm,
                    reference[candidate_label],
                ),
            )

        if target_label != normalized:
            relabeled.append(
                {
                    "from_label": normalized,
                    "to_label": target_label,
                    "depth_m": float(depth_m),
                    "real_w_cm": float(real_w_cm),
                    "real_h_cm": float(real_h_cm),
                }
            )
        normalized_object_data.append((det, target_label, depth_m, real_w_cm, real_h_cm))

    return normalized_object_data, relabeled
