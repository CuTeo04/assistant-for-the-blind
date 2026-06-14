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
        return {}

    with SIZE_REFERENCE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    classes = payload.get("classes", {})
    if not isinstance(classes, dict):
        return {}
    return {_normalize_label(label): rule for label, rule in classes.items() if isinstance(rule, dict)}


def _matches_range(real_w_cm: float, real_h_cm: float, rule: dict) -> bool:
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


def filter_object_data_by_size(object_data: list[tuple], size_reference: dict[str, dict] | None = None):
    reference = load_size_reference() if size_reference is None else {
        _normalize_label(label): rule
        for label, rule in size_reference.items()
    }

    kept: list[tuple] = []
    rejected: list[dict] = []

    for item in object_data:
        det, label, depth_m, real_w_cm, real_h_cm = item
        normalized = _normalize_label(label)
        rule = reference.get(normalized)
        if rule is None:
            kept.append(item)
            continue

        if _matches_range(real_w_cm, real_h_cm, rule):
            kept.append(item)
            continue

        rejected.append(
            {
                "label": normalized,
                "depth_m": float(depth_m),
                "real_w_cm": float(real_w_cm),
                "real_h_cm": float(real_h_cm),
                "width_cm": _as_range(rule, "width_cm"),
                "height_cm": _as_range(rule, "height_cm"),
            }
        )

    return kept, rejected
