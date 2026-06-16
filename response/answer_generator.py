import re
import unicodedata

from openai import OpenAI

from log_settings import print_if_enabled
from prompt.prompts import (
    API_O_PHIA_TRUOC_CO_GI_USER_TEMPLATE,
    API_REGION_QUERY_USER_TEMPLATE,
    API_TIM_DEN_LAY_USER_TEMPLATE,
    RESPONSE_O_PHIA_TRUOC_CO_GI_SYSTEM_PROMPT,
    RESPONSE_REGION_QUERY_SYSTEM_PROMPT,
    RESPONSE_TIM_DEN_LAY_SYSTEM_PROMPT,
)
from vision.scene_builder import generate_description


def init_client(api_key: str, base_url: str):
    return OpenAI(api_key=api_key, base_url=base_url)

TARGET_LABEL_ALIASES = {
    "ban": {"table", "dining table"},
    "ban an": {"dining table", "table"},
    "ba lo": {"backpack"},
    "chai": {"bottle"},
    "chen": {"bowl", "cup"},
    "coc": {"cup"},
    "dien thoai": {"cell phone", "phone"},
    "ghe": {"chair"},
    "ghe sofa": {"couch", "sofa"},
    "giuong": {"bed"},
    "ke sach": {"book"},
    "laptop": {"laptop"},
    "ly": {"cup"},
    "may tinh": {"laptop", "keyboard", "mouse"},
    "remote": {"remote"},
    "sach": {"book"},
    "sofa": {"couch", "sofa"},
    "tivi": {"tv"},
    "ti vi": {"tv"},
    "tv": {"tv"},
}

YOLO_LABEL_VI = {
    "bed": "giường",
    "backpack": "ba lô",
    "book": "cuốn sách",
    "bottle": "chai",
    "bowl": "cái chén",
    "cell phone": "điện thoại",
    "chair": "cái ghế",
    "couch": "ghế sofa",
    "cup": "cái cốc",
    "dining table": "bàn ăn",
    "keyboard": "bàn phím",
    "laptop": "máy tính xách tay",
    "mouse": "chuột máy tính",
    "person": "người",
    "remote": "điều khiển",
    "sofa": "ghế sofa",
    "table": "cái bàn",
    "tv": "ti vi",
}

TARGET_STOPWORDS = {
    "cai",
    "chiec",
    "con",
    "mot",
    "vat",
    "do",
    "do vat",
    "toi",
    "muon",
    "tim",
    "lay",
    "den",
}


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"[^a-zA-Z0-9\s:;.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _target_terms(target_object: str) -> set[str]:
    normalized = _normalize_text(target_object)
    terms = {normalized} if normalized else set()

    tokens = [tok for tok in normalized.split() if tok and tok not in TARGET_STOPWORDS]
    terms.update(tokens)

    for key, aliases in TARGET_LABEL_ALIASES.items():
        key_norm = _normalize_text(key)
        if key_norm == normalized or key_norm in terms or key_norm in normalized:
            terms.update(_normalize_text(alias) for alias in aliases)

    # Add Vietnamese forms for YOLO English labels already present in terms.
    for en_label, vi_label in YOLO_LABEL_VI.items():
        en_norm = _normalize_text(en_label)
        if en_norm in terms:
            terms.add(_normalize_text(vi_label))

    return {term for term in terms if term and term not in TARGET_STOPWORDS}


def _contains_target(text: str, terms: set[str]) -> bool:
    normalized = _normalize_text(text)
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)


def _starts_with_target(text: str, terms: set[str]) -> bool:
    normalized = _normalize_text(text)
    return any(
        re.match(rf"^(muc tieu\s+\d+\s*:\s*)?(cai|chiec)?\s*{re.escape(term)}\b", normalized)
        for term in terms
    )


def _extract_target_distance(distance_description: str, target_object: str) -> str:
    matched = _collect_target_distance_parts(distance_description, target_object, limit=3)
    if matched:
        return "; ".join(matched)
    return "Không có dữ liệu ưu tiên cho vật mục tiêu."


def _extract_target_neighborhood(raw_description: str, target_object: str) -> str:
    terms = _target_terms(target_object)
    if not raw_description or not terms:
        return "Không có mô tả vật mục tiêu và các vật xung quanh."

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", raw_description) if part.strip()]
    target_indexes = [idx for idx, sentence in enumerate(sentences) if _contains_target(sentence, terms)]
    if not target_indexes:
        return "Không có mô tả vật mục tiêu và các vật xung quanh."

    included_indexes = set()
    for idx in target_indexes:
        included_indexes.update(range(max(0, idx - 1), min(len(sentences), idx + 2)))

    return " ".join(sentences[idx] for idx in sorted(included_indexes))


def _opposite_direction(direction: str) -> str:
    opposites = {
        "trước": "sau",
        "sau": "trước",
        "trái": "phải",
        "phải": "trái",
        "trên": "dưới",
        "dưới": "trên",
        "ngay cạnh": "ngay cạnh",
    }
    return opposites.get(direction, direction)


def _distance_to_cm(distance_text: str) -> float | None:
    match = re.search(r"([0-9]+(?:[,.][0-9]+)?)\s*(cm|m)\b", distance_text or "")
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    return value * 100.0 if unit == "m" else value


def _clock_hour(info_text: str) -> int | None:
    match = re.search(r"\bhướng\s+([0-9]{1,2})\s+giờ\b", info_text or "", flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    return hour if 1 <= hour <= 12 else None


def _parse_distance_entries(distance_description: str) -> list[dict]:
    entries: list[dict] = []
    for index, part in enumerate([p.strip() for p in (distance_description or "").split(";") if p.strip()]):
        label, sep, info_text = part.partition(":")
        if not sep:
            continue
        entries.append(
            {
                "index": index,
                "label": label.strip(),
                "info_text": info_text.strip(),
                "distance_cm": _distance_to_cm(info_text),
                "distance_text": _split_distance_clock(info_text)[0],
                "clock_text": _split_distance_clock(info_text)[1],
                "clock_hour": _clock_hour(info_text),
                "translated_part": _translate_yolo_labels(part),
            }
        )
    return entries


def _collect_target_distance_parts(distance_description: str, target_object: str, limit: int = 3) -> list[str]:
    if not distance_description or not target_object or limit <= 0:
        return []

    terms = _target_terms(target_object)
    if not terms:
        return []

    matched: list[tuple[float, int, str]] = []
    for entry in _parse_distance_entries(distance_description):
        if not _contains_target(entry["label"], terms):
            continue
        sort_key = float("inf") if entry["distance_cm"] is None else float(entry["distance_cm"])
        matched.append((sort_key, int(entry["index"]), str(entry["translated_part"])))

    matched.sort(key=lambda item: (item[0], item[1]))
    return [part for _distance_cm, _index, part in matched[:limit]]


def _collect_target_objects(object_brief: list[dict] | None, target_object: str, limit: int = 3) -> list[dict]:
    if not object_brief or not target_object or limit <= 0:
        return []

    terms = _target_terms(target_object)
    if not terms:
        return []

    matched = []
    for index, obj in enumerate(object_brief):
        label = str(obj.get("label", ""))
        display_label = str(obj.get("display_label", label))
        if not (_contains_target(label, terms) or _contains_target(display_label, terms)):
            continue
        distance_m = obj.get("distance_m")
        sort_key = float("inf")
        if isinstance(distance_m, (int, float)):
            sort_key = float(distance_m)
        matched.append((sort_key, index, obj))

    matched.sort(key=lambda item: (item[0], item[1]))
    return [obj for _sort_key, _index, obj in matched[:limit]]


def _split_distance_clock(info_text: str) -> tuple[str, str]:
    text = (info_text or "").strip()
    clock_match = re.search(r"\bhướng\s+([0-9]{1,2})\s+giờ\b", text, flags=re.IGNORECASE)
    clock_text = f"hướng {clock_match.group(1)} giờ" if clock_match else ""
    distance_match = re.search(r"([0-9]+(?:[,.][0-9]+)?\s*(?:cm|m))", text, flags=re.IGNORECASE)
    distance_text = distance_match.group(1) if distance_match else ""
    return distance_text.strip(), clock_text.strip()


def _format_object_distance_text(distance_m: float | None) -> str:
    if not isinstance(distance_m, (int, float)) or distance_m <= 0:
        return ""
    return f"{float(distance_m):.2f}m"


def _format_target_position_line(target_name: str, info_text: str) -> str:
    distance_text, clock_text = _split_distance_clock(info_text)
    if clock_text and distance_text:
        return f"Mục tiêu: {target_name} ở {clock_text}, cách bạn {distance_text}."
    if clock_text:
        return f"Mục tiêu: {target_name} ở {clock_text}."
    if distance_text:
        return f"Mục tiêu: {target_name} cách bạn {distance_text}."
    return f"Mục tiêu: {target_name}."


def _format_hand_distance_text(distance_cm: float | None) -> str:
    if not isinstance(distance_cm, (int, float)) or distance_cm <= 0:
        return ""
    if float(distance_cm) < 100.0:
        return f"{int(round(float(distance_cm)))}cm"
    return f"{float(distance_cm) / 100.0:.2f}m"


def _target_rank_phrase(index: int, total: int) -> str:
    if total <= 1:
        return "gần nhất"
    if index == 0:
        return "gần nhất"
    if index == total - 1:
        return "xa nhất"
    return "xa hơn"


def _direction_phrase(direction: str, immediate: bool = False) -> str:
    direction = (direction or "").lower()
    if direction in {"trái", "phải"}:
        return f"{'ngay ' if immediate else ''}bên {direction}"
    if direction in {"trước", "sau"}:
        return f"{'ngay ' if immediate else ''}phía {direction}"
    if direction in {"trên", "dưới"}:
        return f"{'ngay ' if immediate else ''}bên {direction}"
    return f"{'ngay ' if immediate else ''}{direction}".strip()


def _format_relation_line(other_name: str, direction: str, target_name: str, distance: str = "") -> str:
    distance_cm = _distance_to_cm(distance)
    if distance_cm is not None and distance_cm < 25.0:
        return f"Quan hệ: {other_name} ở {_direction_phrase(direction, immediate=True)} {target_name}."
    if distance:
        return (
            f"Quan hệ: {other_name} ở {_direction_phrase(direction)} "
            f"{target_name} khoảng {distance}."
        )
    return f"Quan hệ: {other_name} ở {_direction_phrase(direction)} {target_name}."


def _translate_yolo_labels(text: str) -> str:
    output = text or ""
    for label in sorted(YOLO_LABEL_VI, key=len, reverse=True):
        replacement = YOLO_LABEL_VI[label]
        output = re.sub(rf"\bcái\s+{re.escape(label)}\b", replacement, output, flags=re.IGNORECASE)
        output = re.sub(rf"\bchiếc\s+{re.escape(label)}\b", replacement, output, flags=re.IGNORECASE)
        output = re.sub(rf"\b{re.escape(label)}\b", replacement, output, flags=re.IGNORECASE)
    return output


def _strip_object_ids(text: str) -> str:
    output = text or ""
    for vi_label in sorted(set(YOLO_LABEL_VI.values()), key=len, reverse=True):
        output = re.sub(
            rf"\b{re.escape(vi_label)}\s+\d+\b",
            vi_label,
            output,
            flags=re.IGNORECASE,
        )
    return output


def _translate_object_name_with_id(name: str) -> str:
    text = re.sub(r"^(cái|chiếc)\s+", "", (name or "").strip(), flags=re.IGNORECASE)
    match = re.match(r"(.+?)\s+(\d+)$", text)
    if match:
        label = match.group(1).strip()
        object_id = match.group(2)
    else:
        label = text
        object_id = ""

    translated = YOLO_LABEL_VI.get(label.lower(), label)
    return f"{translated} {object_id}".strip()


def _direct_touch_answer(object_brief: list[dict] | None, target_object: str) -> str:
    target_objects = _collect_target_objects(object_brief, target_object, limit=1)
    if not target_objects:
        return ""

    touched_target = target_objects[0]
    if not bool(touched_target.get("hand_touching", False)):
        return ""

    target_name = _translate_object_name_with_id(
        str(touched_target.get("display_label", touched_target.get("label", target_object)))
    )
    return _finalize_answer(f"Bạn đã chạm vào {target_name}.", target_object)


def _build_rule_context_for_target(
    api_string: str,
    target_object: str,
    distance_description: str,
    raw_description: str,
    object_brief: list[dict] | None = None,
) -> str:
    lines: list[str] = []

    target_objects = _collect_target_objects(object_brief, target_object, limit=3)
    total_targets = len(target_objects)
    for index, obj in enumerate(target_objects):
        target_name = _translate_object_name_with_id(str(obj.get("display_label", obj.get("label", target_object))))
        rank_phrase = _target_rank_phrase(index, total_targets)
        hand_reference_valid = bool(obj.get("hand_reference_valid", False))
        hand_touching = bool(obj.get("hand_touching", False))
        hand_relation_label = str(obj.get("hand_relation_label") or "").strip()
        hand_relation_distance_text = _format_hand_distance_text(obj.get("hand_relation_distance_cm"))
        clock_text = str(obj.get("clock_label") or "").strip()
        distance_text = _format_object_distance_text(obj.get("distance_m"))
        if hand_reference_valid and hand_touching:
            position_text = f"bạn đã chạm vào {target_name}"
            if distance_text:
                position_text += f", vật ở cách bạn {distance_text}"
            lines.append(f"Mục tiêu {index + 1}: {position_text}.")
            continue

        if hand_reference_valid and hand_relation_label:
            position_text = f"{target_name} ở {hand_relation_label} mu bàn tay"
            if hand_relation_distance_text and hand_relation_label != "chạm":
                position_text += f" khoảng {hand_relation_distance_text}"
            if distance_text:
                position_text += f", cách bạn {distance_text}"
            lines.append(f"Mục tiêu {index + 1}: {position_text}.")
            continue

        position_text = f"{target_name} ở {rank_phrase}"
        if clock_text:
            position_text += f", {clock_text}"
        if distance_text:
            position_text += f", cách bạn {distance_text}"
        lines.append(f"Mục tiêu {index + 1}: {position_text}.")

    if not lines:
        target_parts = _collect_target_distance_parts(distance_description, target_object, limit=3)
        total_targets = len(target_parts)
        for index, part in enumerate(target_parts):
            label, sep, distance = part.partition(":")
            if not sep:
                continue
            target_name = _translate_object_name_with_id(label)
            rank_phrase = _target_rank_phrase(index, total_targets)
            distance_text, clock_text = _split_distance_clock(distance)
            position_text = f"{target_name} ở {rank_phrase}"
            if clock_text:
                position_text += f", {clock_text}"
            if distance_text:
                position_text += f", cách bạn {distance_text}"
            lines.append(f"Mục tiêu {index + 1}: {position_text}.")

    terms = _target_terms(target_object)

    for sentence in re.split(r"(?<=[.!?])\s+", raw_description or ""):
        sentence = sentence.strip()
        if not sentence or not _contains_target(sentence, terms):
            continue

        relation_match = re.search(
            r"(Phía|Bên)\s+(trước|sau|trái|phải)\s+cái\s+(.+?)\s+"
            r"([0-9]+(?:[,.][0-9]+)?\s*(?:cm|m))\s+là\s+cái\s+(.+?)\s*\.",
            sentence,
            flags=re.IGNORECASE,
        )
        if relation_match:
            direction = relation_match.group(2).lower()
            anchor_raw = relation_match.group(3)
            distance = relation_match.group(4)
            other_raw = relation_match.group(5)
            anchor_name = _translate_object_name_with_id(anchor_raw)
            other_name = _translate_object_name_with_id(other_raw)
            if _contains_target(anchor_raw, terms):
                lines.append(_format_relation_line(other_name, direction, anchor_name, distance))
            elif _contains_target(other_raw, terms):
                opposite = _opposite_direction(direction)
                lines.append(_format_relation_line(anchor_name, opposite, other_name, distance))
            continue

        adjacent_match = re.search(
            r"Ngay\s+cạnh\s+cái\s+(.+?)\s+"
            r"([0-9]+(?:[,.][0-9]+)?\s*(?:cm|m))\s+là\s+cái\s+(.+?)\s*\.",
            sentence,
            flags=re.IGNORECASE,
        )
        if adjacent_match:
            anchor_raw = adjacent_match.group(1)
            distance = adjacent_match.group(2)
            other_raw = adjacent_match.group(3)
            anchor_name = _translate_object_name_with_id(anchor_raw)
            other_name = _translate_object_name_with_id(other_raw)
            if _contains_target(anchor_raw, terms):
                lines.append(_format_relation_line(other_name, "ngay cạnh", anchor_name, distance))
            elif _contains_target(other_raw, terms):
                lines.append(_format_relation_line(anchor_name, "ngay cạnh", other_name, distance))
            continue

        vertical_match = re.search(
            r"Bên\s+(trên|dưới)\s+cái\s+(.+?)\s+là\s+cái\s+(.+?)\s*\.",
            sentence,
            flags=re.IGNORECASE,
        )
        if vertical_match:
            direction = vertical_match.group(1).lower()
            anchor_raw = vertical_match.group(2)
            other_raw = vertical_match.group(3)
            anchor_name = _translate_object_name_with_id(anchor_raw)
            other_name = _translate_object_name_with_id(other_raw)
            if _contains_target(anchor_raw, terms):
                lines.append(_format_relation_line(other_name, direction, anchor_name))
            elif _contains_target(other_raw, terms):
                opposite = _opposite_direction(direction)
                lines.append(_format_relation_line(anchor_name, opposite, other_name))

    if not lines:
        return _extract_target_neighborhood(raw_description, target_object)
    return "\n".join(lines)


def _looks_like_target_context(raw_description: str) -> bool:
    return bool(re.search(r"(^|\n)(Mục tiêu|Quan hệ):", raw_description or ""))


REGION_QUERY_INTENT_MAP = {
    "O_BEN_TRAI_CO_GI": {
        "region_name": "bên trái",
        "clock_hours": {8, 9, 10},
    },
    "O_BEN_PHAI_CO_GI": {
        "region_name": "bên phải",
        "clock_hours": {2, 3, 4},
    },
    "O_PHIA_DUOI_CO_GI": {
        "region_name": "phía dưới",
        "clock_hours": {5, 6, 7},
    },
}


def _is_region_query_intent(api_string: str) -> bool:
    api_upper = (api_string or "").upper()
    return any(api_upper.startswith(intent) for intent in REGION_QUERY_INTENT_MAP)


def _region_query_config(api_string: str) -> dict | None:
    api_upper = (api_string or "").upper()
    for intent, config in REGION_QUERY_INTENT_MAP.items():
        if api_upper.startswith(intent):
            return config
    return None


def _build_rule_context_for_region(
    api_string: str,
    distance_description: str,
    raw_description: str,
    object_brief: list[dict] | None = None,
) -> str:
    config = _region_query_config(api_string)
    if not config:
        return _translate_yolo_labels(raw_description)

    region_name = str(config["region_name"])
    region_hours = set(config["clock_hours"])
    selected: list[dict] = []
    if object_brief:
        region_objects = []
        for index, obj in enumerate(object_brief):
            clock_hour = obj.get("clock_hour")
            if clock_hour is None or clock_hour not in region_hours:
                continue
            distance_m = obj.get("distance_m")
            sort_key = float("inf")
            if isinstance(distance_m, (int, float)):
                sort_key = float(distance_m)
            region_objects.append((sort_key, index, obj))

        region_objects.sort(key=lambda item: (item[0], item[1]))
        selected = [obj for _sort_key, _index, obj in region_objects[:3]]

    lines = [f"Khu vực ưu tiên: {region_name}."]
    total_selected = len(selected)
    for index, obj in enumerate(selected):
        rank_phrase = _target_rank_phrase(index, total_selected)
        target_name = _translate_object_name_with_id(str(obj.get("display_label", obj.get("label", "vật thể"))))
        clock_text = str(obj.get("clock_label") or "").strip()
        distance_text = _format_object_distance_text(obj.get("distance_m"))
        position_text = f"{target_name} ở {rank_phrase}"
        if clock_text:
            position_text += f", {clock_text}"
        if distance_text:
            position_text += f", cách bạn {distance_text}"
        lines.append(f"Vật {index + 1}: {position_text}.")

    if len(lines) == 1:
        entries = []
        for entry in _parse_distance_entries(distance_description):
            clock_hour = entry["clock_hour"]
            if clock_hour is None or clock_hour not in region_hours:
                continue
            sort_key = float("inf") if entry["distance_cm"] is None else float(entry["distance_cm"])
            entries.append((sort_key, int(entry["index"]), entry))

        entries.sort(key=lambda item: (item[0], item[1]))
        fallback_selected = [entry for _sort_key, _index, entry in entries[:3]]
        total_selected = len(fallback_selected)
        for index, entry in enumerate(fallback_selected):
            rank_phrase = _target_rank_phrase(index, total_selected)
            target_name = _translate_object_name_with_id(str(entry["label"]))
            position_text = f"{target_name} ở {rank_phrase}"
            if entry["clock_text"]:
                position_text += f", {entry['clock_text']}"
            if entry["distance_text"]:
                position_text += f", cách bạn {entry['distance_text']}"
            lines.append(f"Vật {index + 1}: {position_text}.")

    if len(lines) == 1:
        lines.append(f"Chưa có vật nào được xác định rõ ở khu vực {region_name}.")
        if raw_description:
            lines.append(f"Bối cảnh chung: {_translate_yolo_labels(raw_description)}")

    return "\n".join(lines)


def _build_general_scene_description_from_scene_facts(scene_graph: dict | None) -> str:
    if not scene_graph:
        return "Không phát hiện được vật thể hợp lệ để mô tả."

    description = generate_description(scene_graph)
    if int(scene_graph.get("object_count", 0) or 0) == 1:
        return f"{description} Không thấy vật nào khác xung quanh."
    return description


def build_raw_description_from_scene_facts(
    api_string: str,
    scene_graph: dict | None,
    distance_description: str,
    object_brief: list[dict] | None = None,
    hand_info: dict | None = None,
) -> str:
    """Build an intent-specific raw description from Task 1 + Task 2 structured facts."""
    del hand_info
    general_scene_description = _build_general_scene_description_from_scene_facts(scene_graph)
    api_upper = (api_string or "").upper()
    if _is_region_query_intent(api_string):
        return _build_rule_context_for_region(
            api_string,
            distance_description,
            general_scene_description,
            object_brief,
        )
    if not api_upper.startswith("TIM_DEN_LAY"):
        return _translate_yolo_labels(general_scene_description)

    target_object = api_string.split(":", 1)[1].strip() if ":" in (api_string or "") else ""
    if not target_object:
        return _translate_yolo_labels(general_scene_description)

    touch_answer = _direct_touch_answer(object_brief, target_object)
    if touch_answer:
        return touch_answer

    return _build_rule_context_for_target(
        api_string,
        target_object,
        distance_description,
        general_scene_description,
        object_brief,
    )


def _remove_redundant_target_front_sentence(text: str, target_object: str) -> str:
    if not text or not target_object:
        return text

    terms = _target_terms(target_object)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= 1:
        return text

    kept = []
    front_phrase = _normalize_text("phía trước")
    back_phrase = _normalize_text("phía sau")
    in_front_of_user_phrase = _normalize_text("phía trước bạn")
    in_front_of_face_phrase = _normalize_text("trước mặt bạn")
    for idx, sentence in enumerate(sentences):
        normalized = _normalize_text(sentence)
        is_target_sentence = _contains_target(sentence, terms)
        starts_with_target = _starts_with_target(sentence, terms)
        is_front_of_user = in_front_of_user_phrase in normalized or in_front_of_face_phrase in normalized
        is_target_relative_clause = (
            (front_phrase in normalized or back_phrase in normalized)
            and _normalize_text("cách bạn") not in normalized
        )
        has_target_distance = bool(re.search(rf"\b{re.escape(_normalize_text('cách bạn'))}\b", normalized)) and bool(
            re.search(r"\d", sentence)
        )
        if idx > 0 and starts_with_target and is_target_relative_clause and not has_target_distance:
            continue
        if idx > 0 and is_target_sentence and is_front_of_user and not has_target_distance:
            continue
        kept.append(sentence)

    return " ".join(kept)


def _extract_distance_number(sentence: str) -> float | None:
    match = re.search(r"\bcách bạn\s+([0-9]+(?:[,.][0-9]+)?)", sentence, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _fix_relative_depth_by_distance(text: str, target_object: str) -> str:
    if not text or not target_object:
        return text

    terms = _target_terms(target_object)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    target_distance = None
    for sentence in sentences:
        if _starts_with_target(sentence, terms):
            target_distance = _extract_distance_number(sentence)
            if target_distance is not None:
                break

    if target_distance is None:
        return text

    fixed = []
    back_phrase = _normalize_text("phía sau")
    front_phrase = _normalize_text("phía trước")
    for sentence in sentences:
        object_distance = _extract_distance_number(sentence)
        if object_distance is None or _starts_with_target(sentence, terms):
            fixed.append(sentence)
            continue

        normalized = _normalize_text(sentence)
        if back_phrase in normalized and object_distance < target_distance:
            sentence = re.sub(r"phía sau", "phía trước", sentence, flags=re.IGNORECASE)
        elif front_phrase in normalized and object_distance > target_distance:
            sentence = re.sub(r"phía trước", "phía sau", sentence, flags=re.IGNORECASE)
        fixed.append(sentence)

    return " ".join(fixed)


def _capitalize_sentences(text: str) -> str:
    def repl(match):
        return match.group(1) + match.group(2).upper()

    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return re.sub(r"(^|[.!?]\s+)([a-zà-ỹ])", repl, text)


def _finalize_answer(text: str, target_object: str = "") -> str:
    output = _remove_redundant_target_front_sentence(text, target_object)
    output = _fix_relative_depth_by_distance(output, target_object)
    output = _translate_yolo_labels(output)
    output = _strip_object_ids(output)
    output = re.sub(r"\s+", " ", output).strip()
    return _capitalize_sentences(output)


def _relation_subject(relation_sentence: str) -> str:
    match = re.match(r"\s*Quan hệ:\s+(.+?)\s+ở\s+", relation_sentence or "", flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _relation_to_answer_sentence(relation_sentence: str) -> str:
    text = re.sub(r"^\s*Quan hệ:\s*", "", relation_sentence or "", flags=re.IGNORECASE).strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _preserve_target_relations(answer: str, raw_description: str) -> str:
    """Append any target relation lines that the LLM omitted."""
    if not _looks_like_target_context(raw_description):
        return answer

    output = answer or ""
    normalized_output = _normalize_text(output)
    missing_sentences = []
    for relation in re.findall(r"Quan hệ:\s*[^.]+(?:\.)?", raw_description or "", flags=re.IGNORECASE):
        subject = _relation_subject(relation)
        if not subject:
            continue
        subject_without_id = re.sub(r"\s+\d+\b", "", subject).strip()
        if _normalize_text(subject_without_id) in normalized_output:
            continue
        missing_sentences.append(_relation_to_answer_sentence(relation))

    if not missing_sentences:
        return output
    return " ".join([output.strip(), *missing_sentences]).strip()


def _extract_target_distance_from_raw(raw_description: str) -> str:
    for line in (raw_description or "").splitlines():
        line = line.strip()
        if not line.startswith("Mục tiêu"):
            continue
        _, _, detail = line.partition(":")
        detail = detail.strip()
        if detail:
            return detail
    return "Xem chi tiết trong mô tả ngữ cảnh."


def _fallback_answer(
    api_string: str,
    target_object: str,
    raw_description: str,
) -> str:
    api_upper = (api_string or "").upper()
    if api_upper.startswith("TIM_DEN_LAY"):
        if _looks_like_target_context(raw_description):
            first_line = next(
                (line.strip() for line in (raw_description or "").splitlines() if line.strip().startswith("Mục tiêu")),
                "",
            )
            if first_line:
                _, _, first_line_text = first_line.partition(":")
                if first_line_text.strip():
                    return _finalize_answer(first_line_text.strip(), target_object)
        if raw_description:
            return _finalize_answer(raw_description, target_object)
        return f"Chưa xác định được vị trí của {target_object}."

    return _finalize_answer(raw_description, target_object)


def answer_from_api(
    client,
    api_string: str,
    transcript: str,
    raw_description: str,
    model: str,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool = False,
    reasoning_effort: str | None = None,
    log_cache_usage: bool = False,
):
    """Use LLM to answer based on API intent and raw scene description."""
    if not raw_description:
        return ""

    api_upper = (api_string or "").upper()
    target_object = ""

    if api_upper.startswith("TIM_DEN_LAY"):
        if ":" in api_string:
            target_object = api_string.split(":", 1)[1].strip()
        system_prompt = RESPONSE_TIM_DEN_LAY_SYSTEM_PROMPT
    elif _is_region_query_intent(api_string):
        system_prompt = RESPONSE_REGION_QUERY_SYSTEM_PROMPT
    elif not api_upper.startswith("O_PHIA_TRUOC_CO_GI"):
        return raw_description
    else:
        system_prompt = RESPONSE_O_PHIA_TRUOC_CO_GI_SYSTEM_PROMPT

    if target_object:
        target_distance_text = _extract_target_distance_from_raw(raw_description)
        user_prompt = API_TIM_DEN_LAY_USER_TEMPLATE.format(
            api_string=api_string,
            target_object=target_object,
            target_distance_description=target_distance_text,
            raw_description=raw_description,
        )
    elif _is_region_query_intent(api_string):
        region_config = _region_query_config(api_string)
        region_name = str(region_config["region_name"]) if region_config else "khu vực được hỏi"
        user_prompt = API_REGION_QUERY_USER_TEMPLATE.format(
            api_string=api_string,
            region_name=region_name,
            raw_description=raw_description,
        )
    else:
        user_prompt = API_O_PHIA_TRUOC_CO_GI_USER_TEMPLATE.format(
            api_string=api_string,
            raw_description=raw_description,
        )

    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if thinking_enabled:
        request["extra_body"] = {"thinking": {"type": "enabled"}}
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
    else:
        request["extra_body"] = {"thinking": {"type": "disabled"}}
        request["temperature"] = temperature

    res = client.chat.completions.create(**request)
    if log_cache_usage and getattr(res, "usage", None):
        cache_hit = getattr(res.usage, "prompt_cache_hit_tokens", None)
        cache_miss = getattr(res.usage, "prompt_cache_miss_tokens", None)
        if cache_hit is not None or cache_miss is not None:
            print_if_enabled("response_debug", f"DeepSeek cache usage | hit={cache_hit or 0} | miss={cache_miss or 0}")

    choice = res.choices[0]
    message = choice.message
    content = (getattr(message, "content", None) or "").strip()
    if log_cache_usage:
        reasoning = getattr(message, "reasoning_content", None) or ""
        print_if_enabled(
            "response_debug",
            "DeepSeek response usage | "
            f"finish_reason={getattr(choice, 'finish_reason', None)} | "
            f"content_chars={len(content)} | reasoning_chars={len(reasoning)}"
        )
    if not content:
        return _fallback_answer(api_string, target_object, raw_description)
    content = _preserve_target_relations(content, raw_description) if target_object else content
    return _finalize_answer(content, target_object)
