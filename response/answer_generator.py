import re
import unicodedata

from openai import OpenAI

from log_settings import print_if_enabled
from prompt.prompts import (
    API_O_PHIA_TRUOC_CO_GI_USER_TEMPLATE,
    API_TIM_DEN_LAY_USER_TEMPLATE,
    RESPONSE_O_PHIA_TRUOC_CO_GI_SYSTEM_PROMPT,
    RESPONSE_TIM_DEN_LAY_SYSTEM_PROMPT,
)


def init_client(api_key: str, base_url: str):
    return OpenAI(api_key=api_key, base_url=base_url)

TARGET_LABEL_ALIASES = {
    "ban": {"table", "dining table"},
    "ban an": {"dining table", "table"},
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
        re.match(rf"^(cai|chiec)?\s*{re.escape(term)}\b", normalized)
        for term in terms
    )


def _extract_target_distance(distance_description: str, target_object: str) -> str:
    if not distance_description or not target_object:
        return "Không có dữ liệu ưu tiên cho vật mục tiêu."

    terms = _target_terms(target_object)
    if not terms:
        return "Không có dữ liệu ưu tiên cho vật mục tiêu."

    parts = [p.strip() for p in distance_description.split(";") if p.strip()]
    matched = []
    for part in parts:
        label = part.split(":", 1)[0]
        if _contains_target(label, terms):
            matched.append(_translate_yolo_labels(part))

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


def _split_distance_clock(info_text: str) -> tuple[str, str]:
    text = (info_text or "").strip()
    clock_match = re.search(r"\bhướng\s+([0-9]{1,2})\s+giờ\b", text, flags=re.IGNORECASE)
    clock_text = f"hướng {clock_match.group(1)} giờ" if clock_match else ""
    distance_match = re.search(r"([0-9]+(?:[,.][0-9]+)?\s*(?:cm|m))", text, flags=re.IGNORECASE)
    distance_text = distance_match.group(1) if distance_match else ""
    return distance_text.strip(), clock_text.strip()


def _format_target_position_line(target_name: str, info_text: str) -> str:
    distance_text, clock_text = _split_distance_clock(info_text)
    if clock_text and distance_text:
        return f"Mục tiêu: {target_name} ở {clock_text}, cách bạn {distance_text}."
    if clock_text:
        return f"Mục tiêu: {target_name} ở {clock_text}."
    if distance_text:
        return f"Mục tiêu: {target_name} cách bạn {distance_text}."
    return f"Mục tiêu: {target_name}."


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


def _build_rule_context_for_target(
    api_string: str,
    target_object: str,
    distance_description: str,
    raw_description: str,
) -> str:
    terms = _target_terms(target_object)
    lines: list[str] = []

    for part in [p.strip() for p in (distance_description or "").split(";") if p.strip()]:
        label, sep, distance = part.partition(":")
        if sep and _contains_target(label, terms):
            target_name = _translate_object_name_with_id(label)
            lines.append(_format_target_position_line(target_name, distance))
            break

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


def select_raw_description_for_api(
    api_string: str,
    distance_description: str,
    raw_description: str,
) -> str:
    """Choose the raw description shape that matches the classified API intent."""
    api_upper = (api_string or "").upper()
    if not api_upper.startswith("TIM_DEN_LAY"):
        return _translate_yolo_labels(raw_description)

    target_object = api_string.split(":", 1)[1].strip() if ":" in (api_string or "") else ""
    if not target_object:
        return _translate_yolo_labels(raw_description)

    return _build_rule_context_for_target(
        api_string,
        target_object,
        distance_description,
        raw_description,
    )


def _remove_redundant_target_front_sentence(text: str, target_object: str) -> str:
    if not text or not target_object:
        return text

    terms = _target_terms(target_object)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= 1:
        return text

    kept = []
    for idx, sentence in enumerate(sentences):
        normalized = _normalize_text(sentence)
        is_target_sentence = _contains_target(sentence, terms)
        starts_with_target = _starts_with_target(sentence, terms)
        is_front_of_user = "phia truoc ban" in normalized or "truoc mat ban" in normalized
        is_target_relative_clause = (
            ("phia truoc" in normalized or "phia sau" in normalized)
            and "cach ban" not in normalized
        )
        has_target_distance = bool(re.search(r"\bcach ban\b", normalized)) and bool(
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
    for sentence in sentences:
        object_distance = _extract_distance_number(sentence)
        if object_distance is None or _starts_with_target(sentence, terms):
            fixed.append(sentence)
            continue

        normalized = _normalize_text(sentence)
        if "phia sau" in normalized and object_distance < target_distance:
            sentence = re.sub(r"phía sau", "phía trước", sentence, flags=re.IGNORECASE)
        elif "phia truoc" in normalized and object_distance > target_distance:
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


def _fallback_answer(
    api_string: str,
    target_object: str,
    distance_description: str,
    raw_description: str,
) -> str:
    api_upper = (api_string or "").upper()
    if api_upper.startswith("TIM_DEN_LAY"):
        target_distance = _extract_target_distance(distance_description, target_object)
        if target_distance.startswith("Không có"):
            return f"Chưa xác định được vị trí của {target_object}."
        first_match = target_distance.split(";", 1)[0].strip()
        info_text = first_match.split(":", 1)[1].strip() if ":" in first_match else ""
        distance, clock_text = _split_distance_clock(info_text)
        if distance and clock_text:
            return _finalize_answer(
                f"Cái {target_object} ở {clock_text}, cách bạn {distance}.",
                target_object,
            )
        if distance:
            return _finalize_answer(f"Cái {target_object} cách bạn {distance}.", target_object)
        if clock_text:
            return _finalize_answer(f"Cái {target_object} ở {clock_text}.", target_object)
        return f"Chưa xác định được vị trí của {target_object}."

    return _finalize_answer(raw_description, target_object)


def answer_from_api(
    client,
    api_string: str,
    transcript: str,
    distance_description: str,
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
    elif not api_upper.startswith("O_PHIA_TRUOC_CO_GI"):
        return raw_description
    else:
        system_prompt = RESPONSE_O_PHIA_TRUOC_CO_GI_SYSTEM_PROMPT

    if target_object:
        target_distance_text = _extract_target_distance(distance_description or "", target_object)
        if _looks_like_target_context(raw_description):
            target_context_text = raw_description
        else:
            target_context_text = _build_rule_context_for_target(
                api_string,
                target_object,
                distance_description,
                raw_description,
            )
        user_prompt = API_TIM_DEN_LAY_USER_TEMPLATE.format(
            api_string=api_string,
            target_object=target_object,
            target_distance_description=target_distance_text,
            raw_description=target_context_text,
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
        return _fallback_answer(api_string, target_object, distance_description, raw_description)
    content = _preserve_target_relations(content, raw_description) if target_object else content
    return _finalize_answer(content, target_object)
