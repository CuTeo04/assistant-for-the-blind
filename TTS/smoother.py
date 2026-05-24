from openai import OpenAI
import re

from prompt.prompts import (
    API_O_PHIA_TRUOC_CO_GI_SYSTEM_PROMPT,
    API_TIM_DEN_LAY_SYSTEM_PROMPT,
    API_O_PHIA_TRUOC_CO_GI_USER_TEMPLATE,
    API_TIM_DEN_LAY_USER_TEMPLATE,
    TTS_GENERAL_SMOOTHER_SYSTEM_PROMPT,
)


def init_client(api_key: str, base_url: str):
    return OpenAI(api_key=api_key, base_url=base_url)

def _extract_target_distance(distance_description: str, target_object: str) -> str:
    if not distance_description or not target_object:
        return "Khong co du lieu uu tien cho vat muc tieu."

    target_tokens = [tok for tok in re.split(r"\s+", target_object.lower().strip()) if tok]
    if not target_tokens:
        return "Khong co du lieu uu tien cho vat muc tieu."

    parts = [p.strip() for p in distance_description.split(";") if p.strip()]
    matched = []
    for part in parts:
        lower = part.lower()
        if any(tok in lower for tok in target_tokens):
            matched.append(part)

    if matched:
        return "; ".join(matched)
    return "Khong co du lieu uu tien cho vat muc tieu."


def answer_from_api(
    client,
    api_string: str,
    transcript: str,
    distance_description: str,
    raw_description: str,
    model: str,
    max_tokens: int,
    temperature: float,
):
    """Use LLM to answer based on API intent and raw scene description."""
    if not raw_description:
        return ""

    api_upper = (api_string or "").upper()
    api_prompt = ""
    target_object = ""

    if api_upper.startswith("TIM_DEN_LAY"):
        api_prompt = API_TIM_DEN_LAY_SYSTEM_PROMPT
        if ":" in api_string:
            target_object = api_string.split(":", 1)[1].strip()
    elif api_upper.startswith("O_PHIA_TRUOC_CO_GI"):
        api_prompt = API_O_PHIA_TRUOC_CO_GI_SYSTEM_PROMPT
    else:
        return raw_description

    system_prompt = f"{TTS_GENERAL_SMOOTHER_SYSTEM_PROMPT}\n\n{api_prompt}" if api_prompt else TTS_GENERAL_SMOOTHER_SYSTEM_PROMPT

    if target_object:
        transcript_text = transcript or ""
        distance_text = distance_description or "Khong co du lieu khoang cach."
        target_distance_text = _extract_target_distance(distance_text, target_object)
        user_prompt = API_TIM_DEN_LAY_USER_TEMPLATE.format(
            api_string=api_string,
            target_object=target_object,
            target_distance_description=target_distance_text,
            transcript=transcript_text,
            distance_description=distance_text,
            raw_description=raw_description,
        )
    else:
        user_prompt = API_O_PHIA_TRUOC_CO_GI_USER_TEMPLATE.format(
            api_string=api_string,
            raw_description=raw_description,
        )

    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content.strip()
