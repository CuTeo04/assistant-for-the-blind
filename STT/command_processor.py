import time

import re
import unicodedata

from log_settings import print_if_enabled
from openai import OpenAI

from .config import voice_config as cfg
from prompt.prompts import STT_CLASSIFY_SYSTEM_PROMPT

# Keep local name for minimal changes in call sites.
LLM_SYSTEM_PROMPT = STT_CLASSIFY_SYSTEM_PROMPT

DIRECT_CONFIG_COMMANDS = (
    "thiet lap camera",
    "thiet lap cau hinh",
)


def init_client(api_key: str, base_url: str):
    return OpenAI(api_key=api_key, base_url=base_url)


def transcribe_audio(client, audio_path: str, model: str, language: str, prompt: str):
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    start_time = time.time()
    transcript = client.audio.transcriptions.create(
        file=(audio_path, audio_data),
        model=model,
        language=language,
        prompt=prompt,
        response_format="text",
    )
    latency = time.time() - start_time
    return transcript.strip(), latency


def classify_command(
    client,
    text: str,
    model: str,
    temperature: float = 0,
    max_tokens: int | None = None,
):
    start_time = time.time()
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    return res.choices[0].message.content.strip(), latency


def parse_api_line(llm_text: str) -> str:
    """Parse the API-only format from the LLM output.

    Expected:
      API: ...
    """
    for line in (llm_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("API:"):
            return line.split(":", 1)[1].strip()
    return ""


def normalize_api_output(raw_text: str) -> str:
    if not raw_text:
        return "KHONG_XAC_DINH"

    # Preserve the original object text (case/diacritics) like testspeed.py.
    match = re.search(r"TIM_DEN_LAY\s*:\s*([^\n\r]+)", raw_text.strip(), flags=re.IGNORECASE)
    if match:
        obj = match.group(1).strip()
        return f"TIM_DEN_LAY: {obj}" if obj else "TIM_DEN_LAY"

    text_up = raw_text.strip().upper()
    if "THIET_LAP_CAU_HINH" in text_up:
        return "THIET_LAP_CAU_HINH"
    if "O_PHIA_TRUOC_CO_GI" in text_up:
        return "O_PHIA_TRUOC_CO_GI"

    return "KHONG_XAC_DINH"


def _normalize_transcript_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", (text or "").strip())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.lower().replace("đ", "d")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _match_direct_intent(transcript: str) -> str | None:
    normalized = _normalize_transcript_text(transcript)
    for command in DIRECT_CONFIG_COMMANDS:
        if command in normalized:
            return "THIET_LAP_CAU_HINH"
    return None


def execute_command(api_string: str):
    response = format_command_response(api_string)
    print_if_enabled("stt_debug", response)


def format_command_response(api_string: str) -> str:
    if "TIM_DEN_LAY" in api_string:
        do_vat = api_string.split(":")[-1].strip()
        if do_vat:
            return f"Đang tìm và lấy {do_vat}."
        return "Đang tìm và lấy vật thể."
    if "THIET_LAP_CAU_HINH" in api_string:
        return "Mở bảng thiết lập cấu hình."
    if "O_PHIA_TRUOC_CO_GI" in api_string:
        return "Bật camera quét phía trước."
    return "Lệnh không xác định."


def process_voice_command(audio_path: str, config=cfg):
    if not audio_path:
        return

    try:
        client = init_client(config.GROQ_API_KEY, config.GROQ_BASE_URL)

        print_if_enabled("stt_debug", "Đang gửi audio lên Whisper...")
        text, whisper_latency = transcribe_audio(
            client,
            audio_path,
            config.WHISPER_MODEL,
            config.WHISPER_LANG,
            config.WHISPER_PROMPT,
        )

        if not text or len(text) < 3:
            print_if_enabled("stt_debug", "Không nghe rõ hoặc chỉ có tiếng ồn.")
            return

        print_if_enabled("stt_debug", f"Nghe duoc: \"{text}\"")
        print_if_enabled("stt_debug", f"Whisper latency: {whisper_latency:.3f} giay\n")

        print_if_enabled("stt_debug", "Đang phân loại lệnh bằng LLM...")
        llm_text, llm_latency = classify_command(
            client,
            text,
            config.LLM_MODEL,
            temperature=getattr(config, "LLM_TEMPERATURE", 0),
            max_tokens=getattr(config, "LLM_MAX_TOKENS", None),
        )
        api_string = parse_api_line(llm_text) or llm_text

        api_string = normalize_api_output(api_string)
        if api_string and "KHONG_LIEN_QUAN" in api_string.upper():
            api_string = "KHONG_XAC_DINH"

        print_if_enabled("stt_debug", f"LENH API: {api_string}")
        print_if_enabled("stt_debug", f"LLM latency: {llm_latency:.3f} giay")

        execute_command(api_string)

    except Exception as exc:
        print_if_enabled("stt_debug", f"Lỗi xử lý API: {exc}")


def process_voice_command_return(audio_path: str, config=cfg):
    api_string, transcript = process_voice_command_api(audio_path, config=config)
    if not api_string:
        return None
    if api_string == "KHONG_XAC_DINH":
        return "Không nghe rõ hoặc chỉ có tiếng ồn." if not transcript else "Lệnh không xác định."
    return format_command_response(api_string)


def process_voice_command_api(audio_path: str, config=cfg, include_latency: bool = False):
    if not audio_path:
        return (None, None, None) if include_latency else (None, None)

    try:
        client = init_client(config.GROQ_API_KEY, config.GROQ_BASE_URL)

        text, whisper_latency = transcribe_audio(
            client,
            audio_path,
            config.WHISPER_MODEL,
            config.WHISPER_LANG,
            config.WHISPER_PROMPT,
        )

        if not text or len(text) < 3:
            if include_latency:
                return "KHONG_XAC_DINH", text, {"whisper_s": whisper_latency, "llm_s": 0.0}
            return "KHONG_XAC_DINH", text

        direct_api = _match_direct_intent(text)
        if direct_api:
            if include_latency:
                return direct_api, text, {"whisper_s": whisper_latency, "llm_s": 0.0}
            return direct_api, text

        llm_text, llm_latency = classify_command(
            client,
            text,
            config.LLM_MODEL,
            temperature=getattr(config, "LLM_TEMPERATURE", 0),
            max_tokens=getattr(config, "LLM_MAX_TOKENS", None),
        )
        api_string = parse_api_line(llm_text) or llm_text

        api_string = normalize_api_output(api_string)
        if api_string and "KHONG_LIEN_QUAN" in api_string.upper():
            api_string = "KHONG_XAC_DINH"
        if include_latency:
            steps = {"whisper_s": whisper_latency, "llm_s": llm_latency}
            return api_string, text, steps
        return api_string, text

    except Exception:
        return (None, None, None) if include_latency else (None, None)
