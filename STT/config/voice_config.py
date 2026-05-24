from app_config import get_config
from prompt.prompts import STT_WHISPER_PROMPT

_cfg = get_config()["stt"]

GROQ_API_KEY = _cfg["groq_api_key"]
GROQ_BASE_URL = _cfg["groq_base_url"]

DEVICE_ID = _cfg["device_id"]
SAMPLE_RATE = _cfg["sample_rate"]
DURATION_SEC = _cfg["duration_sec"]
SILENCE_DB = _cfg["silence_db"]
FRAME_MS = _cfg["frame_ms"]

WHISPER_MODEL = _cfg["whisper_model"]
WHISPER_LANG = _cfg["whisper_lang"]
WHISPER_PROMPT = STT_WHISPER_PROMPT

LLM_MODEL = _cfg["llm_model"]
LLM_TEMPERATURE = _cfg["llm_temperature"]
LLM_MAX_TOKENS = _cfg["llm_max_tokens"]

AUDIO_TEMP_FILE = _cfg["audio_temp_file"]
