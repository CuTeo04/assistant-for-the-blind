from app_config import get_config

_cfg = get_config()["tts"]

GROQ_API_KEY = _cfg["groq_api_key"]
GROQ_BASE_URL = _cfg["groq_base_url"]

LLM_MODEL = _cfg["llm_model"]
MAX_TOKENS = _cfg["max_tokens"]
TEMPERATURE = _cfg["temperature"]
