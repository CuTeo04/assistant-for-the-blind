from app_config import get_config

_cfg = get_config()["response"]

API_KEY = _cfg["api_key"]
BASE_URL = _cfg["base_url"]

LLM_MODEL = _cfg["llm_model"]
MAX_TOKENS = _cfg["max_tokens"]
TEMPERATURE = _cfg["temperature"]
THINKING_ENABLED = bool(_cfg.get("thinking_enabled", False))
REASONING_EFFORT = _cfg.get("reasoning_effort")
LOG_CACHE_USAGE = bool(_cfg.get("log_cache_usage", False))
