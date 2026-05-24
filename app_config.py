import json
import os
from pathlib import Path

from dotenv import load_dotenv

_CONFIG = None


def _resolve_env(value):
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            return os.environ.get(key, "")
        return value
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def get_config():
    global _CONFIG
    if _CONFIG is None:
        load_dotenv()
        config_path = Path(__file__).resolve().with_name("server_config.json")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _CONFIG = _resolve_env(data)
    return _CONFIG
