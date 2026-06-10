import logging


LOG_SETTINGS = {
    "global_enabled": True,
    "startup": False,
    "healthcheck": True,
    "request_summary": True,
    "vision_detector_debug": False,
    "vision_object_debug": False,
    "hand_debug": True,
    "stt_debug": False,
    "response_debug": True,
    "http_client": True,
    "uvicorn_access": False,
}


def is_enabled(key: str, default: bool = False) -> bool:
    return bool(LOG_SETTINGS.get(key, default))


def print_if_enabled(key: str, *args, **kwargs) -> None:
    if is_enabled(key):
        print(*args, **kwargs)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO if is_enabled("global_enabled", True) else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not is_enabled("http_client"):
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    if not is_enabled("uvicorn_access"):
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
