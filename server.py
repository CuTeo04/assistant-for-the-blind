from app_config import get_config
from log_settings import is_enabled
from service.server import app


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()["server"]
    ssl_certfile = cfg.get("ssl_certfile") or None
    ssl_keyfile = cfg.get("ssl_keyfile") or None
    uvicorn.run(
        app,
        host=cfg["host"],
        port=int(cfg["port"]),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        access_log=is_enabled("uvicorn_access", True),
    )
