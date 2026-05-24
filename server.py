from app_config import get_config
from service.server import app


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()["server"]
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]))
