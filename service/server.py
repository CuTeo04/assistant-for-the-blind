import asyncio
import logging
import os
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app_config import get_config
from service.vision_pipeline import describe_image_with_models, init_models
from TTS.config import tts_config as tcfg
from TTS.smoother import answer_from_api, init_client as init_tts_client
from STT.command_processor import process_voice_command_api

_cfg = get_config()["server"]
LOG_ENABLED = bool(_cfg.get("log_enabled", True))

logging.basicConfig(
    level=logging.INFO if LOG_ENABLED else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("voice_server")

if not LOG_ENABLED:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(title="Voice Command Server")

yolo = None
da_model = None
hands_full = None
hands_crop = None


@app.on_event("startup")
def load_models():
    global yolo, da_model, hands_full, hands_crop
    logger.info("Loading vision models...")
    yolo, da_model, hands_full, hands_crop = init_models()
    logger.info("Vision models ready")


@app.get("/health")
def health_check():
    logger.info("Health check")
    return {"status": "ok"}


@app.post("/process")
async def process_audio(audio: UploadFile = File(...), image: UploadFile = File(...)):
    req_start = time.perf_counter()
    if not audio.filename or not image.filename:
        logger.warning("Missing filename in upload")
        return JSONResponse(status_code=400, content={"error": "Missing filename"})

    audio_suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    image_suffix = os.path.splitext(image.filename)[-1] or ".jpg"
    audio_path = None
    image_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_audio:
            audio_path = tmp_audio.name
            audio_content = await audio.read()
            tmp_audio.write(audio_content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp_image:
            image_path = tmp_image.name
            image_content = await image.read()
            tmp_image.write(image_content)

        def run_api():
            start = time.perf_counter()
            result = process_voice_command_api(audio_path, include_latency=True)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return result, latency_ms

        def run_vision():
            start = time.perf_counter()
            result = describe_image_with_models(
                image_path, yolo, da_model, hands_full, hands_crop
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return result, latency_ms

        (api_result, api_latency), (vision_result, vision_latency) = await asyncio.gather(
            asyncio.to_thread(run_api),
            asyncio.to_thread(run_vision),
        )
        api_string, transcript, api_steps = api_result
        description, timings, distance_desc = vision_result
        if not api_string:
            logger.warning("No audio provided after saving file")
            return JSONResponse(status_code=400, content={"error": "No audio provided"})

        whisper_ms = 0.0
        llm_ms = 0.0
        if api_steps:
            whisper_ms = api_steps.get("whisper_s", 0.0) * 1000.0
            llm_ms = api_steps.get("llm_s", 0.0) * 1000.0

        logger.info(
            "Task1 STT+LLM | api=%s | transcript=%s | total=%.1f ms | whisper=%.1f ms | llm=%.1f ms",
            api_string,
            transcript,
            api_latency,
            whisper_ms,
            llm_ms,
        )

        logger.info(
            "Task2 Vision | total=%.1f ms | steps(ms): load+resize=%.1f yolo=%.1f hand=%.1f focal=%.1f depth=%.1f calib+obj=%.1f scene=%.1f | raw=%s",
            timings.get("total_ms", 0.0),
            timings.get("load_resize_ms", 0.0),
            timings.get("yolo_ms", 0.0),
            timings.get("hand_ms", 0.0),
            timings.get("focal_ms", 0.0),
            timings.get("depth_ms", 0.0),
            timings.get("calib_objects_ms", 0.0),
            timings.get("scene_ms", 0.0),
            description,
        )
        logger.info("Distance matrix (camera -> objects): %s", distance_desc)

        tts_latency = 0.0
        try:
            tts_client = init_tts_client(tcfg.GROQ_API_KEY, tcfg.GROQ_BASE_URL)
            tts_start = time.perf_counter()
            smooth_text = answer_from_api(
                tts_client,
                api_string,
                transcript,
                distance_desc,
                description,
                model=tcfg.LLM_MODEL,
                max_tokens=tcfg.MAX_TOKENS,
                temperature=tcfg.TEMPERATURE,
            )
            tts_latency = (time.perf_counter() - tts_start) * 1000.0
        except Exception as exc:
            logger.warning("LLM answer failed: %s", exc)
            smooth_text = description

        response = {"api": api_string, "text": smooth_text}
        total_ms = (time.perf_counter() - req_start) * 1000.0
        logger.info("Task3 TTS | total=%.1f ms", tts_latency)
        logger.info("Request total latency: %.1f ms", total_ms)
        return response

    except Exception as exc:
        logger.exception("Server error while processing audio: %s", exc)
        total_ms = (time.perf_counter() - req_start) * 1000.0
        logger.info("Request total latency (error): %.1f ms", total_ms)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_cfg["host"], port=int(_cfg["port"]))
