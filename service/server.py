import asyncio
import logging
import os
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app_config import get_config
from service.vision_pipeline import describe_image_with_models, init_models
from response.answer_generator import (
    answer_from_api,
    init_client as init_response_client,
    select_raw_description_for_api,
)
from response.config import tts_config as tcfg
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

detector = None
da_model = None
hands_full = None
hands_crop = None
response_client = None


def get_response_client():
    global response_client
    if response_client is None:
        response_client = init_response_client(tcfg.API_KEY, tcfg.BASE_URL)
    return response_client


@app.on_event("startup")
def load_models():
    global detector, da_model, hands_full, hands_crop
    logger.info("Loading vision models...")
    detector, da_model, hands_full, hands_crop = init_models()
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
                image_path, detector, da_model, hands_full, hands_crop
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return result, latency_ms

        parallel_start = time.perf_counter()
        (api_result, api_latency), (vision_result, vision_latency) = await asyncio.gather(
            asyncio.to_thread(run_api),
            asyncio.to_thread(run_vision),
        )
        parallel_wall_ms = (time.perf_counter() - parallel_start) * 1000.0
        api_string, transcript, api_steps = api_result
        description, timings, distance_desc = vision_result
        if not api_string:
            logger.warning("Task1 returned empty api_string, fallback to KHONG_XAC_DINH")
            api_string = "KHONG_XAC_DINH"
        if transcript is None:
            transcript = ""

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
            "Task2 Vision | wall=%.1f ms | internal=%.1f ms | accounted=%.1f ms | unaccounted=%.1f ms | steps(ms): load+resize=%.1f detector_total=%.1f yolo_primary=%.1f yolo_world=%.1f hand=%.1f focal=%.1f depth=%.1f calib+obj=%.1f filter=%.1f scene=%.1f distance_desc=%.1f | detector_substeps(ms): yolo_full=%.1f yolo_filter=%.1f yolo_tiled=%.1f yolo_nms=%.1f yolo_redetect=%.1f yw_full=%.1f yw_filter=%.1f yw_tiled=%.1f yw_nms=%.1f yw_redetect=%.1f merge=%.1f detector_overhead=%.1f | raw=%s",
            vision_latency,
            timings.get("total_ms", 0.0),
            timings.get("accounted_ms", 0.0),
            timings.get("unaccounted_ms", 0.0),
            timings.get("load_resize_ms", 0.0),
            timings.get("detector_total_ms", timings.get("yolo_ms", 0.0)),
            timings.get("yolo_ms", 0.0),
            timings.get("yolo_world_ms", 0.0),
            timings.get("hand_ms", 0.0),
            timings.get("focal_ms", 0.0),
            timings.get("depth_ms", 0.0),
            timings.get("calib_objects_ms", 0.0),
            timings.get("filter_ms", 0.0),
            timings.get("scene_ms", 0.0),
            timings.get("distance_desc_ms", 0.0),
            timings.get("yolo_full_infer_ms", 0.0),
            timings.get("yolo_full_filter_ms", 0.0),
            timings.get("yolo_tiled_ms", 0.0),
            timings.get("yolo_nms_ms", 0.0),
            timings.get("yolo_redetect_ms", 0.0),
            timings.get("yolo_world_full_infer_ms", 0.0),
            timings.get("yolo_world_full_filter_ms", 0.0),
            timings.get("yolo_world_tiled_ms", 0.0),
            timings.get("yolo_world_nms_ms", 0.0),
            timings.get("yolo_world_redetect_ms", 0.0),
            timings.get("detector_merge_ms", 0.0),
            timings.get("detector_overhead_ms", 0.0),
            description,
        )
        logger.info(
            "Parallel STT+Vision wall=%.1f ms | expected~max(task1,task2)",
            parallel_wall_ms,
        )
        logger.info("Distance matrix (camera -> objects): %s", distance_desc)
        selected_description = select_raw_description_for_api(
            api_string,
            distance_desc,
            description,
        )
        if selected_description != description:
            logger.info(
                "Selected raw description for api=%s | raw=%s",
                api_string,
                selected_description,
            )

        response_latency = 0.0
        try:
            response_start = time.perf_counter()
            smooth_text = answer_from_api(
                get_response_client(),
                api_string,
                transcript,
                distance_desc,
                selected_description,
                model=tcfg.LLM_MODEL,
                max_tokens=tcfg.MAX_TOKENS,
                temperature=tcfg.TEMPERATURE,
                thinking_enabled=tcfg.THINKING_ENABLED,
                reasoning_effort=tcfg.REASONING_EFFORT,
                log_cache_usage=tcfg.LOG_CACHE_USAGE,
            )
            response_latency = (time.perf_counter() - response_start) * 1000.0
        except Exception as exc:
            logger.warning("LLM answer failed: %s", exc)
            smooth_text = selected_description

        response = {"api": api_string, "text": smooth_text}
        total_ms = (time.perf_counter() - req_start) * 1000.0
        logger.info("Task3 Response | total=%.1f ms", response_latency)
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

    ssl_certfile = _cfg.get("ssl_certfile") or None
    ssl_keyfile = _cfg.get("ssl_keyfile") or None
    uvicorn.run(
        app,
        host=_cfg["host"],
        port=int(_cfg["port"]),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
