from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time

from vision.config import vision_config as cfg

from .backends import ensure_detector_backend
from .pipeline import (
    boxes_to_detection_records,
    build_open_vocab_prompts,
    filter_allowed_classes,
    merge_detection_records,
    redetect_config_from_runtime,
    run_detector_pipeline,
)

logger = logging.getLogger("voice_server.vision")


class VisionDetectorService:
    def __init__(self, primary_backend, open_vocab_backend=None):
        self.primary_backend = ensure_detector_backend(primary_backend, source_name="yolo")
        self.open_vocab_backend = ensure_detector_backend(
            open_vocab_backend,
            source_name="yolo_world",
        )
        self.last_detection_timings_ms: dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision-detector")

    def reset_predictors(self):
        if self.primary_backend is not None:
            self.primary_backend.reset()
        if self.open_vocab_backend is not None:
            self.open_vocab_backend.reset()

    def _resolve_open_vocab_labels(self, prompts: list[str]) -> set[str]:
        if self.open_vocab_backend is None:
            return set()
        if self.open_vocab_backend.supports_prompt_labels():
            return set(prompts)
        available_labels = set(self.open_vocab_backend.label_lookup().values())
        return {label for label in prompts if label in available_labels}

    def prepare_open_vocab_candidates(self) -> tuple[list[str], set[str], float]:
        start = time.perf_counter()
        prompts = build_open_vocab_prompts(self.primary_backend)
        candidate_labels = self._resolve_open_vocab_labels(prompts)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return prompts, candidate_labels, elapsed_ms

    def _run_primary_detection(self, img, orig_img=None) -> tuple[list[dict], float, dict[str, float]]:
        start = time.perf_counter()
        final_boxes, pipeline_timings = run_detector_pipeline(
            self.primary_backend,
            img,
            orig_img=orig_img,
            conf_threshold=cfg.YOLO_PREDICT_CONF,
            imgsz=cfg.YOLO_IMGSZ,
            device=cfg.YOLO_DEVICE,
            filter_fn=filter_allowed_classes,
            redetect_config=redetect_config_from_runtime(),
            log_prefix="YOLO",
            return_timings=True,
            tiled_enabled=cfg.TILED_ENABLED,
        )
        records = boxes_to_detection_records(
            final_boxes,
            self.primary_backend.label_lookup(),
            source="yolo",
        )
        return records, (time.perf_counter() - start) * 1000.0, pipeline_timings

    def _run_open_vocab_detection(
        self,
        img,
        candidate_labels: set[str],
        orig_img=None,
    ) -> tuple[list[dict], float, dict[str, float]]:
        if self.open_vocab_backend.supports_prompt_labels():
            self.open_vocab_backend.set_prompt_labels(sorted(candidate_labels))
        redetect_config = redetect_config_from_runtime(
            classes=candidate_labels,
            conf_thr={"default": cfg.OPEN_VOCAB_CONF_THRESHOLD},
            enabled=cfg.OPEN_VOCAB_REDETECT_ENABLED,
        )
        start = time.perf_counter()
        boxes, pipeline_timings = run_detector_pipeline(
            self.open_vocab_backend,
            img,
            orig_img=orig_img,
            conf_threshold=cfg.OPEN_VOCAB_CONF_THRESHOLD,
            imgsz=cfg.YOLO_IMGSZ,
            device=cfg.OPEN_VOCAB_DEVICE,
            redetect_config=redetect_config,
            log_prefix="YOLO-World",
            return_timings=True,
            tiled_enabled=cfg.OPEN_VOCAB_TILED_ENABLED,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if boxes.size == 0:
            return [], elapsed_ms, pipeline_timings

        records = boxes_to_detection_records(
            boxes.astype("float32"),
            self.open_vocab_backend.label_lookup(),
            source="yolo_world",
        )
        filtered_records = [
            record
            for record in records
            if record["label"] in candidate_labels
            and record["label"] not in cfg.OPEN_VOCAB_EXCLUDE_LABELS
        ]
        return filtered_records, elapsed_ms, pipeline_timings

    def detect_objects(self, img, orig_img=None) -> list[dict]:
        request_start = time.perf_counter()
        self.last_detection_timings_ms = {}

        prompts, candidate_labels, prompt_ms = self.prepare_open_vocab_candidates()
        self.last_detection_timings_ms["open_vocab_prompt_ms"] = prompt_ms

        primary_future = self.executor.submit(self._run_primary_detection, img, orig_img)
        open_vocab_future = None
        if self.open_vocab_backend is not None and cfg.OPEN_VOCAB_ENABLED and candidate_labels:
            open_vocab_future = self.executor.submit(
                self._run_open_vocab_detection,
                img,
                candidate_labels,
                orig_img,
            )
        elif self.open_vocab_backend is not None and cfg.OPEN_VOCAB_ENABLED:
            logger.warning(
                "YOLO-World branch skipped because no configured open-vocab labels exist in the active backend vocabulary. prompts=%d",
                len(prompts),
            )

        yolo_records, yolo_ms, primary_pipeline_timings = primary_future.result()
        self.last_detection_timings_ms["yolo_ms"] = yolo_ms
        for key, value in primary_pipeline_timings.items():
            self.last_detection_timings_ms[f"yolo_{key}"] = value
        if open_vocab_future is not None:
            open_vocab_records, yolo_world_ms, open_vocab_pipeline_timings = open_vocab_future.result()
        else:
            open_vocab_records, yolo_world_ms, open_vocab_pipeline_timings = [], 0.0, {}
        self.last_detection_timings_ms["yolo_world_ms"] = yolo_world_ms
        for key, value in open_vocab_pipeline_timings.items():
            self.last_detection_timings_ms[f"yolo_world_{key}"] = value

        step_start = time.perf_counter()
        merged_records = merge_detection_records(
            yolo_records,
            open_vocab_records,
            cfg.OPEN_VOCAB_IOU_THRESHOLD,
        )
        self.last_detection_timings_ms["detector_merge_ms"] = (
            time.perf_counter() - step_start
        ) * 1000.0
        self.last_detection_timings_ms["detector_total_ms"] = (
            time.perf_counter() - request_start
        ) * 1000.0
        accounted_ms = max(yolo_ms, yolo_world_ms) + self.last_detection_timings_ms["detector_merge_ms"]
        self.last_detection_timings_ms["detector_overhead_ms"] = max(
            0.0,
            self.last_detection_timings_ms["detector_total_ms"] - accounted_ms,
        )
        if open_vocab_records:
            candidate_labels = sorted({record["label"] for record in open_vocab_records})
            logger.info(
                "YOLO-World extra detections outside YOLO labels: labels=%s count=%d merged_total=%d",
                candidate_labels,
                len(open_vocab_records),
                len(merged_records),
            )
        logger.info(
            "Detector latency breakdown: total=%.1f ms | yolo=%.1f ms | yolo_world=%.1f ms | merge=%.1f ms | overhead=%.1f ms",
            self.last_detection_timings_ms.get("detector_total_ms", 0.0),
            self.last_detection_timings_ms.get("yolo_ms", 0.0),
            self.last_detection_timings_ms.get("yolo_world_ms", 0.0),
            self.last_detection_timings_ms.get("detector_merge_ms", 0.0),
            self.last_detection_timings_ms.get("detector_overhead_ms", 0.0),
        )
        return merged_records
