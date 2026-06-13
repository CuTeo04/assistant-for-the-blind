from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time

import numpy as np

from log_settings import is_enabled
from vision.config import vision_config as cfg

from .backends import ensure_detector_backend
from .pipeline import (
    boxes_to_detection_records,
    build_open_vocab_prompts,
    class_aware_nms,
    filter_allowed_classes,
    merge_detection_records,
    redetect_config_from_runtime,
    run_detector_pipeline,
)

logger = logging.getLogger("voice_server.vision")


class VisionDetectorService:
    def __init__(
        self,
        primary_backend,
        open_vocab_backend=None,
        *,
        primary_full_backend=None,
        primary_tile_backend=None,
        open_vocab_full_backend=None,
        open_vocab_tile_backend=None,
    ):
        self.primary_backend = ensure_detector_backend(primary_backend, source_name="yolo")
        self.open_vocab_backend = ensure_detector_backend(
            open_vocab_backend,
            source_name="yolo_world",
        )
        self.primary_full_backend = ensure_detector_backend(
            primary_full_backend,
            source_name="yolo_full",
        )
        self.primary_tile_backend = ensure_detector_backend(
            primary_tile_backend,
            source_name="yolo_tile",
        )
        self.open_vocab_full_backend = ensure_detector_backend(
            open_vocab_full_backend,
            source_name="yolo_world_full",
        )
        self.open_vocab_tile_backend = ensure_detector_backend(
            open_vocab_tile_backend,
            source_name="yolo_world_tile",
        )
        self.last_detection_timings_ms: dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision-detector")
        self.full_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-detector-full")
        self.split_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vision-detector-split")

    def reset_predictors(self):
        for backend in (
            self.primary_backend,
            self.open_vocab_backend,
            self.primary_full_backend,
            self.primary_tile_backend,
            self.open_vocab_full_backend,
            self.open_vocab_tile_backend,
        ):
            if backend is not None:
                backend.reset()

    def _primary_label_backend(self):
        return self.primary_backend or self.primary_full_backend or self.primary_tile_backend

    def _open_vocab_label_backend(self):
        return self.open_vocab_backend or self.open_vocab_full_backend or self.open_vocab_tile_backend

    def _merge_split_boxes(self, full_boxes: np.ndarray, tile_boxes: np.ndarray) -> np.ndarray:
        if full_boxes.size == 0:
            return tile_boxes.astype(np.float32)
        if tile_boxes.size == 0:
            return full_boxes.astype(np.float32)
        combined = np.vstack([full_boxes, tile_boxes]).astype(np.float32)
        return class_aware_nms(combined, cfg.NMS_IOU_THRESHOLD)

    def _resolve_open_vocab_labels(self, prompts: list[str]) -> set[str]:
        backend = self._open_vocab_label_backend()
        if backend is None:
            return set()
        if backend.supports_prompt_labels():
            return set(prompts)
        available_labels = set(backend.label_lookup().values())
        return {label for label in prompts if label in available_labels}

    def prepare_open_vocab_candidates(self) -> tuple[list[str], set[str], float]:
        start = time.perf_counter()
        prompts = build_open_vocab_prompts(self._primary_label_backend())
        candidate_labels = self._resolve_open_vocab_labels(prompts)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return prompts, candidate_labels, elapsed_ms

    def _run_primary_full_detection(self, img) -> tuple[np.ndarray, dict[str, float]]:
        return run_detector_pipeline(
            self.primary_full_backend,
            img,
            orig_img=None,
            conf_threshold=cfg.YOLO_PREDICT_CONF,
            imgsz=cfg.YOLO_IMGSZ,
            device=cfg.YOLO_DEVICE,
            filter_fn=filter_allowed_classes,
            redetect_config=redetect_config_from_runtime(),
            log_prefix="YOLO(full)",
            return_timings=True,
            tiled_enabled=False,
            full_image_enabled=True,
            tile_mode=cfg.TILE_MODE,
            center_tile_count=cfg.CENTER_TILE_COUNT,
        )

    def _run_primary_tile_detection(self, img, orig_img) -> tuple[np.ndarray, dict[str, float]]:
        return run_detector_pipeline(
            self.primary_tile_backend,
            img,
            orig_img=orig_img,
            conf_threshold=cfg.YOLO_PREDICT_CONF,
            imgsz=cfg.YOLO_TILE_IMGSZ,
            device=cfg.YOLO_DEVICE,
            filter_fn=filter_allowed_classes,
            redetect_config=redetect_config_from_runtime(),
            log_prefix="YOLO(tile)",
            return_timings=True,
            tiled_enabled=cfg.TILED_ENABLED,
            full_image_enabled=False,
            tile_imgsz=cfg.YOLO_TILE_IMGSZ,
            tile_size=cfg.TILE_SIZE,
            tile_overlap=cfg.TILE_OVERLAP,
            tile_mode=cfg.TILE_MODE,
            center_tile_count=cfg.CENTER_TILE_COUNT,
        )

    def _run_primary_detection(self, img, orig_img=None) -> tuple[list[dict], float, dict[str, float]]:
        start = time.perf_counter()
        if self.primary_full_backend is not None and self.primary_tile_backend is not None:
            full_future = self.full_executor.submit(self._run_primary_full_detection, img)
            tile_future = self.split_executor.submit(self._run_primary_tile_detection, img, orig_img)
            full_boxes, full_timings = full_future.result()
            tile_boxes, tile_timings = tile_future.result()
            step_start = time.perf_counter()
            final_boxes = self._merge_split_boxes(full_boxes, tile_boxes)
            merge_ms = (time.perf_counter() - step_start) * 1000.0
            pipeline_timings = {
                "split_full_pipeline_ms": full_timings.get("pipeline_ms", 0.0),
                "split_tile_pipeline_ms": tile_timings.get("pipeline_ms", 0.0),
                "split_merge_ms": merge_ms,
                "full_infer_ms": full_timings.get("full_infer_ms", 0.0),
                "full_filter_ms": full_timings.get("full_filter_ms", 0.0),
                "tiled_ms": tile_timings.get("tiled_ms", 0.0),
                "redetect_ms": full_timings.get("redetect_ms", 0.0) + tile_timings.get("redetect_ms", 0.0),
                "nms_ms": full_timings.get("nms_ms", 0.0) + tile_timings.get("nms_ms", 0.0),
                "pipeline_ms": max(
                    full_timings.get("pipeline_ms", 0.0),
                    tile_timings.get("pipeline_ms", 0.0),
                )
                + merge_ms,
            }
            label_lookup_backend = self.primary_full_backend
        else:
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
                tile_imgsz=cfg.YOLO_TILE_IMGSZ,
                tile_size=cfg.TILE_SIZE,
                tile_overlap=cfg.TILE_OVERLAP,
                tile_mode=cfg.TILE_MODE,
                center_tile_count=cfg.CENTER_TILE_COUNT,
            )
            label_lookup_backend = self.primary_backend
        records = boxes_to_detection_records(
            final_boxes,
            label_lookup_backend.label_lookup(),
            source="yolo",
        )
        return records, (time.perf_counter() - start) * 1000.0, pipeline_timings

    def _run_open_vocab_full_detection(
        self,
        img,
        candidate_labels: set[str],
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.open_vocab_full_backend.supports_prompt_labels():
            self.open_vocab_full_backend.set_prompt_labels(sorted(candidate_labels))
        redetect_config = redetect_config_from_runtime(
            classes=candidate_labels,
            conf_thr={"default": cfg.OPEN_VOCAB_CONF_THRESHOLD},
            enabled=cfg.OPEN_VOCAB_REDETECT_ENABLED,
        )
        return run_detector_pipeline(
            self.open_vocab_full_backend,
            img,
            orig_img=None,
            conf_threshold=cfg.OPEN_VOCAB_CONF_THRESHOLD,
            imgsz=cfg.YOLO_IMGSZ,
            device=cfg.OPEN_VOCAB_DEVICE,
            redetect_config=redetect_config,
            log_prefix="YOLO-World(full)",
            return_timings=True,
            tiled_enabled=False,
            full_image_enabled=True,
            tile_mode=cfg.OPEN_VOCAB_TILE_MODE,
            center_tile_count=cfg.OPEN_VOCAB_CENTER_TILE_COUNT,
        )

    def _run_open_vocab_tile_detection(
        self,
        img,
        candidate_labels: set[str],
        orig_img,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.open_vocab_tile_backend.supports_prompt_labels():
            self.open_vocab_tile_backend.set_prompt_labels(sorted(candidate_labels))
        redetect_config = redetect_config_from_runtime(
            classes=candidate_labels,
            conf_thr={"default": cfg.OPEN_VOCAB_CONF_THRESHOLD},
            enabled=cfg.OPEN_VOCAB_REDETECT_ENABLED,
        )
        return run_detector_pipeline(
            self.open_vocab_tile_backend,
            img,
            orig_img=orig_img,
            conf_threshold=cfg.OPEN_VOCAB_CONF_THRESHOLD,
            imgsz=cfg.OPEN_VOCAB_TILE_IMGSZ,
            device=cfg.OPEN_VOCAB_DEVICE,
            redetect_config=redetect_config,
            log_prefix="YOLO-World(tile)",
            return_timings=True,
            tiled_enabled=cfg.OPEN_VOCAB_TILED_ENABLED,
            full_image_enabled=False,
            tile_imgsz=cfg.OPEN_VOCAB_TILE_IMGSZ,
            tile_size=cfg.OPEN_VOCAB_TILE_SIZE,
            tile_overlap=cfg.OPEN_VOCAB_TILE_OVERLAP,
            tile_mode=cfg.OPEN_VOCAB_TILE_MODE,
            center_tile_count=cfg.OPEN_VOCAB_CENTER_TILE_COUNT,
        )

    def _run_open_vocab_detection(
        self,
        img,
        candidate_labels: set[str],
        orig_img=None,
    ) -> tuple[list[dict], float, dict[str, float]]:
        start = time.perf_counter()
        if self.open_vocab_full_backend is not None and self.open_vocab_tile_backend is not None:
            full_future = self.full_executor.submit(
                self._run_open_vocab_full_detection,
                img,
                candidate_labels,
            )
            tile_future = self.split_executor.submit(
                self._run_open_vocab_tile_detection,
                img,
                candidate_labels,
                orig_img,
            )
            full_boxes, full_timings = full_future.result()
            tile_boxes, tile_timings = tile_future.result()
            step_start = time.perf_counter()
            boxes = self._merge_split_boxes(full_boxes, tile_boxes)
            merge_ms = (time.perf_counter() - step_start) * 1000.0
            pipeline_timings = {
                "split_full_pipeline_ms": full_timings.get("pipeline_ms", 0.0),
                "split_tile_pipeline_ms": tile_timings.get("pipeline_ms", 0.0),
                "split_merge_ms": merge_ms,
                "full_infer_ms": full_timings.get("full_infer_ms", 0.0),
                "full_filter_ms": full_timings.get("full_filter_ms", 0.0),
                "tiled_ms": tile_timings.get("tiled_ms", 0.0),
                "redetect_ms": full_timings.get("redetect_ms", 0.0) + tile_timings.get("redetect_ms", 0.0),
                "nms_ms": full_timings.get("nms_ms", 0.0) + tile_timings.get("nms_ms", 0.0),
                "pipeline_ms": max(
                    full_timings.get("pipeline_ms", 0.0),
                    tile_timings.get("pipeline_ms", 0.0),
                )
                + merge_ms,
            }
            label_lookup_backend = self.open_vocab_full_backend
        else:
            if self.open_vocab_backend.supports_prompt_labels():
                self.open_vocab_backend.set_prompt_labels(sorted(candidate_labels))
            redetect_config = redetect_config_from_runtime(
                classes=candidate_labels,
                conf_thr={"default": cfg.OPEN_VOCAB_CONF_THRESHOLD},
                enabled=cfg.OPEN_VOCAB_REDETECT_ENABLED,
            )
            boxes, pipeline_timings = run_detector_pipeline(
                self.open_vocab_backend,
                img,
                orig_img=orig_img,
                conf_threshold=cfg.OPEN_VOCAB_CONF_THRESHOLD,
                imgsz=cfg.OPEN_VOCAB_TILE_IMGSZ,
                device=cfg.OPEN_VOCAB_DEVICE,
                redetect_config=redetect_config,
                log_prefix="YOLO-World",
                return_timings=True,
                tiled_enabled=cfg.OPEN_VOCAB_TILED_ENABLED,
                tile_imgsz=cfg.OPEN_VOCAB_TILE_IMGSZ,
                tile_size=cfg.OPEN_VOCAB_TILE_SIZE,
                tile_overlap=cfg.OPEN_VOCAB_TILE_OVERLAP,
                tile_mode=cfg.OPEN_VOCAB_TILE_MODE,
                center_tile_count=cfg.OPEN_VOCAB_CENTER_TILE_COUNT,
            )
            label_lookup_backend = self.open_vocab_backend
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if boxes.size == 0:
            return [], elapsed_ms, pipeline_timings

        records = boxes_to_detection_records(
            boxes.astype("float32"),
            label_lookup_backend.label_lookup(),
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
        open_vocab_ready = self._open_vocab_label_backend() is not None

        prompts, candidate_labels, prompt_ms = self.prepare_open_vocab_candidates()
        self.last_detection_timings_ms["open_vocab_prompt_ms"] = prompt_ms

        primary_future = self.executor.submit(self._run_primary_detection, img, orig_img)
        open_vocab_future = None
        if open_vocab_ready and cfg.OPEN_VOCAB_ENABLED and candidate_labels:
            open_vocab_future = self.executor.submit(
                self._run_open_vocab_detection,
                img,
                candidate_labels,
                orig_img,
            )
        elif open_vocab_ready and cfg.OPEN_VOCAB_ENABLED:
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
        if open_vocab_records and is_enabled("vision_detector_debug", True):
            candidate_labels = sorted({record["label"] for record in open_vocab_records})
            logger.info(
                "YOLO-World extra detections outside YOLO labels: labels=%s count=%d merged_total=%d",
                candidate_labels,
                len(open_vocab_records),
                len(merged_records),
            )
        if is_enabled("vision_detector_debug", True):
            logger.info(
                "Detector latency breakdown: total=%.1f ms | yolo=%.1f ms | yolo_world=%.1f ms | merge=%.1f ms | overhead=%.1f ms",
                self.last_detection_timings_ms.get("detector_total_ms", 0.0),
                self.last_detection_timings_ms.get("yolo_ms", 0.0),
                self.last_detection_timings_ms.get("yolo_world_ms", 0.0),
                self.last_detection_timings_ms.get("detector_merge_ms", 0.0),
                self.last_detection_timings_ms.get("detector_overhead_ms", 0.0),
            )
        return merged_records
