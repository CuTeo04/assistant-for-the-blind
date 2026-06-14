import os

from app_config import get_config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_cfg = get_config()["vision"]

REAL_HAND_LENGTH_CM = _cfg["real_hand_length_cm"]
KNOWN_DISTANCE_CM = _cfg["known_distance_cm"]
FIXED_FOCAL_LENGTH_PX = float(_cfg.get("fixed_focal_length_px", 800.0))
DEFAULT_DEPTH_SCALE = float(_cfg.get("default_depth_scale", 1.0))
MAX_SIZE = _cfg["max_size"]
DEVICE = _cfg["device"]
NUM_THREADS = _cfg["num_threads"]
CPU_GENERAL_THREADS = _cfg.get("cpu_general_threads", 4)
INPUT_SIZE_DEPTH = _cfg["input_size_depth"]

YOLO_MODEL_PATH = os.path.join(ROOT_DIR, _cfg["yolo_model_path"])
YOLO_ONNX_PATH = os.path.join(
    ROOT_DIR,
    _cfg.get("yolo_onnx_path", "models/YOLO11s.onnx"),
)
YOLO_BACKEND = str(_cfg.get("yolo_backend", "ultralytics")).strip().lower()
YOLO_EXECUTION_PROVIDER = str(_cfg.get("yolo_execution_provider", "cpu")).strip().lower()
YOLO_ONNX_INTRA_OP_THREADS = _cfg.get("yolo_onnx_intra_op_threads")
YOLO_ONNX_INTER_OP_THREADS = _cfg.get("yolo_onnx_inter_op_threads")
YOLO_COMBINED_BATCH_ENABLED = bool(_cfg.get("yolo_combined_batch_enabled", False))
YOLO_COMBINED_MAX_BATCH = int(_cfg.get("yolo_combined_max_batch", 5))
DA2_CHECKPOINT = os.path.join(ROOT_DIR, _cfg["da2_checkpoint"])
DA2_ONNX_PATH = os.path.join(ROOT_DIR, _cfg["da2_onnx_path"])
DA2_USE_ONNX = _cfg["da2_use_onnx"]
DA2_ONNX_INTRA_OP_THREADS = _cfg.get("da2_onnx_intra_op_threads")
DA2_ONNX_INTER_OP_THREADS = _cfg.get("da2_onnx_inter_op_threads")
DA2_DIR = os.path.join(ROOT_DIR, "models", "Depth-Anything-V2")
DA2_METRIC_DIR = os.path.join(DA2_DIR, "metric_depth")
DA2_CONFIG = _cfg["da2_config"]

CONF_THRESHOLD = _cfg["conf_threshold"]
ALLOWED_CLASSES = {str(label).strip().lower() for label in _cfg.get("allowed_classes", [])}
MIN_DEPTH_M = _cfg["min_depth_m"]
MAX_DEPTH_M = _cfg["max_depth_m"]
DESCRIPTION_MAX_OBJECTS = int(_cfg["description_max_objects"])
DEBUG_IMAGE_MAX_OBJECTS = int(_cfg["debug_image_max_objects"])

YOLO_DEVICE = _cfg.get("yolo_device", DEVICE)
YOLO_IMGSZ = int(_cfg.get("yolo_imgsz", 960))
YOLO_TILE_IMGSZ = int(_cfg.get("yolo_tile_imgsz", YOLO_IMGSZ))
YOLO_PREDICT_CONF = _cfg.get("yolo_predict_conf", 0.25)
TILED_ENABLED = bool(_cfg.get("tiled_enabled", False))
TILE_SIZE = int(_cfg.get("tile_size", 960))
TILE_OVERLAP = float(_cfg.get("tile_overlap", 0.25))
TILE_MODE = str(_cfg.get("tile_mode", "center_batch")).strip().lower()
CENTER_TILE_COUNT = int(_cfg.get("center_tile_count", 2))
TILE_CONF_THRESHOLD = float(_cfg.get("tile_conf_threshold", YOLO_PREDICT_CONF))
NMS_IOU_THRESHOLD = float(_cfg.get("nms_iou_threshold", 0.50))
REDETECT_ENABLED = bool(_cfg.get("redetect_enabled", False))
REDETECT_CLASSES = {str(label).strip().lower() for label in _cfg.get("redetect_classes", [])}
REDETECT_CONTAINMENT_THR = float(_cfg.get("redetect_containment_thr", 0.75))
REDETECT_IOU_THR = float(_cfg.get("redetect_iou_thr", 0.45))
REDETECT_CENTER_DIST_RATIO = float(_cfg.get("redetect_center_dist_ratio", 0.35))
REDETECT_PADDING_RATIO = float(_cfg.get("redetect_padding_ratio", 0.15))
REDETECT_IMGSZ = int(_cfg.get("redetect_imgsz", 960))
REDETECT_CONF_THR = _cfg.get("redetect_conf_thr", {})
REDETECT_MAX_CANDIDATES = int(_cfg.get("redetect_max_candidates", 5))
REDETECT_FALLBACK = str(_cfg.get("redetect_fallback", "keep_both"))
REDETECT_MIN_BOX_SIZE = int(_cfg.get("redetect_min_box_size", 16))
REDETECT_MIN_CROP_SIZE = int(_cfg.get("redetect_min_crop_size", 32))

OPEN_VOCAB_ENABLED = bool(_cfg.get("open_vocab_enabled", False))
YOLO_WORLD_MODEL_PATH = os.path.join(
    ROOT_DIR,
    _cfg.get("yolo_world_model_path", "models/yolov8x-worldv2.onnx"),
)
YOLO_WORLD_WEIGHTS_PATH = os.path.join(
    ROOT_DIR,
    _cfg.get("yolo_world_weights_path", "models/yolov8x-worldv2.pt"),
)
YOLO_WORLD_BACKEND = str(_cfg.get("yolo_world_backend", "ultralytics")).strip().lower()
YOLO_WORLD_EXECUTION_PROVIDER = str(_cfg.get("yolo_world_execution_provider", "cpu")).strip().lower()
YOLO_WORLD_ONNX_INTRA_OP_THREADS = _cfg.get("yolo_world_onnx_intra_op_threads")
YOLO_WORLD_ONNX_INTER_OP_THREADS = _cfg.get("yolo_world_onnx_inter_op_threads")
OPEN_VOCAB_COMBINED_BATCH_ENABLED = bool(_cfg.get("open_vocab_combined_batch_enabled", False))
OPEN_VOCAB_COMBINED_MAX_BATCH = int(_cfg.get("open_vocab_combined_max_batch", 5))
OPEN_VOCAB_EXTRA_CLASSES = [
    str(label).strip().lower()
    for label in _cfg.get(
        "open_vocab_extra_classes",
        _cfg.get("open_vocab_classes", []),
    )
    if str(label).strip()
]
OPEN_VOCAB_EXCLUDE_LABELS = {
    str(label).strip().lower()
    for label in _cfg.get("open_vocab_exclude_labels", ["person"])
    if str(label).strip()
}
OPEN_VOCAB_CONF_THRESHOLD = float(_cfg.get("open_vocab_conf_threshold", 0.20))
OPEN_VOCAB_IOU_THRESHOLD = float(_cfg.get("open_vocab_iou_threshold", NMS_IOU_THRESHOLD))
OPEN_VOCAB_DEVICE = _cfg.get("open_vocab_device", YOLO_DEVICE)
OPEN_VOCAB_TILE_IMGSZ = int(_cfg.get("open_vocab_tile_imgsz", YOLO_TILE_IMGSZ))
OPEN_VOCAB_TILED_ENABLED = bool(_cfg.get("open_vocab_tiled_enabled", False))
OPEN_VOCAB_TILE_SIZE = int(_cfg.get("open_vocab_tile_size", TILE_SIZE))
OPEN_VOCAB_TILE_OVERLAP = float(_cfg.get("open_vocab_tile_overlap", TILE_OVERLAP))
OPEN_VOCAB_TILE_MODE = str(_cfg.get("open_vocab_tile_mode", TILE_MODE)).strip().lower()
OPEN_VOCAB_CENTER_TILE_COUNT = int(_cfg.get("open_vocab_center_tile_count", CENTER_TILE_COUNT))
OPEN_VOCAB_REDETECT_ENABLED = bool(_cfg.get("open_vocab_redetect_enabled", False))
