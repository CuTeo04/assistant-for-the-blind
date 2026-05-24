import os

from app_config import get_config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_cfg = get_config()["vision"]

REAL_HAND_LENGTH_CM = _cfg["real_hand_length_cm"]
KNOWN_DISTANCE_CM = _cfg["known_distance_cm"]
MAX_SIZE = _cfg["max_size"]
DEVICE = _cfg["device"]
NUM_THREADS = _cfg["num_threads"]
INPUT_SIZE_DEPTH = _cfg["input_size_depth"]

YOLO_MODEL_PATH = os.path.join(ROOT_DIR, _cfg["yolo_model_path"])
DA2_CHECKPOINT = os.path.join(ROOT_DIR, _cfg["da2_checkpoint"])
DA2_ONNX_PATH = os.path.join(ROOT_DIR, _cfg["da2_onnx_path"])
DA2_USE_ONNX = _cfg["da2_use_onnx"]
DA2_DIR = os.path.join(ROOT_DIR, "models", "Depth-Anything-V2")
DA2_METRIC_DIR = os.path.join(DA2_DIR, "metric_depth")
DA2_CONFIG = _cfg["da2_config"]

CONF_THRESHOLD = _cfg["conf_threshold"]
MIN_DEPTH_M = _cfg["min_depth_m"]
MAX_DEPTH_M = _cfg["max_depth_m"]
MAX_OBJECTS = _cfg["max_objects"]

YOLO_DEVICE = _cfg.get("yolo_device", DEVICE)
