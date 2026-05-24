from ultralytics import YOLO

from .config import vision_config as cfg


def load_yolo(model_path: str):
    return YOLO(model_path)


def detect_objects(yolo, img):
    results = yolo.predict(img, device=cfg.YOLO_DEVICE, verbose=False)[0]
    return results.boxes.data.cpu().numpy()


def compute_object_size_cm(box_px, depth_m: float, focal_px: float):
    x1, y1, x2, y2 = box_px
    depth_cm = depth_m * 100.0
    real_w_cm = ((x2 - x1) / focal_px) * depth_cm
    real_h_cm = ((y2 - y1) / focal_px) * depth_cm
    return real_w_cm, real_h_cm
