import os
import sys

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from .config import vision_config as cfg


def _ensure_depth_anything_path():
    preferred_paths = [cfg.DA2_METRIC_DIR, cfg.DA2_DIR]
    for path in reversed(preferred_paths):
        if path in sys.path:
            sys.path.remove(path)
    for path in reversed(preferred_paths):
        sys.path.insert(0, path)


_ensure_depth_anything_path()

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class DepthAnythingOnnx:
    def __init__(self, onnx_path: str, input_size: int = 518):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX inference")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Depth Anything V2 ONNX not found at: {onnx_path}")

        self.input_size = input_size
        session_options = ort.SessionOptions()
        if cfg.DA2_ONNX_INTRA_OP_THREADS is not None:
            session_options.intra_op_num_threads = int(cfg.DA2_ONNX_INTRA_OP_THREADS)
        if cfg.DA2_ONNX_INTER_OP_THREADS is not None:
            session_options.inter_op_num_threads = int(cfg.DA2_ONNX_INTER_OP_THREADS)
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.fixed_h = input_shape[2] if isinstance(input_shape[2], int) else None
        self.fixed_w = input_shape[3] if isinstance(input_shape[3], int) else None

    def _build_transform(self, input_size: int):
        return Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def infer_image(self, raw_image, input_size: int = 518):
        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        if self.fixed_h and self.fixed_w:
            image = cv2.resize(image, (self.fixed_w, self.fixed_h), interpolation=cv2.INTER_CUBIC)
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        else:
            image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image = image[None, :, :, :].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: image})
        depth = np.squeeze(outputs[0])
        if depth.ndim == 3:
            depth = depth[0]

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        return depth.astype(np.float32)


def load_da2_model(config: dict, checkpoint: str, device):
    if cfg.DA2_USE_ONNX:
        return DepthAnythingOnnx(cfg.DA2_ONNX_PATH, input_size=cfg.INPUT_SIZE_DEPTH)

    # 1. Khởi tạo cấu trúc model
    model = DepthAnythingV2(**config)    
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Depth Anything V2 checkpoint not found tại: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location='cpu')  
    model.load_state_dict(state_dict)
    model.to(device).eval()   
    return model


def infer_depth(model, img, input_size: int):
    h, w = img.shape[:2]
    if isinstance(model, DepthAnythingOnnx):
        depth_raw = model.infer_image(img, input_size)
        return depth_raw, (h, w), 1.0

    scale_ratio = input_size / max(h, w)
    new_h = int(h * scale_ratio)
    new_w = int(w * scale_ratio)
    img_small = cv2.resize(img, (new_w, new_h))
    depth_raw = model.infer_image(img_small)
    return depth_raw, (new_h, new_w), scale_ratio


def calibrate_depth(depth_raw, raw_hand_depth: float, known_distance_cm: float):
    depth_scale = (known_distance_cm / 100.0) / raw_hand_depth
    depth_scaled = depth_raw * depth_scale
    return depth_scaled, depth_scale
