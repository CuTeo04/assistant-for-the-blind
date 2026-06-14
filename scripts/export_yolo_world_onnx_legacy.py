import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
YOLO_CONFIG_DIR = ROOT_DIR / ".cache" / "ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)

import torch
from ultralytics import YOLOWorld

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app_config import get_config


def _normalize_labels(labels) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip().lower()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def parse_args():
    vision_cfg = get_config()["vision"]
    parser = argparse.ArgumentParser(
        description="Export YOLO-World ONNX using legacy torch.onnx exporter (dynamo=False).",
    )
    parser.add_argument(
        "--weights",
        default=str(vision_cfg.get("yolo_world_weights_path", "models/yolov8x-worldv2.pt")),
        help="Input YOLO-World .pt weights path.",
    )
    parser.add_argument(
        "--output",
        default=str(vision_cfg["yolo_world_model_path"]),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(vision_cfg.get("yolo_imgsz", vision_cfg.get("tile_size", 960))),
        help="Export image size.",
    )
    parser.add_argument("--batch", type=int, default=3, help="Static export batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version override")
    parser.add_argument("--device", default="cpu", help="Device used during export")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shapes",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run ONNX simplifier after export when available",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vision_cfg = get_config()["vision"]

    weights_path = (ROOT_DIR / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)
    output_path = (ROOT_DIR / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO-World weights not found at: {weights_path}")

    labels = _normalize_labels(vision_cfg.get("open_vocab_extra_classes", []))
    if not labels:
        raise ValueError("vision.open_vocab_extra_classes is empty; nothing to embed into YOLO-World ONNX.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_export = torch.onnx.export

    def legacy_export(*export_args, **export_kwargs):
        export_kwargs.setdefault("dynamo", False)
        return original_export(*export_args, **export_kwargs)

    torch.onnx.export = legacy_export
    try:
        model = YOLOWorld(str(weights_path))
        model.set_classes(labels)
        export_kwargs = {
            "format": "onnx",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "dynamic": args.dynamic,
            "simplify": args.simplify,
            "device": args.device,
            "opset": args.opset,
        }

        exported_path = Path(model.export(**export_kwargs))
    finally:
        torch.onnx.export = original_export

    if exported_path.resolve() != output_path:
        if output_path.exists():
            output_path.unlink()
        exported_path.replace(output_path)

    print(f"Embedded {len(labels)} labels into YOLO-World ONNX")
    print(f"Exported legacy ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
