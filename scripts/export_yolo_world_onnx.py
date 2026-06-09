import argparse
import sys
from pathlib import Path

from ultralytics import YOLOWorld

ROOT_DIR = Path(__file__).resolve().parents[1]
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
        description="Export YOLO-World ONNX using open-vocab labels from server_config.json",
    )
    parser.add_argument(
        "--weights",
        default=str(vision_cfg.get("yolo_world_weights_path", "models/yolov8x-worldv2.pt")),
        help="Input YOLO-World .pt weights path. Defaults to vision.yolo_world_weights_path in server_config.json.",
    )
    parser.add_argument(
        "--output",
        default=str(vision_cfg["yolo_world_model_path"]),
        help="Output ONNX path. Defaults to vision.yolo_world_model_path in server_config.json.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(vision_cfg.get("yolo_imgsz", vision_cfg.get("tile_size", 960))),
        help="Export image size.",
    )
    parser.add_argument("--opset", type=int, default=None, help="Optional ONNX opset version override")
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

    model = YOLOWorld(str(weights_path))
    model.set_classes(labels)
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    if args.opset is not None:
        export_kwargs["opset"] = args.opset

    exported_path = Path(model.export(**export_kwargs))

    if exported_path.resolve() != output_path:
        if output_path.exists():
            output_path.unlink()
        exported_path.replace(output_path)

    print(f"Embedded {len(labels)} labels into YOLO-World ONNX")
    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
