import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
YOLO_CONFIG_DIR = ROOT_DIR / ".cache" / "ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX")
    parser.add_argument("--weights", default="models/yolo11s.pt", help="Input YOLO weights path")
    parser.add_argument("--output", default="models/YOLO11s.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=960, help="Export image size")
    parser.add_argument("--batch", type=int, default=1, help="Static export batch size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
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
    weights_path = (ROOT_DIR / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)
    output_path = (ROOT_DIR / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    exported_path = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=args.device,
    ))
    if exported_path.resolve() != output_path:
        if output_path.exists():
            output_path.unlink()
        exported_path.replace(output_path)

    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
