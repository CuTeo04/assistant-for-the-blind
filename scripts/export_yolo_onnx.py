import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX")
    parser.add_argument("--weights", default="models/yolo11s.pt", help="Input YOLO weights path")
    parser.add_argument("--imgsz", type=int, default=960, help="Export image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
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
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")

    model = YOLO(str(weights_path))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    print(f"Exported ONNX model to: {exported_path}")


if __name__ == "__main__":
    main()
