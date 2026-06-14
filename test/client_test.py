import argparse
import json
import mimetypes
import os
from pathlib import Path

import requests

from recorder import (
    DEFAULT_AUDIO_TEMP_FILE,
    DEFAULT_DEVICE_ID,
    DEFAULT_DURATION_SEC,
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_DB,
    record_audio,
)


DEFAULT_IMAGE_PATH = "test_room.jpg"
DEFAULT_CALIBRATION_PATH = "camera_calibration.json"


def _load_calibration(path: str) -> dict | None:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    try:
        return json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_calibration(path: str, calibration: dict) -> None:
    payload = {
        "focal_length_px": calibration.get("focal_length_px"),
        "depth_scale": calibration.get("depth_scale"),
        "known_distance_m": calibration.get("known_distance_m"),
    }
    calibration_path = Path(path)
    calibration_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_audio(args) -> str | None:
    audio_path = args.out
    if args.audio_path:
        audio_path = args.audio_path

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        audio_path = record_audio(
            device_id=args.device,
            duration=args.duration,
            fs=args.fs,
            filename=args.out,
            silence_db=args.silence_db,
            frame_ms=args.frame_ms,
        )
    return audio_path


def _build_request_data(args, calibration: dict | None) -> dict:
    data: dict[str, str] = {}

    if args.mode == "setup":
        if args.hand_distance_cm is not None:
            data["hand_distance_cm"] = str(args.hand_distance_cm)
        return data

    if args.focal_length_px is not None:
        data["focal_length_px"] = str(args.focal_length_px)
    elif calibration and calibration.get("focal_length_px") is not None:
        data["focal_length_px"] = str(calibration["focal_length_px"])

    if args.depth_scale is not None:
        data["depth_scale"] = str(args.depth_scale)
    elif calibration and calibration.get("depth_scale") is not None:
        data["depth_scale"] = str(calibration["depth_scale"])

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/process")
    parser.add_argument("--mode", choices=["setup", "normal"], default="normal")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--fs", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--silence-db", type=float, default=DEFAULT_SILENCE_DB)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--out", default=DEFAULT_AUDIO_TEMP_FILE)
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--hand-distance-cm", type=float, default=None)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--calibration-file", default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--no-save-calibration", action="store_true")
    args = parser.parse_args()

    audio_path = _ensure_audio(args)
    if not audio_path:
        print("Không có audio để gửi.")
        return

    image_path = args.image
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"
    calibration = _load_calibration(args.calibration_file)
    data = _build_request_data(args, calibration)

    try:
        with open(audio_path, "rb") as audio_f, open(image_path, "rb") as image_f:
            files = {
                "audio": (os.path.basename(audio_path), audio_f, "audio/wav"),
                "image": (os.path.basename(image_path), image_f, mime_type),
            }
            res = requests.post(args.url, files=files, data=data, timeout=120)

        if not res.ok:
            print(f"Lỗi server: {res.status_code} - {res.text}")
            return

        payload = res.json()
        print(f"API: {payload.get('api')}")
        print(f"Text: {payload.get('text')}")

        camera_calibration = payload.get("camera_calibration")
        if camera_calibration:
            print("Camera calibration:")
            print(json.dumps(camera_calibration, ensure_ascii=False, indent=2))
            if not args.no_save_calibration:
                _save_calibration(args.calibration_file, camera_calibration)
                print(f"Đã lưu calibration vào {args.calibration_file}")
    except FileNotFoundError as exc:
        print(f"Không tìm thấy file: {exc.filename}")
    except Exception as exc:
        print(f"Lỗi không xác định: {exc}")


if __name__ == "__main__":
    main()
