import argparse
import mimetypes
import os
import requests
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from STT.config import voice_config as cfg
from STT.recorder import record_audio


# Hardcode đường dẫn ảnh
IMAGE_PATH = "test_room.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/process")
    parser.add_argument("--device", type=int, default=cfg.DEVICE_ID)
    parser.add_argument("--duration", type=int, default=cfg.DURATION_SEC)
    parser.add_argument("--fs", type=int, default=cfg.SAMPLE_RATE)
    parser.add_argument("--silence-db", type=float, default=cfg.SILENCE_DB)
    parser.add_argument("--frame-ms", type=int, default=cfg.FRAME_MS)
    parser.add_argument("--out", default=cfg.AUDIO_TEMP_FILE)
    args = parser.parse_args()

    audio_path = args.out
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        # Ghi âm
        audio_path = record_audio(
            device_id=args.device,
            duration=args.duration,
            fs=args.fs,
            filename=args.out,
            silence_db=args.silence_db,
            frame_ms=args.frame_ms,
        )

    if not audio_path:
        print("Khong co audio de gui.")
        return

    # Xác định MIME type của ảnh
    mime_type, _ = mimetypes.guess_type(IMAGE_PATH)
    mime_type = mime_type or "image/jpeg"

    try:
        with open(audio_path, "rb") as audio_f, open(IMAGE_PATH, "rb") as image_f:
            files = {
                "audio": (audio_path, audio_f, "audio/wav"),
                "image": ("test_room.jpg", image_f, mime_type),
            }

            res = requests.post(args.url, files=files, timeout=120)

        if res.ok:
            data = res.json()
            print(f"API: {data.get('api')}")
            print(f"Text: {data.get('text')}")
        else:
            print(f"Loi server: {res.status_code} - {res.text}")

    except FileNotFoundError as e:
        print(f"Khong tim thay file: {e.filename}")
    except Exception as e:
        print(f"Loi khong xac dinh: {e}")


if __name__ == "__main__":
    main()