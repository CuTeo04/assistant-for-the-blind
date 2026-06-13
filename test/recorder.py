import sounddevice as sd
import numpy as np
from scipy.io import wavfile


DEFAULT_DEVICE_ID = 2
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION_SEC = 3
DEFAULT_SILENCE_DB = -38
DEFAULT_FRAME_MS = 10
DEFAULT_AUDIO_TEMP_FILE = "current_cmd.wav"


def trim_silence(audio_data, threshold_db: float, frame_duration_ms: int, fs: int):
    if len(audio_data) == 0:
        return audio_data

    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)

    frame_length = int(fs * frame_duration_ms / 1000)
    hop_length = frame_length // 2

    rms = []
    for i in range(0, len(audio_float), hop_length):
        frame = audio_float[i : i + frame_length]
        if len(frame) < frame_length // 2:
            break
        rms.append(np.sqrt(np.mean(frame**2)))

    rms = np.array(rms)
    threshold = 10 ** (threshold_db / 20.0)

    speech_frames = np.where(rms > threshold)[0]

    if len(speech_frames) == 0:
        start = int(len(audio_data) * 0.1)
        end = int(len(audio_data) * 0.9)
        return audio_data[start:end]

    start_frame = speech_frames[0]
    end_frame = speech_frames[-1]

    start_sample = max(0, start_frame * hop_length - frame_length)
    end_sample = min(len(audio_data), (end_frame + 1) * hop_length + frame_length)

    return audio_data[start_sample:end_sample]


def record_audio(
    device_id: int,
    duration: int,
    fs: int,
    filename: str,
    silence_db: float,
    frame_ms: int,
):
    print(f"Đang nghe... (hãy nói lệnh trong {duration} giây)")
    try:
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype="int16",
            device=device_id,
        )
        sd.wait()

        trimmed = trim_silence(recording, threshold_db=silence_db, frame_duration_ms=frame_ms, fs=fs)
        wavfile.write(filename, fs, trimmed)

        original_len = len(recording) / fs
        trimmed_len = len(trimmed) / fs
        print(f"Đã cắt im lặng: {original_len:.2f}s --> {trimmed_len:.2f}s")

        return filename

    except Exception as exc:
        print(f"Lỗi ghi âm: {exc}")
        return None
