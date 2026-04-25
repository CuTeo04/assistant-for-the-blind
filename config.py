# ================= CONFIG SYSTEM =================

# ===== IMAGE =====
IMAGE_PATH = "test_room.jpg"

# ===== RESIZE CONTROL =====
MIN_WIDTH = 640     # ảnh nhỏ hơn sẽ upscale
MAX_WIDTH = 1280    # ảnh lớn hơn sẽ downscale

# ===== HAND DISTANCE =====
CALIBRATION_K = 7000   # cần calibrate theo thực tế

# ===== MEDIAPIPE =====
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6

# ===== DISPLAY =====
WINDOW_NAME = "Hand Distance Estimation"
FONT_SCALE = 1
TEXT_COLOR = (0, 255, 0)
TEXT_THICKNESS = 2