import cv2
import mediapipe as mp
import math
import config


# =====================================================
# CONFIG CALIBRATION
# =====================================================
KNOWN_DISTANCE_CM = 45.0   # khoảng cách từ camera tới tay (cm)
REAL_HAND_LENGTH_CM = 7.0  # khoảng cách thật giữa landmark 5 → 17 (cm)


# =====================================================
# FIXED RESIZE 1280x960
# =====================================================
def resize_fixed(image):
    return cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LINEAR)


# =====================================================
# MEDIAPIPE INIT
# =====================================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# =====================================================
# DISTANCE 3D (QUAN TRỌNG)
# =====================================================
def calc_distance_3d(p1, p2, width, height):
    # MediaPipe: x, y normalized [0-1], z relative
    x1, y1, z1 = p1.x * width, p1.y * height, p1.z * width
    x2, y2, z2 = p2.x * width, p2.y * height, p2.z * width

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2 +
        (z1 - z2) ** 2
    )


# =====================================================
# LOAD IMAGE
# =====================================================
orig = cv2.imread(config.IMAGE_PATH)

if orig is None:
    print("❌ Không đọc được ảnh")
    exit()

print("\n========== ORIGINAL ==========")
print(f"Resolution: {orig.shape[1]}x{orig.shape[0]}")


# =====================================================
# RESIZE FIXED
# =====================================================
img = resize_fixed(orig)

print("\n========== AFTER RESIZE ==========")
print(f"Resolution: {img.shape[1]}x{img.shape[0]}")


# =====================================================
# DETECT
# =====================================================
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = hands.process(rgb)

print("\n--- Detect ---")

if result.multi_hand_landmarks:

    for hand in result.multi_hand_landmarks:

        mp_draw.draw_landmarks(
            img,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # =====================================================
        # CHỌN LANDMARK 5 → 17 (span lòng bàn tay)
        # =====================================================
        p5 = hand.landmark[5]
        p17 = hand.landmark[17]

        pixel_dist = calc_distance_3d(
            p5, p17,
            img.shape[1],
            img.shape[0]
        )

        # =====================================================
        # CALCULATE FOCAL LENGTH
        # =====================================================
        focal_length = (pixel_dist * KNOWN_DISTANCE_CM) / REAL_HAND_LENGTH_CM

        print(f"Distance (5→17) = {pixel_dist:.2f} px")
        print(f"Focal Length ≈ {focal_length:.2f} px")

else:
    print("❌ Không phát hiện tay")


# =====================================================
# SHOW
# =====================================================
cv2.imshow("Calibration", img)
cv2.waitKey(0)
cv2.destroyAllWindows()