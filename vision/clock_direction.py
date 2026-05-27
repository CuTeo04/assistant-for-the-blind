import math


def compute_clock_direction(
    u: float,
    v: float,
    image_width: int,
    image_height: int,
    principal_cx: float | None = None,
    principal_cy: float | None = None,
    dead_zone_px: float = 0.0,
) -> int:
    center_x = principal_cx if principal_cx is not None else image_width / 2.0
    center_y = principal_cy if principal_cy is not None else image_height / 2.0

    dx = float(u) - center_x
    dy = float(v) - center_y

    if dead_zone_px > 0 and math.hypot(dx, dy) < dead_zone_px:
        return 12

    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0

    clock = int(round(angle_deg / 30.0)) % 12
    return 12 if clock == 0 else clock


def format_clock_label(clock: int | None) -> str:
    if clock is None:
        return ""
    return f"hướng {int(clock)} giờ"
