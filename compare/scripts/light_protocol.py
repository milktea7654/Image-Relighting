from __future__ import annotations

import math


DIRECTION_TO_PAN = {
    "S": 0.0,
    "SE": 45.0,
    "E": 90.0,
    "NE": 135.0,
    "N": 180.0,
    "NW": 225.0,
    "W": 270.0,
    "SW": 315.0,
}


def clean_float(value: float, eps: float = 1e-6, digits: int = 8) -> float:
    if abs(value) < eps:
        return 0.0
    return float(f"{value:.{digits}f}")


def kelvin_to_rgb(kelvin: int) -> list[float]:
    temp = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if temp <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19.0 else 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        blue = 255.0
    return [clean_float(max(0.0, min(255.0, v)) / 255.0) for v in (red, green, blue)]


def deepintrinsic_light_code(pan_deg: float, tilt_deg: float, rgb: list[float]) -> list[float]:
    pan = math.radians(float(pan_deg))
    tilt = math.radians(float(tilt_deg))
    code = [
        math.cos(pan) * 0.5 + 0.5,
        math.sin(pan) * 0.5 + 0.5,
        math.cos(tilt),
        math.sin(tilt),
        *rgb,
    ]
    return [clean_float(v) for v in code]


def stage4_point_light_position(pan_deg: float, tilt_deg: float, radius: float = 2.0) -> list[float]:
    """
    Convert relighting dataset pan/tilt into Stage4 camera coordinates.

    Stage4 camera convention:
      - camera is at origin
      - x is image right
      - y is image up
      - camera looks along -Z

    RSR/VIDIT light convention:
      - tilt=0 is the center/front light
      - pan=90 is right
      - pan=180 is up
      - pan=270 is left
      - pan=0 is down
    """
    pan = math.radians(float(pan_deg))
    tilt = math.radians(float(tilt_deg))
    image_plane_radius = radius * math.sin(tilt)
    x = image_plane_radius * math.sin(pan)
    y = -image_plane_radius * math.cos(pan)
    z = -radius * math.cos(tilt)
    return [clean_float(v) for v in (x, y, z)]


def stage4_point_light_rgb(rgb: list[float], scale: float = 40.0, ceiling: float = 120.0) -> list[float]:
    return [clean_float(max(0.0, min(ceiling, float(v) * scale))) for v in rgb]


def light_record(
    *,
    pan: float,
    tilt: float,
    rgb: list[float],
    radius: float = 2.0,
    point_scale: float = 40.0,
    point_ceiling: float = 120.0,
    **extra,
) -> dict:
    rgb = [clean_float(float(v)) for v in rgb]
    return {
        **extra,
        "pan": clean_float(float(pan)),
        "tilt": clean_float(float(tilt)),
        "rgb": rgb,
        "deepintrinsic_light_code": deepintrinsic_light_code(float(pan), float(tilt), rgb),
        "stage4_point_light_position": stage4_point_light_position(float(pan), float(tilt), radius),
        "stage4_point_light_rgb": stage4_point_light_rgb(rgb, point_scale, point_ceiling),
        "stage4_point_radius": float(radius),
        "stage4_point_scale": float(point_scale),
        "coordinate_note": (
            "Stage4 position is camera-space XYZ: x=image-right, y=image-up, camera looks -Z; "
            "pan=90 right, pan=180 up, pan=270 left, pan=0 down, tilt=0 center/front."
        ),
    }
