"""Shared visual conventions for the OmniContact MuJoCo viewers."""

from __future__ import annotations

from math import sqrt
from typing import Any


AXIS_LENGTH = 0.22
AXIS_RADIUS = 0.006
PYRAMID_SIDE = 0.10
PYRAMID_HEIGHT = 0.10


def pyramid_vertices(
    side_m: float = PYRAMID_SIDE,
    height_m: float = PYRAMID_HEIGHT,
) -> str:
    """Return the shared triangular Tracker marker as an MJCF vertex string."""

    side = float(side_m)
    height = float(height_m)
    if side <= 0.0 or height <= 0.0:
        raise ValueError("pyramid dimensions must be positive")
    half = side * 0.5
    y0 = side * sqrt(3.0) / 6.0
    y1 = side * sqrt(3.0) / 3.0
    vertices = (
        (-half, -y0, 0.0),
        (half, -y0, 0.0),
        (0.0, y1, 0.0),
        (0.0, 0.0, -height),
    )
    return " ".join(f"{value:.9g}" for point in vertices for value in point)


def pyramid_faces() -> str:
    """Return outward-facing triangles for :func:`pyramid_vertices`."""

    return "0 1 2  0 3 1  1 3 2  2 3 0"


def axis_geoms_xml(
    prefix: str,
    *,
    length_m: float = AXIS_LENGTH,
    radius_m: float = AXIS_RADIUS,
) -> str:
    """Build the standard red-X, green-Y, blue-Z MJCF frame geometry."""

    length = float(length_m)
    radius = float(radius_m)
    if length <= 0.0 or radius <= 0.0:
        raise ValueError("axis dimensions must be positive")
    return f'''\
      <geom name="{prefix}_axis_x" type="capsule" fromto="0 0 0 {length:.9g} 0 0" size="{radius:.9g}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
      <geom name="{prefix}_axis_y" type="capsule" fromto="0 0 0 0 {length:.9g} 0" size="{radius:.9g}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
      <geom name="{prefix}_axis_z" type="capsule" fromto="0 0 0 0 0 {length:.9g}" size="{radius:.9g}" contype="0" conaffinity="0" rgba="0.10 0.38 1 1"/>'''


def configure_camera(
    camera: Any,
    *,
    lookat: tuple[float, float, float] = (0.8, 0.0, 0.8),
    distance: float = 3.0,
) -> None:
    """Apply the common OmniContact viewpoint to a passive viewer camera."""

    camera.lookat[:] = lookat
    camera.distance = float(distance)
    camera.azimuth = 135.0
    camera.elevation = -18.0
