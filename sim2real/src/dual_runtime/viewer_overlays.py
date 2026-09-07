"""Non-physical coordinate frames and robot digits for the native viewer."""

import mujoco
import numpy as np


AXIS_COLORS = ((0.95, 0.08, 0.08, 1), (0.08, 0.90, 0.18, 1), (0.10, 0.38, 1, 1))
ROBOT_COLORS = ((1.0, 0.35, 0.05, 1.0), (0.0, 0.75, 1.0, 1.0))
# Same crossed-plane digits as Dual_G1_MJ's _RobotIndexLabels.
DIGIT_SEGMENTS = (
    (((-0.5, 0.5), (0.5, 0.5)), ((0.5, 0.5), (0.5, -0.5)),
     ((0.5, -0.5), (-0.5, -0.5)), ((-0.5, -0.5), (-0.5, 0.5))),
    (((-0.3, 0.25), (0.0, 0.5)), ((0.0, 0.5), (0.0, -0.5)),
     ((-0.3, -0.5), (0.3, -0.5))),
)


def _geom(scene, kind, position, size, color):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float),
                       np.asarray(position, dtype=float), np.eye(3).ravel(),
                       np.asarray(color, dtype=np.float32))
    scene.ngeom += 1
    return geom


def _stroke(scene, start, end, radius, color, kind):
    geom = _geom(scene, kind, start, (radius, radius, radius), color)
    if geom is not None:
        mujoco.mjv_connector(geom, kind, radius, start, end)


def draw_coordinate_frame(scene, origin, rotation, *, length, radius, origin_radius):
    """Draw local +X/+Y/+Z, with rotation mapping local axes into world space."""
    origin = np.asarray(origin, dtype=float)
    _geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, origin,
          (origin_radius,) * 3, (1, 0.85, 0.1, 1))
    for axis, color in zip(np.asarray(rotation).reshape(3, 3).T, AXIS_COLORS):
        _stroke(scene, origin, origin + length * axis, radius, color,
                mujoco.mjtGeom.mjGEOM_CAPSULE)


def draw_robot_index(scene, root, index):
    center = np.asarray(root) + np.array([0, 0, 1.35])
    up = np.array([0, 0, 1])
    drawn = set()
    for horizontal in (np.array([1, 0, 0]), np.array([0, 1, 0])):
        for start_uv, end_uv in DIGIT_SEGMENTS[index]:
            start = center + 0.38 * (start_uv[0] * horizontal + start_uv[1] * up)
            end = center + 0.38 * (end_uv[0] * horizontal + end_uv[1] * up)
            key = tuple(np.round(np.concatenate((start, end)), 6))
            if key not in drawn:
                drawn.add(key)
                _stroke(scene, start, end, 0.025, ROBOT_COLORS[index],
                        mujoco.mjtGeom.mjGEOM_CYLINDER)
