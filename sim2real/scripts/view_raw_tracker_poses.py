#!/usr/bin/env python3
"""Visualize raw OpenVR Tracker poses with two MuJoCo triangular pyramids.

The displayed poses are the values returned by OpenVR in
``TrackingUniverseStanding``.  This script intentionally does not apply
``world_from_steamvr`` or either Tracker mounting transform.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnicontact.perception.openvr_tracker import OpenVRTrackerReader, ViveSample


PYRAMID_SIDE_M = 0.10
PYRAMID_HEIGHT_M = 0.10
TRACKER_AXIS_LENGTH_M = 0.20
WORLD_AXIS_LENGTH_M = 0.50
AXIS_RADIUS_M = 0.006


def _pyramid_vertices(side_m: float, height_m: float) -> str:
    """Return a pyramid whose tip points along the local negative Z axis."""
    half_side = side_m * 0.5
    base_y = side_m * np.sqrt(3.0) / 6.0
    apex_y = side_m * np.sqrt(3.0) / 3.0
    vertices = (
        (-half_side, -base_y, 0.0),
        (half_side, -base_y, 0.0),
        (0.0, apex_y, 0.0),
        (0.0, 0.0, -height_m),
    )
    return " ".join(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)


def _viewer_xml() -> str:
    vertices = _pyramid_vertices(PYRAMID_SIDE_M, PYRAMID_HEIGHT_M)
    # The base and three side faces form a triangular pyramid.  The two mesh
    # copies allow the viewer to use different colors for the two Trackers.
    faces = "0 2 1  0 1 3  1 2 3  2 0 3"
    return f"""
<mujoco model="raw_tracker_pose_viewer">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 0" timestep="0.005"/>
  <visual>
    <headlight ambient="0.52 0.52 0.52"
              diffuse="0.72 0.72 0.72" specular="0.28 0.28 0.28"/>
  </visual>
  <asset>
    <texture name="skybox_texture" type="skybox" builtin="gradient"
             width="512" height="512" rgb1="0.72 0.78 0.88" rgb2="0.36 0.43 0.54"/>
    <texture name="ground_texture" type="2d" builtin="checker"
             width="512" height="512" rgb1="0.58 0.61 0.66" rgb2="0.30 0.33 0.38"
             /><material name="ground_material" texture="ground_texture"
             texrepeat="16 16" reflectance="0.18"/>
    <mesh name="robot_tracker_pyramid" vertex="{vertices}" face="{faces}"/>
    <mesh name="object_tracker_pyramid" vertex="{vertices}" face="{faces}"/>
  </asset>
  <worldbody>
    <light name="key" pos="1 -1 3" dir="-1 1 -3"
           diffuse="1 1 1" specular="0.35 0.35 0.35"/>
    <light name="fill" pos="-2 1 2" dir="2 -1 -2"
           diffuse="0.75 0.80 0.90" specular="0.12 0.12 0.12"/>
    <geom name="reference_plane" type="plane" size="6 6 0.01"
          material="ground_material" contype="0" conaffinity="0"/>
    <!-- World frame at the raw-coordinate origin: X=red, Y=green, Z=blue. -->
    <geom name="world_axis_x" type="capsule" fromto="0 0 0 {WORLD_AXIS_LENGTH_M} 0 0"
          size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
    <geom name="world_axis_y" type="capsule" fromto="0 0 0 0 {WORLD_AXIS_LENGTH_M} 0"
          size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
    <geom name="world_axis_z" type="capsule" fromto="0 0 0 0 0 {WORLD_AXIS_LENGTH_M}"
          size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.10 0.38 1.0 1"/>
    <geom name="world_origin" type="sphere" pos="0 0 0" size="0.018"
          contype="0" conaffinity="0" rgba="1.0 0.85 0.10 1"/>
    <body name="robot_tracker" pos="0 0 0">
      <freejoint name="robot_tracker_free"/>
      <geom name="robot_tracker_geom" type="mesh" mesh="robot_tracker_pyramid"
            contype="0" conaffinity="0" rgba="0.12 0.48 1.0 1"/>
      <!-- Tracker-local frame, rotating with the pyramid. -->
      <geom name="robot_axis_x" type="capsule" fromto="0 0 0 {TRACKER_AXIS_LENGTH_M} 0 0"
            size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
      <geom name="robot_axis_y" type="capsule" fromto="0 0 0 0 {TRACKER_AXIS_LENGTH_M} 0"
            size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
      <geom name="robot_axis_z" type="capsule" fromto="0 0 0 0 0 {TRACKER_AXIS_LENGTH_M}"
            size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.10 0.38 1.0 1"/>
    </body>
    <body name="object_tracker" pos="0 0 0">
      <freejoint name="object_tracker_free"/>
      <geom name="object_tracker_geom" type="mesh" mesh="object_tracker_pyramid"
            contype="0" conaffinity="0" rgba="1.0 0.38 0.10 1"/>
      <!-- Tracker-local frame, rotating with the pyramid. -->
      <geom name="object_axis_x" type="capsule" fromto="0 0 0 {TRACKER_AXIS_LENGTH_M} 0 0"
            size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.95 0.08 0.08 1"/>
      <geom name="object_axis_y" type="capsule" fromto="0 0 0 0 {TRACKER_AXIS_LENGTH_M} 0"
            size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.08 0.90 0.18 1"/>
      <geom name="object_axis_z" type="capsule" fromto="0 0 0 0 0 {TRACKER_AXIS_LENGTH_M}"
            size="{AXIS_RADIUS_M}" contype="0" conaffinity="0" rgba="0.10 0.38 1.0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _free_joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"MuJoCo model is missing free joint {name!r}")
    return int(model.jnt_qposadr[joint_id])


def _set_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_address: int,
    geom_name: str,
    sample: ViveSample | None,
) -> bool:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise RuntimeError(f"MuJoCo model is missing geom {geom_name!r}")

    if sample is None:
        # Keep the last pose in qpos but make the marker translucent when the
        # corresponding OpenVR device is disconnected or its pose is invalid.
        model.geom_rgba[geom_id, 3] = 0.12
        return False

    model.geom_rgba[geom_id, 3] = 1.0
    data.qpos[qpos_address : qpos_address + 3] = (
        sample.line_x_m,
        sample.line_y_m,
        sample.line_z_m,
    )
    # ViveSample stores xyzw; MuJoCo free-joint qpos stores wxyz.
    data.qpos[qpos_address + 3 : qpos_address + 7] = (
        sample.qw,
        sample.qx,
        sample.qy,
        sample.qz,
    )
    return True


def _choose_serials(
    devices: dict[str, int],
    robot_serial: str | None,
    object_serial: str | None,
) -> tuple[str, str]:
    available = set(devices)
    if robot_serial is not None and robot_serial not in available:
        raise RuntimeError(f"robot Tracker {robot_serial!r} was not found")
    if object_serial is not None and object_serial not in available:
        raise RuntimeError(f"object Tracker {object_serial!r} was not found")

    if robot_serial is None or object_serial is None:
        remaining = sorted(available - {serial for serial in (robot_serial, object_serial) if serial})
        if robot_serial is None:
            if not remaining:
                raise RuntimeError("at least two GenericTrackers are required")
            robot_serial = remaining.pop(0)
        if object_serial is None:
            if not remaining:
                raise RuntimeError("at least two GenericTrackers are required")
            object_serial = remaining.pop(0)

    if robot_serial == object_serial:
        raise RuntimeError("robot and object Tracker serials must differ")
    return robot_serial, object_serial


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw OpenVR Tracker poses as 10 cm triangular pyramids"
    )
    parser.add_argument(
        "--robot-serial",
        default=None,
        help="hardware serial for the blue robot Tracker; defaults to the first detected Tracker",
    )
    parser.add_argument(
        "--object-serial",
        default=None,
        help="hardware serial for the orange object Tracker; defaults to the second detected Tracker",
    )
    parser.add_argument("--fps", type=float, default=90.0)
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=None,
        help="camera distance in meters; defaults to an automatic distance framing the world origin and Trackers",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.camera_distance is not None and args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reader = OpenVRTrackerReader()
    try:
        devices = reader.start()
        robot_serial, object_serial = _choose_serials(
            devices,
            args.robot_serial,
            args.object_serial,
        )
        reader.required_serials = (robot_serial, object_serial)
        print("Raw OpenVR Tracker viewer")
        print(f"  robot  (blue):   {robot_serial}")
        print(f"  object (orange): {object_serial}")
        print(
            "  coordinates: OpenVR TrackingUniverseStanding; "
            "no world calibration or mounting transform is applied"
        )
        print(
            f"  pyramid: side={PYRAMID_SIDE_M:.3f} m, height={PYRAMID_HEIGHT_M:.3f} m"
        )

        model = mujoco.MjModel.from_xml_string(_viewer_xml())
        data = mujoco.MjData(model)
        robot_qpos = _free_joint_qpos_address(model, "robot_tracker_free")
        object_qpos = _free_joint_qpos_address(model, "object_tracker_free")
        mujoco.mj_forward(model, data)

        frame_period = 1.0 / args.fps
        last_log = time.monotonic()
        last_valid = (False, False)
        try:
            with mujoco.viewer.launch_passive(
                model,
                data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                # OpenVR's raw Standing coordinates can be far from the
                # MuJoCo origin (including negative Z).  Frame the camera once
                # on the first valid sample; afterwards the camera stays fixed
                # so Tracker translation remains visible in the scene.
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0
                camera_initialized = False

                while viewer.is_running():
                    loop_start = time.monotonic()
                    samples = reader.read_all((robot_serial, object_serial))
                    robot_sample = samples.get(robot_serial)
                    object_sample = samples.get(object_serial)
                    robot_valid = _set_body_pose(
                        model,
                        data,
                        robot_qpos,
                        "robot_tracker_geom",
                        robot_sample,
                    )
                    object_valid = _set_body_pose(
                        model,
                        data,
                        object_qpos,
                        "object_tracker_geom",
                        object_sample,
                    )
                    mujoco.mj_forward(model, data)

                    if (
                        not camera_initialized
                        and (robot_sample is not None or object_sample is not None)
                    ):
                        positions = [
                            np.array(
                                [sample.line_x_m, sample.line_y_m, sample.line_z_m],
                                dtype=np.float64,
                            )
                            for sample in (robot_sample, object_sample)
                            if sample is not None
                        ]
                        # Include the world origin so the world frame and the
                        # Tracker pair are visible in one camera view.
                        frame_positions = positions + [np.zeros(3, dtype=np.float64)]
                        bounds_min = np.min(frame_positions, axis=0)
                        bounds_max = np.max(frame_positions, axis=0)
                        center = 0.5 * (bounds_min + bounds_max)
                        viewer.cam.lookat[:] = center
                        if args.camera_distance is None:
                            radius = max(
                                float(np.linalg.norm(position - center))
                                for position in frame_positions
                            )
                            viewer.cam.distance = max(0.75, 2.2 * radius)
                        elif not camera_initialized:
                            viewer.cam.distance = args.camera_distance
                        camera_initialized = True
                        print(
                            "Camera framed once at raw Tracker coordinates: "
                            f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}], "
                            f"distance={viewer.cam.distance:.3f} m; camera translation is fixed"
                        )

                    viewer.sync()

                    now = time.monotonic()
                    if (robot_valid, object_valid) != last_valid or now - last_log >= 1.0:
                        robot_text = (
                            "valid" if robot_valid else "invalid/disconnected"
                        )
                        object_text = (
                            "valid" if object_valid else "invalid/disconnected"
                        )
                        print(f"robot={robot_text}, object={object_text}")
                        if robot_sample is not None:
                            print(
                                "  robot raw xyz="
                                f"[{robot_sample.line_x_m:.3f}, {robot_sample.line_y_m:.3f}, "
                                f"{robot_sample.line_z_m:.3f}]"
                            )
                        if object_sample is not None:
                            print(
                                "  object raw xyz="
                                f"[{object_sample.line_x_m:.3f}, {object_sample.line_y_m:.3f}, "
                                f"{object_sample.line_z_m:.3f}]"
                            )
                        last_valid = (robot_valid, object_valid)
                        last_log = now

                    remaining = frame_period - (time.monotonic() - loop_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            print("\nRaw Tracker viewer stopped")
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
