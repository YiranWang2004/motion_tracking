#!/usr/bin/env python3
"""Two-G1 MuJoCo bridge for the production ScaleBFM residual deploy entry."""

from __future__ import annotations

import argparse
import json
import signal
import os
import select
import sys
from collections import deque
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from common.udp_latest import LatestPacket
from common.udp_transport import UDPRobotLow
from dual_runtime.constants import POLICY_JOINT_NAMES
from dual_runtime.reference import (_yaw_from_wxyz, _yaw_quaternion,
                                    quat_apply_batch, quat_mul_left_batch)
from dual_runtime.sim_control import load_default_command


ROOT = Path(__file__).resolve().parents[1]
BUTTONS = ("start", "stop", "A", "B", "up", "down")
STICKS = ("lx", "ly", "rx", "ry")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _object_id(model: mujoco.MjModel, kind: Any, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo model is missing {name!r}")
    return int(value)


def _wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(4)[[1, 2, 3, 0]]


@dataclass(frozen=True)
class RobotBinding:
    root_qpos: int
    root_dof: int
    joint_qpos: np.ndarray
    joint_dof: np.ndarray
    actuators: np.ndarray
    gyro_sensor: tuple[int, int] | None
    linacc_sensor: tuple[int, int] | None


@dataclass(frozen=True)
class SimCommand:
    q_des: np.ndarray
    qd_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    enable: int
    state_time_ns: int
    phase: str = "zero_torque"
    frame: int = -1
    reference_position: tuple[float, float, float] | None = None


class DualScaleBFMSim2Sim:
    """Lockstep two-bridge replacement backed by one shared MuJoCo snapshot."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        headless: bool = False,
        scalebfm_only: bool = False,
        reference_bundle: str | Path | None = None,
        initial_scene=None,
        transports: tuple[Any, Any] | None = None,
    ) -> None:
        self.object_enabled = not scalebfm_only
        self.config_path = Path(config_path).expanduser().resolve()
        self.raw = _load_yaml(self.config_path)
        sim = self.raw.get("simulation")
        if not isinstance(sim, dict):
            raise ValueError("config is missing simulation section")
        self.sim = sim
        self.physical_hz = int(sim.get("physical_frequency_hz", 200))
        self.policy_hz = float(self.raw.get("control_frequency_hz", 50.0))
        ratio = self.physical_hz / self.policy_hz
        self.decimation = int(round(ratio))
        if (
            self.physical_hz <= 0
            or self.policy_hz <= 0
            or not np.isclose(ratio, self.decimation)
        ):
            raise ValueError(
                "physical_frequency_hz must be an integer multiple of control_frequency_hz"
            )
        self.lockstep_timeout_s = float(sim.get("lockstep_timeout_s", 1.0))
        self.startup_timeout_s = float(sim.get("startup_timeout_s", 60.0))
        self.command_retry_s = float(sim.get("state_retry_s", 0.05))
        termination = self.raw.get("task_safety", {})
        self.object_position_z_error_m = float(
            termination.get("object_position_z_error_m", 0.10)
        )
        self.object_position_xyz_error_m = float(
            termination.get("object_position_xyz_error_m", 0.30)
        )
        if (
            min(
                self.lockstep_timeout_s,
                self.startup_timeout_s,
                self.command_retry_s,
                self.object_position_z_error_m,
                self.object_position_xyz_error_m,
            )
            <= 0.0
        ):
            raise ValueError(
                "simulation timeouts and termination thresholds must be positive"
            )

        xml_path = _resolve(self.config_path.parent, sim["xml_path"])
        self.box_half_extents = np.asarray(
            sim.get("box_half_extents", (0.30, 0.15, 0.15)), dtype=np.float32
        ).reshape(3)
        if not np.all(np.isfinite(self.box_half_extents)) or np.any(self.box_half_extents <= 0):
            raise ValueError("box_half_extents must be finite and positive")
        # Compile the candidate dimensions so collision bounds and inertia agree.
        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.geom("box_collision").size = self.box_half_extents.astype(np.float64)
        box = spec.body("box")
        half = self.box_half_extents.astype(np.float64)
        box.inertia = box.mass / 3.0 * np.array([
            half[1]**2 + half[2]**2, half[0]**2 + half[2]**2, half[0]**2 + half[1]**2,
        ])
        if not self.object_enabled:
            # Keep an inert pose placeholder for the common telemetry schema;
            # no box geometry, collision, contact load, or visible object remains.
            spec.delete(spec.geom("box_collision"))
            box.gravcomp = 1.0
        self.model = spec.compile()
        self.model.opt.timestep = 1.0 / self.physical_hz
        self.data = mujoco.MjData(self.model)
        self.bindings = (self._bind_robot("a_"), self._bind_robot("b_"))
        dynamics_xml = sim.get("joint_dynamics_xml")
        if dynamics_xml is not None:
            self._load_joint_dynamics(_resolve(self.config_path.parent, dynamics_xml))
        if self.model.nu != 2 * len(POLICY_JOINT_NAMES):
            raise ValueError(f"dual model must have 58 actuators, got {self.model.nu}")

        self.box_body = _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_joint = _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self.box_qpos = int(self.model.jnt_qposadr[box_joint])
        self.box_dof = int(self.model.jnt_dofadr[box_joint])
        torque = sim.get("torque_limits")
        if torque is None:
            metadata_path = _resolve(
                self.config_path.parent,
                Path(self.raw["artifacts"]["directory"])
                / self.raw["artifacts"]["scalebfm_metadata"],
            )
            torque = json.loads(metadata_path.read_text(encoding="utf-8"))[
                "torque_limit"
            ]
        self.torque_limits = np.asarray(torque, dtype=np.float64).reshape(29)
        if np.any(self.torque_limits <= 0.0):
            raise ValueError("simulation torque limits must be positive")

        reference_path = (
            Path(reference_bundle).expanduser().resolve()
            if reference_bundle is not None
            else _resolve(
                self.config_path.parent,
                Path(self.raw["artifacts"]["directory"])
                / self.raw["artifacts"]["reference_bundle"],
            )
        )
        self.start_frame = int(self.raw.get("start_frame", 1))
        self._initialize_from_reference(reference_path, self.start_frame)
        defaults = load_default_command(
            _resolve(
                self.config_path.parent, self.raw.get("control", {}).get("standing_asset_dir", "omnicontact")
            )
        )
        for binding in self.bindings:
            self.data.qpos[binding.joint_qpos] = defaults.target_pos
            self.data.qpos[binding.root_qpos + 2] = float(
                sim.get("initial_root_height_m", 0.793)
            )
        if initial_scene is not None:
            self._initialize_from_scene(initial_scene)
        mujoco.mj_forward(self.model, self.data)
        self._ghost_data = mujoco.MjData(self.model)
        self._ghost_visible = True
        self._ghost_alignment = None
        self._ghost_frame = self.start_frame
        self._phase = "zero_torque"
        self._policy_frame = -1
        self._command_reference_position = None
        self._roots_released = False
        self._task_started = False
        self._key_queue = deque()
        self._key_lock = threading.Lock()
        self._button_snapshot = {name: False for name in BUTTONS}
        self._button_snapshot_id = -1
        self._last_key_snapshot = -2
        self._initial_root_qpos = tuple(
            self.data.qpos[binding.root_qpos : binding.root_qpos + 7].copy()
            for binding in self.bindings
        )
        self._initial_box_qpos = self.data.qpos[
            self.box_qpos : self.box_qpos + 7
        ].copy()

        robot_cfg = sim.get("robots")
        if not isinstance(robot_cfg, dict):
            raise ValueError("simulation.robots must contain robot_a and robot_b")
        self._condition = threading.Condition()
        self._commands: list[SimCommand | None] = [None, None]
        self._connected = [False, False]
        if transports is None:
            self.transports = tuple(
                UDPRobotLow(
                    self._low_udp_config(robot_cfg[key]["udp"]),
                    on_command_packet=self._handler(index),
                )
                for index, key in enumerate(("robot_a", "robot_b"))
            )
        else:
            self.transports = transports
        self._alive = True
        self._snapshot_id = 0
        self._headless = bool(headless)
        self._viewer = None

    @staticmethod
    def _low_udp_config(high: dict[str, Any]) -> dict[str, Any]:
        return {
            "state_host": high["state_bind_host"],
            "state_port": int(high["state_port"]),
            "cmd_bind_host": high["cmd_host"],
            "cmd_port": int(high["cmd_port"]),
        }

    def _load_joint_dynamics(self, xml_path: Path) -> None:
        """Use the single-G1 passive joints with its unchanged LocoMode gains.

        This is a fixed property of the entire simulation, not a phase-dependent
        dynamics switch. Collision geometry and actuator torque caps are retained.
        """
        source = mujoco.MjModel.from_xml_path(str(xml_path))
        source_dofs = [
            int(source.jnt_dofadr[_object_id(source, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in POLICY_JOINT_NAMES
        ]
        for binding in self.bindings:
            for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
                getattr(self.model, field)[binding.joint_dof] = getattr(source, field)[
                    source_dofs
                ]
        print(f"[sim2sim] joint passive dynamics from {xml_path}", flush=True)

    def _bind_robot(self, prefix: str) -> RobotBinding:
        root = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
        )
        joint_ids = np.asarray(
            [
                _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
                for name in POLICY_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        actuator_ids = []
        for joint_id, name in zip(joint_ids, POLICY_JOINT_NAMES):
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                raise ValueError(
                    f"joint {prefix + name!r} must have exactly one actuator"
                )
            actuator_ids.append(int(matches[0]))
        actuator_ids = np.asarray(actuator_ids, dtype=np.int32)

        def sensor_slice(names: tuple[str, ...]) -> tuple[int, int] | None:
            for name in names:
                sensor_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + name
                )
                if sensor_id >= 0:
                    return (
                        int(self.model.sensor_adr[sensor_id]),
                        int(self.model.sensor_dim[sensor_id]),
                    )
            return None

        return RobotBinding(
            root_qpos=int(self.model.jnt_qposadr[root]),
            root_dof=int(self.model.jnt_dofadr[root]),
            joint_qpos=np.asarray(self.model.jnt_qposadr[joint_ids], dtype=np.int32),
            joint_dof=np.asarray(self.model.jnt_dofadr[joint_ids], dtype=np.int32),
            actuators=actuator_ids,
            gyro_sensor=sensor_slice(("imu_ang_vel", "imu-torso-angular-velocity")),
            linacc_sensor=sensor_slice(
                ("imu_lin_acc", "imu-torso-linear-acceleration")
            ),
        )

    def _initialize_from_reference(self, path: Path, frame: int) -> None:
        with np.load(path, allow_pickle=False) as data:
            order = tuple(str(value) for value in data["training_joint_order"].tolist())
            if order != tuple(POLICY_JOINT_NAMES):
                raise ValueError(
                    "reference joint order does not match deployment policy order"
                )
            frames = int(data["training_object_body_pos_w"].shape[0])
            if not 0 <= frame < frames:
                raise ValueError("start_frame is outside the reference bundle")
            reference_extents = np.asarray(
                data["training_box_half_extents"], dtype=np.float32
            )
            if self.object_enabled and not np.allclose(reference_extents, self.box_half_extents, atol=1e-6):
                raise ValueError("simulation box size does not match reference bundle")
            self._ghost_roots = np.stack([
                np.concatenate((data[f"training_robot_{i}_body_pos_w"][:, 0],
                                data[f"training_robot_{i}_body_quat_w"][:, 0]), axis=-1)
                for i in range(2)])
            self._ghost_joints = np.stack([data[f"training_robot_{i}_joint_pos"] for i in range(2)])
            self._ghost_box_quat = data["training_object_body_quat_w"].copy()
            self._reference_object_position = np.asarray(
                data["training_object_body_pos_w"], dtype=np.float64
            ).copy()
            for index, binding in enumerate(self.bindings):
                position = data[f"training_robot_{index}_body_pos_w"][frame, 0]
                quaternion = data[f"training_robot_{index}_body_quat_w"][frame, 0]
                self.data.qpos[binding.root_qpos : binding.root_qpos + 7] = (
                    np.concatenate((position, quaternion))
                )
                self.data.qpos[binding.joint_qpos] = data[
                    f"training_robot_{index}_joint_pos"
                ][frame]
            self.data.qpos[self.box_qpos : self.box_qpos + 7] = np.concatenate(
                (
                    data["training_object_body_pos_w"][frame],
                    data["training_object_body_quat_w"][frame],
                )
            )
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _initialize_from_scene(self, snapshot) -> None:
        if self.object_enabled and not np.allclose(
            snapshot.object.half_extents, self.box_half_extents, rtol=0, atol=1e-6
        ):
            raise ValueError("Vive box size does not match simulation/reference box size")
        entries = [(binding.root_qpos, pose) for binding, pose in zip(
            self.bindings, (snapshot.robot_a, snapshot.robot_b))]
        if self.object_enabled:
            entries.append((self.box_qpos, snapshot.object))
        for address, pose in entries:
            position = np.asarray(pose.position_w, dtype=float).reshape(3)
            quat = np.asarray(pose.quaternion_xyzw, dtype=float).reshape(4)
            norm = np.linalg.norm(quat)
            if not np.all(np.isfinite(position)) or not np.isfinite(norm) or norm < 1e-8:
                raise ValueError("invalid Vive initial scene pose")
            self.data.qpos[address:address + 7] = np.concatenate(
                (position, (quat / norm)[[3, 0, 1, 2]]))
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        print("[sim2sim] captured Vive scene: "
              f"A={snapshot.robot_a.position_w.tolist()}, "
              f"B={snapshot.robot_b.position_w.tolist()}, "
              f"box={snapshot.object.position_w.tolist() if self.object_enabled else 'disabled'}; "
              "joints=DefaultPose; live tracking closed", flush=True)

    def _reference_frame(self) -> int:
        return min(
            max(self.start_frame, self._policy_frame),
            self._reference_object_position.shape[0] - 1,
        )

    def task_termination_reason(self) -> str | None:
        if not self.object_enabled or self._phase != "executing":
            return None
        frame = self._reference_frame()
        actual = self.data.xpos[self.box_body]
        reference = (
            self._reference_object_position[frame]
            if self._command_reference_position is None
            else np.asarray(self._command_reference_position)
        )
        delta = actual - reference
        z_error = abs(float(delta[2]))
        xyz_error = float(np.linalg.norm(delta))
        if z_error > self.object_position_z_error_m:
            return (
                f"object.position_z frame={frame} error={z_error:.3f}m "
                f"threshold={self.object_position_z_error_m:.3f}m"
            )
        if xyz_error > self.object_position_xyz_error_m:
            return (
                f"object.position_xyz frame={frame} error={xyz_error:.3f}m "
                f"threshold={self.object_position_xyz_error_m:.3f}m"
            )
        return None

    def _handler(self, robot_index: int):
        def receive(packet: LatestPacket) -> None:
            try:
                command = self.parse_command(packet.data)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                print(f"[sim2sim] ignored malformed robot {robot_index} command: {exc}")
                return
            with self._condition:
                self._commands[robot_index] = command
                self._connected[robot_index] = True
                self._condition.notify_all()

        return receive

    @staticmethod
    def parse_command(payload: Any) -> SimCommand:
        if not isinstance(payload, dict):
            raise TypeError("command must be a mapping")
        values = {}
        for name in ("q_des", "qd_des", "kp", "kd"):
            value = np.asarray(payload[name], dtype=np.float64).reshape(-1)
            if value.shape != (29,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 29-vector")
            values[name] = value.copy()
        if np.any(values["kp"] < 0.0) or np.any(values["kd"] < 0.0):
            raise ValueError("PD gains must be non-negative")
        enable = int(payload["enable"])
        if enable not in (0, 1):
            raise ValueError("enable must be 0 or 1")
        control = payload.get("extra_command", {}).get(
            "dual_sim_control", {"phase": "stopped", "frame": -1}
        )
        phase = control["phase"]
        if phase not in {
            "zero_torque",
            "default_pose",
            "loco_standing",
            "scalebfm_standing",
            "executing",
            "stopped",
        }:
            raise ValueError(f"invalid simulation phase: {phase}")
        frame = int(control.get("frame", -1))
        if phase == "executing" and frame < 0:
            raise ValueError("executing command requires a reference frame")
        reference = control.get("reference_position_w")
        if reference is not None:
            reference = np.asarray(reference, dtype=float).reshape(3)
            if not np.all(np.isfinite(reference)):
                raise ValueError("invalid reference position")
            reference = tuple(reference.tolist())
        return SimCommand(
            **values,
            phase=phase,
            frame=frame,
            reference_position=reference,
            enable=enable,
            state_time_ns=int(payload["state_receive_time_ns"]),
        )

    def _robot_state(self, binding: RobotBinding) -> tuple[np.ndarray, ...]:
        q = self.data.qpos[binding.joint_qpos].copy().astype(np.float32)
        dq = self.data.qvel[binding.joint_dof].copy().astype(np.float32)
        quat = self.data.qpos[binding.root_qpos + 3 : binding.root_qpos + 7].copy()
        gyro = np.zeros(3, dtype=np.float32)
        if binding.gyro_sensor is not None and binding.gyro_sensor[1] >= 3:
            address = binding.gyro_sensor[0]
            gyro = self.data.sensordata[address : address + 3].copy().astype(np.float32)
        linacc = np.zeros(3, dtype=np.float32)
        if binding.linacc_sensor is not None and binding.linacc_sensor[1] >= 3:
            address = binding.linacc_sensor[0]
            linacc = (
                self.data.sensordata[address : address + 3].copy().astype(np.float32)
            )
        return q, dq, quat.astype(np.float32), gyro.astype(np.float32), linacc

    def pose_payload(self) -> dict[str, Any]:
        robots = []
        for binding in self.bindings:
            root = self.data.qpos[binding.root_qpos : binding.root_qpos + 7]
            robots.append(
                {
                    "position_w": root[:3].copy().astype(np.float32),
                    "quaternion_xyzw": _wxyz_to_xyzw(root[3:7]),
                }
            )
        box_velocity = self.data.qvel[self.box_dof : self.box_dof + 6]
        return {
            "object_enabled": self.object_enabled,
            "snapshot_id": self._snapshot_id,
            "simulation_time_s": float(self.data.time),
            "robots": robots,
            "object": {
                "position_w": self.data.xpos[self.box_body].copy().astype(np.float32),
                "quaternion_xyzw": _wxyz_to_xyzw(self.data.xquat[self.box_body]),
                "half_extents": self.box_half_extents.copy(),
                "linear_velocity_w": box_velocity[:3].copy().astype(np.float32),
                "angular_velocity_w": box_velocity[3:].copy().astype(np.float32),
            },
        }

    def key_callback(self, keycode: int) -> None:
        key = chr(keycode).lower()
        if key == "g":
            self._ghost_visible = not self._ghost_visible
            print(f"[sim2sim] reference ghosts {'on' if self._ghost_visible else 'off'}", flush=True)
            return
        button = {"s": "start", "b": "B", "a": "A", "x": "stop"}.get(key)
        if button is not None:
            if self._snapshot_id == 0 and button != "stop":
                print(f"[sim2sim] {key.upper()} ignored before initial control handshake; "
                      "wait for ZERO TORQUE, then press S again", flush=True)
                return
            with self._key_lock:
                # Stop cannot be delayed behind a queued startup sequence.
                if button == "stop":
                    self._key_queue.clear()
                self._key_queue.append(button)

    def _read_terminal_keys(self) -> None:
        # Avoid a daemon blocked inside buffered stdin during interpreter exit.
        try:
            descriptor = sys.stdin.fileno()
            while self._alive:
                ready, _, _ = select.select([descriptor], [], [], 0.1)
                if not ready:
                    continue
                chunk = os.read(descriptor, 1024)
                if not chunk:
                    return
                for key in chunk.decode("utf-8", errors="ignore"):
                    self.key_callback(ord(key))
        except (OSError, ValueError):
            return

    def publish_state_pair(self, state_time_ns: int) -> None:
        # Retries of the same snapshot must have exactly the same buttons.
        # Insert a false tick between queued presses, including repeated keys.
        if self._button_snapshot_id != self._snapshot_id:
            self._button_snapshot = {name: False for name in BUTTONS}
            with self._key_lock:
                if self._key_queue and (
                    self._key_queue[0] == "stop"
                    or self._snapshot_id > self._last_key_snapshot + 1
                ):
                    self._button_snapshot[self._key_queue.popleft()] = True
                    self._last_key_snapshot = self._snapshot_id
            self._button_snapshot_id = self._snapshot_id
        pose = self.pose_payload()
        for transport, binding in zip(self.transports, self.bindings):
            q, dq, quat, gyro, linacc = self._robot_state(binding)
            transport.send_state(
                q=q,
                dq=dq,
                quat_wxyz=quat,
                gyro=gyro,
                linacc=linacc,
                buttons=self._button_snapshot.copy(),
                sticks={name: 0.0 for name in STICKS},
                extra_state={"dual_sim_pose": pose},
                state_receive_time_ns=state_time_ns,
            )

    def command_pair_ready(self, state_time_ns: int) -> bool:
        return all(
            command is not None and command.state_time_ns == state_time_ns
            for command in self._commands
        )

    def wait_for_command_pair(
        self, state_time_ns: int
    ) -> tuple[SimCommand, SimCommand] | None:
        timeout = (
            self.lockstep_timeout_s if all(self._connected) else self.startup_timeout_s
        )
        deadline = time.monotonic() + timeout
        while self._alive:
            with self._condition:
                if self.command_pair_ready(state_time_ns):
                    return self._commands[0], self._commands[1]  # type: ignore[return-value]
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(min(self.command_retry_s, remaining))
            if self._viewer is not None:
                self._draw_status()
                self._viewer.sync()
                if not self._viewer.is_running():
                    self.close()
                    return None
            if self._alive and not self.command_pair_ready(state_time_ns):
                self.publish_state_pair(state_time_ns)
        if not self._alive:
            return None
        raise RuntimeError(
            "timed out waiting for A/B commands corresponding to the same simulation snapshot"
        )

    def apply_commands(self, commands: tuple[SimCommand, SimCommand]) -> None:
        for binding, command in zip(self.bindings, commands):
            q = self.data.qpos[binding.joint_qpos]
            dq = self.data.qvel[binding.joint_dof]
            if command.phase == "zero_torque":
                torque = np.zeros(29)
            elif command.enable:
                torque = command.kp * (command.q_des - q) + command.kd * (
                    command.qd_des - dq
                )
            else:
                torque = -command.kd * dq
            self.data.ctrl[binding.actuators] = np.clip(
                torque, -self.torque_limits, self.torque_limits
            )

    def step_policy_interval(self, commands: tuple[SimCommand, SimCommand]) -> None:
        a, b = commands
        if (a.phase, a.frame, a.reference_position) != (
            b.phase,
            b.frame,
            b.reference_position,
        ):
            raise RuntimeError(
                "A/B simulation control phases or reference frames disagree"
            )
        self._phase, self._policy_frame = a.phase, a.frame
        self._command_reference_position = a.reference_position
        if not self._roots_released and a.phase == "executing":
            raise RuntimeError("task command received before paired standing-policy release")
        if a.phase == "stopped":
            self.apply_commands(commands)
            return
        if a.phase == "executing":
            if not self._task_started:
                self._ghost_alignment = self._reference_preview_alignment()
            self._ghost_frame = a.frame
            self._task_started = True
        if a.phase in {"loco_standing", "scalebfm_standing"}:
            self._roots_released = True
        lock_roots = not self._roots_released
        if a.phase == "zero_torque":
            # Like single-robot sim2sim: no physics drift before Start.
            self.data.ctrl[:] = 0.0
            return
        for _ in range(self.decimation):
            # MuJoCo advances at 200 Hz while the policy produces targets at
            # 50 Hz. Recompute PD from the latest q/dq on every physics step,
            # matching the training environment and the OmniContact sim2sim
            # controller. Holding one torque for the whole decimation interval
            # introduces a stale-velocity/position impulse at every policy tick.
            self.apply_commands(commands)
            mujoco.mj_step(self.model, self.data)
            if lock_roots:
                for binding, root_qpos in zip(self.bindings, self._initial_root_qpos):
                    self.data.qpos[binding.root_qpos : binding.root_qpos + 7] = (
                        root_qpos
                    )
                    self.data.qvel[binding.root_dof : binding.root_dof + 6] = 0.0
                self.data.qpos[self.box_qpos : self.box_qpos + 7] = (
                    self._initial_box_qpos
                )
                self.data.qvel[self.box_dof : self.box_dof + 6] = 0.0
                mujoco.mj_forward(self.model, self.data)

    def _run_loop(self, max_policy_steps: int | None) -> None:
        steps = 0
        while self._alive and (max_policy_steps is None or steps < max_policy_steps):
            if self._viewer is not None and not self._viewer.is_running():
                break
            state_time_ns = time.perf_counter_ns()
            self.publish_state_pair(state_time_ns)
            commands = self.wait_for_command_pair(state_time_ns)
            if commands is None:
                break
            self.step_policy_interval(commands)
            if self._phase == "stopped":
                break
            # Task tracking guards run in the shared deployment coordinator.
            self._snapshot_id += 1
            steps += 1
            if steps % max(1, int(round(self.policy_hz))) == 0:
                root_z = [
                    float(self.data.qpos[binding.root_qpos + 2])
                    for binding in self.bindings
                ]
                box_z = float(self.data.xpos[self.box_body, 2])
                joint_errors = [
                    float(
                        np.max(
                            np.abs(command.q_des - self.data.qpos[binding.joint_qpos])
                        )
                    )
                    for binding, command in zip(self.bindings, commands)
                ]
                joint_speeds = [
                    float(np.max(np.abs(self.data.qvel[binding.joint_dof])))
                    for binding in self.bindings
                ]
                torques = [
                    float(np.max(np.abs(self.data.ctrl[binding.actuators])))
                    for binding in self.bindings
                ]
                reference_frame = self._reference_frame()
                object_error = float(
                    np.linalg.norm(
                        self.data.xpos[self.box_body]
                        - self._reference_object_position[reference_frame]
                    )
                )
                frame_text = str(reference_frame) if self._phase == "executing" else "-"
                error_text = f"{object_error:.3f}m" if self.object_enabled and self._phase == "executing" else "n/a"
                print(
                    f"[sim2sim] phase={self._phase} steps={steps} sim_time={self.data.time:.2f}s "
                    f"frame={frame_text} root_z=({root_z[0]:.3f}, {root_z[1]:.3f}) "
                    f"box_z={format(box_z, '.3f') if self.object_enabled else 'n/a'} object_error={error_text} "
                    f"q_error=({joint_errors[0]:.3f}, {joint_errors[1]:.3f})rad "
                    f"dq_max=({joint_speeds[0]:.3f}, {joint_speeds[1]:.3f})rad/s "
                    f"torque_max=({torques[0]:.1f}, {torques[1]:.1f})Nm",
                    flush=True,
                )
            if self._viewer is not None:
                self._draw_status()
                self._viewer.sync()

    def _reference_preview_alignment(self):
        if self.raw.get("reference_alignment", "none") == "none":
            return np.array([1., 0., 0., 0.]), np.zeros(3)
        binding = self.bindings[0]
        actual = self.data.qpos[binding.root_qpos:binding.root_qpos + 7]
        # Policy alignment uses robot A pelvis at raw frame zero, including Z.
        origin = self._ghost_roots[0, 0]
        rotation = _yaw_quaternion(_yaw_from_wxyz(actual[3:]) - _yaw_from_wxyz(origin[3:]))
        translation = actual[:3] - quat_apply_batch(rotation, origin[:3][None])[0]
        return rotation, translation

    def _update_reference_ghost(self):
        rotation, translation = (self._reference_preview_alignment()
                                 if self._ghost_alignment is None else self._ghost_alignment)
        frame = int(np.clip(self._ghost_frame, 0, self._ghost_roots.shape[1] - 1))
        self._ghost_data.qpos[:] = self.data.qpos
        for i, binding in enumerate(self.bindings):
            raw = self._ghost_roots[i, frame]
            self._ghost_data.qpos[binding.root_qpos:binding.root_qpos + 7] = np.r_[
                quat_apply_batch(rotation, raw[:3][None])[0] + translation,
                quat_mul_left_batch(rotation, raw[3:][None])[0]]
            self._ghost_data.qpos[binding.joint_qpos] = self._ghost_joints[i, frame]
        self._ghost_data.qpos[self.box_qpos:self.box_qpos + 7] = np.r_[
            quat_apply_batch(rotation, self._reference_object_position[frame][None])[0] + translation,
            quat_mul_left_batch(rotation, self._ghost_box_quat[frame][None])[0]]
        mujoco.mj_forward(self.model, self._ghost_data)

    def _draw_reference_ghost(self, scene):
        if not self._ghost_visible:
            return
        self._update_reference_ghost()
        start = scene.ngeom
        option = mujoco.MjvOption()
        mujoco.mjv_addGeoms(self.model, self._ghost_data, option, mujoco.MjvPerturb(),
                           mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for index in range(start, scene.ngeom):
            geom = scene.geoms[index]
            geom.rgba[:] = [0.1, 0.85, 1.0, 0.28]
            geom.transparent = True
            geom.matid = -1
            geom.emission = 0.35

    def _draw_status(self) -> None:
        with self._viewer.lock():
            scene = self._viewer.user_scn
            scene.ngeom = 0
            self._draw_reference_ghost(scene)
            if self._task_started:
                return
            for binding in self.bindings:
                position = self.data.qpos[
                    binding.root_qpos : binding.root_qpos + 3
                ].copy()
                position[2] += 0.95
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array([0.035, 0.035, 0.12]),
                    position,
                    np.eye(3).ravel(),
                    np.array([1.0, 0.0, 0.0, 0.8]),
                )
                scene.ngeom += 1

    def run(self, *, max_policy_steps: int | None = None) -> None:
        print(
            f"dual ScaleBFM sim2sim: physics={self.physical_hz} Hz "
            f"policy={self.policy_hz:g} Hz decimation={self.decimation}"
        )
        print("Controls: s → DefaultPose; b → ScaleBFM DefaultPose standing; a → task; x → stop; g → reference ghosts", flush=True)
        if self._headless:
            print(
                "Headless: type one key and Enter in the simulator terminal", flush=True
            )
            threading.Thread(target=self._read_terminal_keys, daemon=True).start()
        try:
            if self._headless:
                self._run_loop(max_policy_steps)
            else:
                with mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self.key_callback,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:
                    self._viewer = viewer
                    self._run_loop(max_policy_steps)
        finally:
            self._viewer = None
            self.close()

    def close(self, *_args: Any) -> None:
        if not self._alive:
            return
        self._alive = False
        with self._condition:
            self._condition.notify_all()
        for transport in self.transports:
            transport.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config/g1/dual_scalebfm_residual.yaml")
    )
    parser.add_argument("--initial-scene-source", choices=("config", "vive"), default="config")
    parser.add_argument("--vive-config", help="dual calibrated JSON; defaults to YAML vive_config")
    parser.add_argument("--vive-hz", type=float, default=100.0)
    parser.add_argument("--pose-wait-timeout", type=float, default=10.0)
    parser.add_argument("--pose-max-age", type=float, default=0.1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scalebfm-only", action="store_true", help="empty-handed tracking: remove box geometry")
    parser.add_argument("--reference-bundle", default=None)
    parser.add_argument("--max-policy-steps", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_policy_steps is not None and args.max_policy_steps <= 0:
        raise SystemExit("--max-policy-steps must be positive")
    initial_scene = None
    if args.initial_scene_source == "vive":
        from dual_runtime.initial_scene import capture_vive_initial_scene
        config_path = Path(args.config).expanduser().resolve()
        raw = _load_yaml(config_path)
        # Explicit CLI paths follow the shell cwd; YAML paths follow the YAML.
        vive_path = (Path(args.vive_config).expanduser().resolve() if args.vive_config
                     else _resolve(config_path.parent, raw["vive_config"]))
        print(f"[sim2sim] capturing fresh Vive scene from {vive_path}", flush=True)
        initial_scene = capture_vive_initial_scene(
            vive_path, require_object=not args.scalebfm_only, poll_hz=args.vive_hz,
            wait_timeout_s=args.pose_wait_timeout, max_age_s=args.pose_max_age)
    simulation = DualScaleBFMSim2Sim(
        args.config,
        headless=args.headless,
        scalebfm_only=args.scalebfm_only,
        reference_bundle=args.reference_bundle,
        initial_scene=initial_scene,
    )
    signal.signal(signal.SIGINT, simulation.close)
    signal.signal(signal.SIGTERM, simulation.close)
    try:
        simulation.run(max_policy_steps=args.max_policy_steps)
    except RuntimeError as exc:
        print(f"[sim2sim] ERROR: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
