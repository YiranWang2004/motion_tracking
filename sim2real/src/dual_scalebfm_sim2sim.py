#!/usr/bin/env python3
"""Two-G1 MuJoCo bridge for the production ScaleBFM residual deploy entry."""

from __future__ import annotations

import argparse
import json
import signal
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
    linacc_sensor: tuple[int, int] | None


@dataclass(frozen=True)
class SimCommand:
    q_des: np.ndarray
    qd_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    enable: int
    state_time_ns: int


class DualScaleBFMSim2Sim:
    """Lockstep two-bridge replacement backed by one shared MuJoCo snapshot."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        headless: bool = False,
        transports: tuple[Any, Any] | None = None,
    ) -> None:
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
        if self.physical_hz <= 0 or self.policy_hz <= 0 or not np.isclose(
            ratio, self.decimation
        ):
            raise ValueError("physical_frequency_hz must be an integer multiple of control_frequency_hz")
        self.lockstep_timeout_s = float(sim.get("lockstep_timeout_s", 1.0))
        self.startup_timeout_s = float(sim.get("startup_timeout_s", 60.0))
        self.command_retry_s = float(sim.get("state_retry_s", 0.05))
        self.disabled_damping_kd = float(sim.get("disabled_damping_kd", 8.0))
        if min(
            self.lockstep_timeout_s,
            self.startup_timeout_s,
            self.command_retry_s,
            self.disabled_damping_kd,
        ) <= 0.0:
            raise ValueError("simulation timeouts and damping must be positive")

        xml_path = _resolve(self.config_path.parent, sim["xml_path"])
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = 1.0 / self.physical_hz
        self.data = mujoco.MjData(self.model)
        self.bindings = (self._bind_robot("a_"), self._bind_robot("b_"))
        if self.model.nu != 2 * len(POLICY_JOINT_NAMES):
            raise ValueError(f"dual model must have 58 actuators, got {self.model.nu}")

        self.box_body = _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_joint = _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self.box_qpos = int(self.model.jnt_qposadr[box_joint])
        self.box_dof = int(self.model.jnt_dofadr[box_joint])
        self.box_half_extents = np.asarray(
            sim.get("box_half_extents", (0.30, 0.15, 0.15)), dtype=np.float32
        ).reshape(3)

        torque = sim.get("torque_limits")
        if torque is None:
            metadata_path = _resolve(
                self.config_path.parent,
                Path(self.raw["artifacts"]["directory"])
                / self.raw["artifacts"]["scalebfm_metadata"],
            )
            torque = json.loads(metadata_path.read_text(encoding="utf-8"))["torque_limit"]
        self.torque_limits = np.asarray(torque, dtype=np.float64).reshape(29)
        if np.any(self.torque_limits <= 0.0):
            raise ValueError("simulation torque limits must be positive")

        reference_path = _resolve(
            self.config_path.parent,
            Path(self.raw["artifacts"]["directory"])
            / self.raw["artifacts"]["reference_bundle"],
        )
        self._initialize_from_reference(reference_path, int(self.raw.get("start_frame", 1)))

        robot_cfg = sim.get("robots")
        if not isinstance(robot_cfg, dict):
            raise ValueError("simulation.robots must contain robot_a and robot_b")
        self._condition = threading.Condition()
        self._commands: list[SimCommand | None] = [None, None]
        self._connected = [False, False]
        if transports is None:
            self.transports = tuple(
                UDPRobotLow(self._low_udp_config(robot_cfg[key]["udp"]), on_command_packet=self._handler(index))
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
        actuator_ids = np.asarray(
            [
                _object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
                for name in POLICY_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            prefix + "imu-torso-linear-acceleration",
        )
        sensor = None
        if sensor_id >= 0:
            sensor = (
                int(self.model.sensor_adr[sensor_id]),
                int(self.model.sensor_dim[sensor_id]),
            )
        return RobotBinding(
            root_qpos=int(self.model.jnt_qposadr[root]),
            root_dof=int(self.model.jnt_dofadr[root]),
            joint_qpos=np.asarray(self.model.jnt_qposadr[joint_ids], dtype=np.int32),
            joint_dof=np.asarray(self.model.jnt_dofadr[joint_ids], dtype=np.int32),
            actuators=actuator_ids,
            linacc_sensor=sensor,
        )

    def _initialize_from_reference(self, path: Path, frame: int) -> None:
        with np.load(path, allow_pickle=False) as data:
            order = tuple(str(value) for value in data["training_joint_order"].tolist())
            if order != tuple(POLICY_JOINT_NAMES):
                raise ValueError("reference joint order does not match deployment policy order")
            frames = int(data["training_object_body_pos_w"].shape[0])
            if not 0 <= frame < frames:
                raise ValueError("start_frame is outside the reference bundle")
            reference_extents = np.asarray(
                data["training_box_half_extents"], dtype=np.float32
            )
            if not np.allclose(reference_extents, self.box_half_extents, atol=1e-6):
                raise ValueError("simulation box size does not match reference bundle")
            for index, binding in enumerate(self.bindings):
                position = data[f"training_robot_{index}_body_pos_w"][frame, 0]
                quaternion = data[f"training_robot_{index}_body_quat_w"][frame, 0]
                self.data.qpos[binding.root_qpos : binding.root_qpos + 7] = np.concatenate(
                    (position, quaternion)
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

    def _handler(self, robot_index: int):
        def receive(packet: LatestPacket) -> None:
            try:
                command = self.parse_command(packet.data)
            except (TypeError, ValueError) as exc:
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
        return SimCommand(
            **values,
            enable=enable,
            state_time_ns=int(payload["state_receive_time_ns"]),
        )

    def _robot_state(self, binding: RobotBinding) -> tuple[np.ndarray, ...]:
        q = self.data.qpos[binding.joint_qpos].copy().astype(np.float32)
        dq = self.data.qvel[binding.joint_dof].copy().astype(np.float32)
        quat = self.data.qpos[binding.root_qpos + 3 : binding.root_qpos + 7].copy()
        gyro = self.data.qvel[binding.root_dof + 3 : binding.root_dof + 6].copy()
        linacc = np.zeros(3, dtype=np.float32)
        if binding.linacc_sensor is not None and binding.linacc_sensor[1] >= 3:
            address = binding.linacc_sensor[0]
            linacc = self.data.sensordata[address : address + 3].copy().astype(np.float32)
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

    def publish_state_pair(self, state_time_ns: int) -> None:
        pose = self.pose_payload()
        for transport, binding in zip(self.transports, self.bindings):
            q, dq, quat, gyro, linacc = self._robot_state(binding)
            transport.send_state(
                q=q,
                dq=dq,
                quat_wxyz=quat,
                gyro=gyro,
                linacc=linacc,
                buttons={name: False for name in BUTTONS},
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
            if command.enable:
                torque = command.kp * (command.q_des - q) + command.kd * (
                    command.qd_des - dq
                )
            else:
                torque = -self.disabled_damping_kd * dq
            self.data.ctrl[binding.actuators] = np.clip(
                torque, -self.torque_limits, self.torque_limits
            )

    def step_policy_interval(self, commands: tuple[SimCommand, SimCommand]) -> None:
        self.apply_commands(commands)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

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
            self._snapshot_id += 1
            steps += 1
            if self._viewer is not None:
                self._viewer.sync()

    def run(self, *, max_policy_steps: int | None = None) -> None:
        print(
            f"dual ScaleBFM sim2sim: physics={self.physical_hz} Hz "
            f"policy={self.policy_hz:g} Hz decimation={self.decimation}"
        )
        try:
            if self._headless:
                self._run_loop(max_policy_steps)
            else:
                with mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-policy-steps", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_policy_steps is not None and args.max_policy_steps <= 0:
        raise SystemExit("--max-policy-steps must be positive")
    simulation = DualScaleBFMSim2Sim(args.config, headless=args.headless)
    signal.signal(signal.SIGINT, simulation.close)
    signal.signal(signal.SIGTERM, simulation.close)
    simulation.run(max_policy_steps=args.max_policy_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
