"""Original OmniContact LocoMode policy adapted to the G1 bridge contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from omnicontact.contracts import PDCommand
from omnicontact.runtime import BridgeState


class LocoModePolicy:
    """Run the original recurrent 29-DoF locomotion policy at zero velocity."""

    OBSERVATION_SIZE = 96
    ACTION_SIZE = 29
    HIDDEN_SIZE = 256

    def __init__(self, asset_dir: str | Path, controller_joint_names: list[str]) -> None:
        self.asset_dir = Path(asset_dir).expanduser().resolve()
        with (self.asset_dir / "LocoMode.yaml").open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        expected_joint_names = list(config["policy_joint_names"])
        if list(controller_joint_names) != expected_joint_names:
            raise ValueError(
                "motion_tracking policy_joint_names do not match LocoMode order\n"
                f"expected={expected_joint_names}\nactual={list(controller_joint_names)}"
            )

        self.default_lab = self._array(config, "default_angles")
        self.kp_lab = self._array(config, "kps")
        self.kd_lab = self._array(config, "kds")
        self.command = np.asarray(config["cmd_init"], dtype=np.float32).reshape(3)
        self.command_scale = np.asarray(config["cmd_scale"], dtype=np.float32).reshape(3)
        self.angular_velocity_scale = float(config["ang_vel_scale"])
        self.joint_position_scale = float(config["dof_pos_scale"])
        self.joint_velocity_scale = float(config["dof_vel_scale"])
        self.action_scale = float(config["action_scale"])

        model_path = self.asset_dir / str(config["onnx_path"])
        self.session = ort.InferenceSession(
            model_path.as_posix(), providers=["CPUExecutionProvider"]
        )
        expected_inputs = {
            "observation": [1, self.OBSERVATION_SIZE],
            "hidden_state": [1, 1, self.HIDDEN_SIZE],
            "cell_state": [1, 1, self.HIDDEN_SIZE],
        }
        actual_inputs = {item.name: item.shape for item in self.session.get_inputs()}
        if actual_inputs != expected_inputs:
            raise ValueError(
                f"invalid LocoMode ONNX inputs: expected={expected_inputs}, actual={actual_inputs}"
            )
        expected_outputs = {
            "action": [1, self.ACTION_SIZE],
            "next_hidden_state": [1, 1, self.HIDDEN_SIZE],
            "next_cell_state": [1, 1, self.HIDDEN_SIZE],
        }
        actual_outputs = {item.name: item.shape for item in self.session.get_outputs()}
        if actual_outputs != expected_outputs:
            raise ValueError(
                "invalid LocoMode ONNX outputs: "
                f"expected={expected_outputs}, actual={actual_outputs}"
            )

        self.action_lab = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self.hidden_state = np.zeros((1, 1, self.HIDDEN_SIZE), dtype=np.float32)
        self.cell_state = np.zeros((1, 1, self.HIDDEN_SIZE), dtype=np.float32)

    @classmethod
    def _array(cls, config: dict, name: str) -> np.ndarray:
        value = np.asarray(config[name], dtype=np.float32).reshape(-1)
        if value.shape != (cls.ACTION_SIZE,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid LocoMode parameter {name}")
        return value

    def reset(self) -> None:
        """Reset the recurrent state whenever the FSM enters LocoMode."""
        self.action_lab.fill(0.0)
        self.hidden_state.fill(0.0)
        self.cell_state.fill(0.0)

    @staticmethod
    def _projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4)
        return np.array(
            [
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ],
            dtype=np.float32,
        )

    def build_observation(self, state: BridgeState) -> np.ndarray:
        observation = np.concatenate(
            (
                state.gyro * self.angular_velocity_scale,
                self._projected_gravity(state.quat_wxyz),
                self.command * self.command_scale,
                (state.q_lab - self.default_lab) * self.joint_position_scale,
                state.dq_lab * self.joint_velocity_scale,
                self.action_lab,
            )
        ).astype(np.float32)
        if observation.shape != (self.OBSERVATION_SIZE,) or not np.all(
            np.isfinite(observation)
        ):
            raise RuntimeError("invalid LocoMode observation")
        return observation

    def compute(self, state: BridgeState) -> PDCommand:
        observation = self.build_observation(state)
        action, next_hidden, next_cell = self.session.run(
            None,
            {
                "observation": observation[None, :],
                "hidden_state": self.hidden_state,
                "cell_state": self.cell_state,
            },
        )
        self.action_lab = np.asarray(action, dtype=np.float32).reshape(self.ACTION_SIZE)
        self.hidden_state = np.asarray(next_hidden, dtype=np.float32).reshape(
            1, 1, self.HIDDEN_SIZE
        )
        self.cell_state = np.asarray(next_cell, dtype=np.float32).reshape(
            1, 1, self.HIDDEN_SIZE
        )
        if not (
            np.all(np.isfinite(self.action_lab))
            and np.all(np.isfinite(self.hidden_state))
            and np.all(np.isfinite(self.cell_state))
        ):
            raise RuntimeError("LocoMode ONNX returned non-finite values")
        target = self.action_lab * self.action_scale + self.default_lab
        return PDCommand(target, self.kp_lab, self.kd_lab)
