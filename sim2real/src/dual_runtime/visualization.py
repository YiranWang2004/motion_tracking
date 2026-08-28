"""Read-only visualization protocol for the dual ScaleBFM MuJoCo twin."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.udp_latest import UDPLatestReceiver, UDPLatestSender
from omnicontact.contracts import ObjectPose, RobotPose

from .dual_pose_provider import DualPoseSnapshot


PROTOCOL = "robojudo.dual_scalebfm.visualization"
VERSION = 1


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(shape)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result.copy()


def _pose_wxyz(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    position_array = _array(position, (3,), "position")
    quaternion = _array(quaternion_xyzw, (4,), "quaternion_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-6:
        raise ValueError("quaternion_xyzw has zero norm")
    quaternion /= norm
    return np.concatenate((position_array, quaternion[[3, 0, 1, 2]]))


def robot_pose_wxyz(pose: RobotPose) -> np.ndarray:
    return _pose_wxyz(pose.position_w, pose.quaternion_xyzw)


def object_pose_wxyz(pose: ObjectPose) -> np.ndarray:
    return _pose_wxyz(pose.position_w, pose.quaternion_xyzw)


def encode_visualization(
    *,
    actual_robot_base_wxyz: Any,
    actual_object_wxyz: Any,
    actual_box_half_extents: Any,
    reference_robot_base_wxyz: Any,
    reference_joint_pos: Any,
    reference_object_wxyz: Any,
    reference_box_half_extents: Any,
    target_joint_pos: Any,
    deployment_state: str,
    frame: int,
    scalebfm_target: Any | None = None,
    residual: Any | None = None,
) -> dict[str, Any]:
    """Build one validated latest-state packet.

    Poses use MuJoCo's ``[x, y, z, qw, qx, qy, qz]`` convention. The real
    robot joint state is intentionally absent: it comes directly from the two
    bridge state-mirror sockets so visualization cannot affect control routing.
    """

    packet: dict[str, Any] = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "actual": {
            "robot_base_wxyz": _array(
                actual_robot_base_wxyz, (2, 7), "actual.robot_base_wxyz"
            ),
            "object_wxyz": _array(actual_object_wxyz, (7,), "actual.object_wxyz"),
            "box_half_extents": _array(
                actual_box_half_extents, (3,), "actual.box_half_extents"
            ),
        },
        "reference": {
            "robot_base_wxyz": _array(
                reference_robot_base_wxyz, (2, 7), "reference.robot_base_wxyz"
            ),
            "joint_pos": _array(reference_joint_pos, (2, 29), "reference.joint_pos"),
            "object_wxyz": _array(
                reference_object_wxyz, (7,), "reference.object_wxyz"
            ),
            "box_half_extents": _array(
                reference_box_half_extents, (3,), "reference.box_half_extents"
            ),
        },
        "policy": {
            "target_joint_pos": _array(
                target_joint_pos, (2, 29), "policy.target_joint_pos"
            ),
            "deployment_state": str(deployment_state),
            "frame": int(frame),
        },
    }
    if not packet["policy"]["deployment_state"]:
        raise ValueError("deployment_state must be non-empty")
    if scalebfm_target is not None:
        packet["policy"]["scalebfm_target"] = _array(
            scalebfm_target, (2, 29), "policy.scalebfm_target"
        )
    if residual is not None:
        packet["policy"]["residual"] = _array(
            residual, (2, 29), "policy.residual"
        )
    return packet


def decode_visualization(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    try:
        if data.get("protocol") != PROTOCOL or int(data.get("version", -1)) != VERSION:
            return None
        actual = data["actual"]
        reference = data["reference"]
        policy = data["policy"]
        return encode_visualization(
            actual_robot_base_wxyz=actual["robot_base_wxyz"],
            actual_object_wxyz=actual["object_wxyz"],
            actual_box_half_extents=actual["box_half_extents"],
            reference_robot_base_wxyz=reference["robot_base_wxyz"],
            reference_joint_pos=reference["joint_pos"],
            reference_object_wxyz=reference["object_wxyz"],
            reference_box_half_extents=reference["box_half_extents"],
            target_joint_pos=policy["target_joint_pos"],
            deployment_state=policy["deployment_state"],
            frame=policy["frame"],
            scalebfm_target=policy.get("scalebfm_target"),
            residual=policy.get("residual"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_runtime_visualization(coordinator: Any, result: Any) -> dict[str, Any] | None:
    """Snapshot coordinator state after a tick without changing control state."""

    snapshot: DualPoseSnapshot | None = coordinator.pose_provider.get_snapshot()
    if snapshot is None or not coordinator.policy.initialized:
        return None
    policy_step = result.policy_step
    frame = coordinator.policy.frame if policy_step is None else policy_step.frame
    reference_a, reference_b = coordinator.policy.reference.frame(frame)
    commands = [robot.last_command for robot in coordinator.robots]
    targets = np.stack(
        [
            coordinator.policy.default_q
            if command is None
            else np.asarray(command.target_pos, dtype=np.float32)
            for command in commands
        ]
    )
    return encode_visualization(
        actual_robot_base_wxyz=np.stack(
            (robot_pose_wxyz(snapshot.robot_a), robot_pose_wxyz(snapshot.robot_b))
        ),
        actual_object_wxyz=object_pose_wxyz(snapshot.object),
        actual_box_half_extents=snapshot.object.half_extents,
        reference_robot_base_wxyz=np.stack(
            [
                np.concatenate((reference_a.body_pos_w[0], reference_a.body_quat_wxyz[0])),
                np.concatenate((reference_b.body_pos_w[0], reference_b.body_quat_wxyz[0])),
            ]
        ),
        reference_joint_pos=np.stack((reference_a.joint_pos, reference_b.joint_pos)),
        reference_object_wxyz=np.concatenate(
            (reference_a.object_pos_w, reference_a.object_quat_wxyz)
        ),
        reference_box_half_extents=coordinator.policy.reference.box_half_extents,
        target_joint_pos=targets,
        deployment_state=result.state.value,
        frame=frame,
        scalebfm_target=None if policy_step is None else policy_step.scalebfm_targets,
        residual=None if policy_step is None else policy_step.residuals,
    )


class DualVisualizationSender:
    """Best-effort sender that never raises into the motor-control loop."""

    def __init__(self, host: str, port: int) -> None:
        self._sender = UDPLatestSender(host, int(port))

    def send(self, packet: dict[str, Any] | None) -> None:
        if packet is None:
            return
        try:
            validated = decode_visualization(packet)
            if validated is not None:
                self._sender.send(validated)
        except (OSError, TypeError, ValueError):
            return

    def close(self) -> None:
        self._sender.close()


def publish_runtime_visualization(
    sender: DualVisualizationSender | None,
    coordinator: Any,
    result: Any,
) -> None:
    """Keep all visualization work outside the control failure domain."""

    if sender is None:
        return
    try:
        sender.send(build_runtime_visualization(coordinator, result))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return


class DualVisualizationReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.receiver = UDPLatestReceiver(host, int(port))

    def start(self) -> None:
        self.receiver.start()

    def read_latest(self, *, with_meta: bool = False):
        packet = self.receiver.read_latest_data(with_meta=with_meta)
        if packet is None:
            return None
        if not with_meta:
            return decode_visualization(packet)
        decoded = decode_visualization(packet.data)
        if decoded is None:
            return None
        return packet.__class__(
            seq=packet.seq,
            send_time_ns=packet.send_time_ns,
            recv_time_ns=packet.recv_time_ns,
            addr=packet.addr,
            data=decoded,
        )

    def close(self) -> None:
        self.receiver.close()
