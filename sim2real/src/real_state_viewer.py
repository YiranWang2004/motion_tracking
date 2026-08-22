"""Read-only MuJoCo digital twin for the physical robot state."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.udp_latest import UDPLatestReceiver
from paths import SUPPORTED_ROBOTS, controller_config_path, robot_xml_path


DEFAULT_VIEWER_PORT = 55003


def _load_joint_names(robot: str) -> list[str]:
    with controller_config_path(robot).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    names = list(config["policy_joint_names"])
    if not names or len(names) != len(set(names)):
        raise ValueError("controller policy_joint_names must be non-empty and unique")
    return names


def _resolve_joint_addresses(
    model: mujoco.MjModel, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}")
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Expected a 1-DoF joint for {name!r}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def _normalized_quaternion(value) -> np.ndarray | None:
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        return None
    return quat / norm


class RealStateViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = args.robot.lower()
        self.joint_names = _load_joint_names(self.robot)
        xml_path = Path(args.xml_path).expanduser().resolve() if args.xml_path else robot_xml_path(self.robot)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.qpos_addresses, self.dof_addresses = _resolve_joint_addresses(self.model, self.joint_names)

        free_joint_id = next(
            (
                joint_id
                for joint_id in range(self.model.njnt)
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            ),
            None,
        )
        if free_joint_id is None:
            raise ValueError("MuJoCo model must contain a floating-base free joint")
        self.root_qpos_address = int(self.model.jnt_qposadr[free_joint_id])
        self.root_dof_address = int(self.model.jnt_dofadr[free_joint_id])
        self.data.qpos[self.root_qpos_address : self.root_qpos_address + 3] = (
            float(args.root_x),
            float(args.root_y),
            float(args.root_height),
        )
        self.data.qpos[self.root_qpos_address + 3 : self.root_qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

        self.receiver = UDPLatestReceiver(args.bind_host, args.port)
        self.receiver.start()
        self.running = True
        self.last_packet_seq: int | None = None
        self.last_state_time = 0.0
        self.stale_reported = False

    def close(self, *_args) -> None:
        if not self.running:
            return
        self.running = False
        self.receiver.close()

    def apply_state(self, payload) -> None:
        q = np.asarray(payload["q"], dtype=np.float64).reshape(-1)
        dq = np.asarray(payload["dq"], dtype=np.float64).reshape(-1)
        if q.size != len(self.joint_names) or dq.size != len(self.joint_names):
            raise ValueError(
                f"state has q/dq sizes {q.size}/{dq.size}; expected {len(self.joint_names)}"
            )
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise ValueError("state q/dq contains NaN or infinity")

        self.data.qpos[self.qpos_addresses] = q
        self.data.qvel[self.dof_addresses] = dq
        if not self.args.no_imu:
            quat = _normalized_quaternion(payload.get("quat_wxyz"))
            if quat is not None:
                self.data.qpos[self.root_qpos_address + 3 : self.root_qpos_address + 7] = quat
            gyro = np.asarray(payload.get("gyro", ()), dtype=np.float64).reshape(-1)
            if gyro.size == 3 and np.all(np.isfinite(gyro)):
                self.data.qvel[self.root_dof_address + 3 : self.root_dof_address + 6] = gyro
        mujoco.mj_forward(self.model, self.data)

    def run(self) -> None:
        print(
            f"[RealStateViewer] Listening on {self.args.bind_host}:{self.args.port}; "
            f"showing {len(self.joint_names)} measured joints"
        )
        print(
            "[RealStateViewer] Read-only: fixed root translation, "
            + ("fixed root orientation" if self.args.no_imu else "measured IMU orientation")
        )
        frame_period = 1.0 / self.args.fps
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0

            while self.running and viewer.is_running():
                loop_start = time.monotonic()
                packet = self.receiver.read_latest_data(with_meta=True)
                if packet is not None and packet.seq != self.last_packet_seq:
                    try:
                        self.apply_state(packet.data)
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"[RealStateViewer] Ignoring malformed state packet: {exc}")
                    else:
                        self.last_packet_seq = packet.seq
                        self.last_state_time = loop_start
                        if self.stale_reported:
                            print("[RealStateViewer] State stream recovered")
                            self.stale_reported = False

                if (
                    self.last_state_time > 0.0
                    and loop_start - self.last_state_time > self.args.stale_timeout
                    and not self.stale_reported
                ):
                    print(
                        f"[RealStateViewer] State stale for more than {self.args.stale_timeout:.2f}s; "
                        "freezing the last displayed pose"
                    )
                    self.stale_reported = True

                viewer.sync()
                remaining = frame_period - (time.monotonic() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the physical robot's measured state in a read-only MuJoCo window"
    )
    parser.add_argument("--robot", choices=list(SUPPORTED_ROBOTS), default="g1")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    parser.add_argument("--root-x", type=float, default=0.0)
    parser.add_argument("--root-y", type=float, default=0.0)
    parser.add_argument("--root-height", type=float, default=0.793)
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="Keep the floating-base orientation fixed instead of using the measured IMU quaternion",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.stale_timeout <= 0.0:
        parser.error("--stale-timeout must be positive")
    return args


def main(argv=None) -> None:
    viewer = RealStateViewer(parse_args(argv))
    signal.signal(signal.SIGINT, viewer.close)
    signal.signal(signal.SIGTERM, viewer.close)
    try:
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
