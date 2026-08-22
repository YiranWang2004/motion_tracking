"""Small OpenVR pose reader with stable serial-number selection.

OpenVR device indices are process/runtime-local and may change after reconnects.
The public multi-tracker API therefore keys every pose by the hardware serial
number while polling all devices in a single OpenVR call.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class ViveSample:
    sec: int
    nsec: int
    line_x_m: float
    line_y_m: float
    line_z_m: float
    qx: float
    qy: float
    qz: float
    qw: float


def _quat_normalize(
    q: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _rotmat_to_quat(
    r00: float,
    r01: float,
    r02: float,
    r10: float,
    r11: float,
    r12: float,
    r20: float,
    r21: float,
    r22: float,
) -> Tuple[float, float, float, float]:
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale
    return _quat_normalize((qx, qy, qz, qw))


class OpenVRTrackerReader:
    """Own one OpenVR session and read multiple GenericTracker devices."""

    def __init__(self, required_serials: Iterable[str] | None = None):
        self.required_serials = tuple(required_serials or ())
        if len(set(self.required_serials)) != len(self.required_serials):
            raise ValueError("required_serials contains duplicates")
        self._openvr = None
        self._vr_system = None
        self._serial_to_index: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def serial_to_index(self) -> dict[str, int]:
        with self._lock:
            return dict(self._serial_to_index)

    def start(self) -> dict[str, int]:
        try:
            import openvr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Python openvr 包，请执行: python -m pip install openvr==2.12.1401"
            ) from exc

        if not openvr.isRuntimeInstalled():
            raise RuntimeError("未找到 SteamVR Runtime，请先安装并启动 SteamVR")
        try:
            vr_system = openvr.init(openvr.VRApplication_Other)
        except openvr.OpenVRError as exc:
            raise RuntimeError(
                f"OpenVR 初始化失败: {exc} (code={getattr(exc, 'error_code', 'N/A')})"
            ) from exc

        with self._lock:
            self._openvr = openvr
            self._vr_system = vr_system
        devices = self.refresh_devices()
        missing = [serial for serial in self.required_serials if serial not in devices]
        if missing:
            self.stop()
            found = ", ".join(sorted(devices)) or "无"
            raise RuntimeError(
                f"未找到指定 Vive Tracker: {', '.join(missing)}；当前发现: {found}"
            )
        if not devices:
            self.stop()
            raise RuntimeError("SteamVR 中没有 GenericTracker，请检查供电、Dongle 和配对")
        return devices

    def refresh_devices(self) -> dict[str, int]:
        with self._lock:
            openvr = self._openvr
            vr_system = self._vr_system
        if openvr is None or vr_system is None:
            return {}

        devices: dict[str, int] = {}
        for index in range(openvr.k_unMaxTrackedDeviceCount):
            if (
                vr_system.getTrackedDeviceClass(index)
                != openvr.TrackedDeviceClass_GenericTracker
            ):
                continue
            try:
                serial = vr_system.getStringTrackedDeviceProperty(
                    index, openvr.Prop_SerialNumber_String
                ).strip()
            except openvr.OpenVRError:
                continue
            if serial:
                devices[serial] = index
        with self._lock:
            self._serial_to_index = devices
        return dict(devices)

    def read_all(
        self, serials: Iterable[str] | None = None
    ) -> dict[str, Optional[ViveSample]]:
        with self._lock:
            openvr = self._openvr
            vr_system = self._vr_system
            devices = dict(self._serial_to_index)
        if openvr is None or vr_system is None:
            return {}

        requested = tuple(serials) if serials is not None else tuple(devices)
        poses = vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        now_ns = time.time_ns()
        sec, nsec = divmod(now_ns, 1_000_000_000)
        output: dict[str, Optional[ViveSample]] = {}
        for serial in requested:
            index = devices.get(serial)
            if index is None:
                output[serial] = None
                continue
            pose = poses[index]
            if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                output[serial] = None
                continue
            matrix = pose.mDeviceToAbsoluteTracking.m
            qx, qy, qz, qw = _rotmat_to_quat(
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
            )
            output[serial] = ViveSample(
                sec=int(sec),
                nsec=int(nsec),
                line_x_m=float(matrix[0][3]),
                line_y_m=float(matrix[1][3]),
                line_z_m=float(matrix[2][3]),
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
        return output

    def stop(self) -> None:
        with self._lock:
            openvr = self._openvr
            was_started = self._vr_system is not None
            self._vr_system = None
            self._serial_to_index = {}
            self._openvr = None
        if was_started and openvr is not None:
            openvr.shutdown()

    def __enter__(self) -> "OpenVRTrackerReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


class ViveTracker:
    """Backward-compatible single-Tracker facade.

    Pass ``serial_number`` whenever more than one Tracker is paired.
    """

    def __init__(self, serial_number: str | None = None):
        self.serial_number = serial_number
        self._reader = OpenVRTrackerReader(
            [serial_number] if serial_number is not None else None
        )
        self._selected_serial: str | None = None

    def start(self) -> int:
        devices = self._reader.start()
        if self.serial_number is not None:
            selected = self.serial_number
        elif len(devices) == 1:
            selected = next(iter(devices))
        else:
            self._reader.stop()
            serials = ", ".join(sorted(devices))
            raise RuntimeError(
                f"检测到多个 Tracker ({serials})，必须通过 serial_number 明确选择"
            )
        self._selected_serial = selected
        return devices[selected]

    def read(self) -> Optional[ViveSample]:
        if self._selected_serial is None:
            return None
        return self._reader.read_all([self._selected_serial]).get(
            self._selected_serial
        )

    def stop(self) -> None:
        self._reader.stop()
        self._selected_serial = None


def make_vive_txt_path(merged_trajectory_dir: str) -> str:
    os.makedirs(merged_trajectory_dir, exist_ok=True)
    return os.path.join(merged_trajectory_dir, "merged_trajectory.txt")


make_vive_csv_path = make_vive_txt_path


def write_vive_header(writer: csv.writer) -> None:
    del writer


def write_vive_row(writer: csv.writer, sample: ViveSample) -> None:
    writer.writerow(
        [
            f"{sample.sec}.{sample.nsec:09d}",
            sample.line_x_m,
            sample.line_y_m,
            sample.line_z_m,
            sample.qx,
            sample.qy,
            sample.qz,
            sample.qw,
        ]
    )


__all__ = [
    "OpenVRTrackerReader",
    "ViveSample",
    "ViveTracker",
    "make_vive_csv_path",
    "make_vive_txt_path",
    "write_vive_header",
    "write_vive_row",
]

