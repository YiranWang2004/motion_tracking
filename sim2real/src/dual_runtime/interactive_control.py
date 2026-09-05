"""Shared Start/B/A/Stop workflow for simulated and real dual-G1 bridges."""

import time
from pathlib import Path
from dataclasses import replace

import numpy as np
import yaml

from omnicontact.contracts import PDCommand
from .policy_coordinator import (
    CoordinatorResult,
    DeploymentState,
    DualPolicyCoordinator,
)


def load_default_command(asset_dir: Path) -> PDCommand:
    config = yaml.safe_load((asset_dir / "OmniContact.yaml").read_text())
    defaults = yaml.safe_load((asset_dir / "DefaultPose.yaml").read_text())
    order = np.asarray(config["mj2lab"], dtype=int)
    return PDCommand(
        *(
            np.asarray(defaults[key], dtype=np.float32)[order]
            for key in ("default_angles", "kps", "kds")
        )
    )


class InteractiveDualCoordinator(DualPolicyCoordinator):
    """s: DefaultPose, b: independent LocoModes, a: coupled policy, x: stop.

    In simulation, phase metadata travels with both PD commands. No simulator key
    may release a root before both policies have produced the matching command.
    """

    def __init__(self, *args, loco_modes, default_command, input_mode="sim",
                 transition_ticks=0, phase_target_delta=None, require_button_release=False,
                 max_tilt_rad=None, task_safety=None, **kwargs):
        super().__init__(*args, **kwargs)
        if input_mode not in {"sim", "hardware"}:
            raise ValueError("input_mode must be sim or hardware")
        self.input_mode = input_mode
        self.transition_ticks = int(transition_ticks)
        if self.transition_ticks < 0:
            raise ValueError("transition_ticks must be non-negative")
        self.phase_target_delta = phase_target_delta or {}
        self._buttons_initialized = not require_button_release
        self.max_tilt_rad = max_tilt_rad
        self.task_safety = task_safety
        self.last_snapshot = None
        self.snapshot_valid = False
        self.last_result = None
        self._transition_start = None
        self._transition_elapsed = 0
        self.loco_modes = loco_modes
        self.default_command = default_command
        self.state = DeploymentState.ZERO_TORQUE
        self._previous_buttons = {}
        self._task_finished = False
        self._last_state_stamp = None
        self._cached_commands = []
        self._last_reference_position = None
        self._last_reference_frame = -1
        print(
            "ZERO TORQUE: Start/s → DefaultPose; B/b → LocoMode; A/a → task; Stop/x → stop",
            flush=True,
        )

    def _read_inputs(self):
        if self.input_mode == "hardware":
            # Independent bridge clocks/ticks cannot require exact equality.
            return super()._read_inputs()
        states = [robot.read(self.state_timeout_s) for robot in self.robots]
        deadline = time.monotonic() + self.state_timeout_s
        # A retry packet can remain in one UDP receiver while the other has
        # already received the next snapshot. Drain only the older side; never
        # compute a command from a mixed pair or advance twice for a retry.
        while all(state is not None for state in states):
            stamps = [state.state_receive_time_ns for state in states]
            if stamps[0] == stamps[1]:
                break
            remaining = deadline - time.monotonic()
            if None in stamps or remaining <= 0:
                return tuple(states), None, "mismatched_simulation_snapshots"
            older = 0 if stamps[0] < stamps[1] else 1
            states[older] = self.robots[older].read(remaining)
        if any(state is None for state in states):
            return None, None, "missing_bridge_state"
        if any(self._state_age_s(state) > self.state_timeout_s for state in states):
            return None, None, "stale_bridge_state"
        if (
            abs(states[0].packet_arrival_ns - states[1].packet_arrival_ns) * 1e-9
            > self.max_state_skew_s
        ):
            return None, None, "skewed_bridge_state"
        snapshot = self.pose_provider.get_snapshot()
        if snapshot is None:
            return tuple(states), None, "missing_simulation_snapshot"
        age = time.monotonic() - min(
            snapshot.robot_a.stamp_s, snapshot.robot_b.stamp_s, snapshot.object.stamp_s
        )
        if age < 0 or age > self.pose_timeout_s:
            return tuple(states), None, "stale_simulation_snapshot"
        return tuple(states), snapshot, "ready"

    def _send(self, states, commands, phase, *, frame=-1, reference_position=None):
        control = {"phase": phase, "frame": frame} if self.input_mode == "sim" else None
        if reference_position is not None and control is not None:
            control["reference_position_w"] = np.asarray(reference_position).tolist()
        self._cached_commands = []
        for robot, state, command, enable in zip(
            self.robots, states, commands, self.enable
        ):
            if phase in self.phase_target_delta:
                robot.limiter.max_target_delta = self.phase_target_delta[phase]
            # DefaultPose/LocoMode share the single-G1 target path on both backends.
            # The simulator clips actual PD torque at every 200 Hz substep.
            # Rewriting targets using a 50 Hz velocity estimate changes that
            # controller and destabilizes standing. Keep the existing residual
            # target safety path for task commands on both backends.
            if phase == "zero_torque":
                enable = 0
            safe = robot.send_pd(
                command,
                state,
                enable=enable,
                sim_control=control,
                limit_estimated_torque=phase not in {"default_pose", "loco_standing"},
            )
            self._cached_commands.append((safe, enable, control))

        self._last_state_stamp = states[0].state_receive_time_ns
        if phase == "executing":
            self._last_reference_position = np.asarray(reference_position).copy()
            self._last_reference_frame = frame

    def _stop(self, states):
        for robot, state in zip(self.robots, states):
            zero = np.zeros(29, dtype=np.float32)
            robot.send_pd(
                PDCommand(state.q_lab, zero, np.full(29, robot.config.damping_kd)),
                state,
                enable=0,
                sim_control=({"phase": "stopped", "frame": -1} if self.input_mode == "sim" else None),
            )
        self.state = DeploymentState.STOPPED
        return CoordinatorResult(True, self.state, "stop")

    def step(self):
        started = time.perf_counter()
        self._input_wait_s = 0.0
        try:
            result = self._step()
        except BaseException as exc:
            if self.state != DeploymentState.FAULT:
                self._send_hold(None)
            self.state = DeploymentState.FAULT
            self.last_result = CoordinatorResult(False, self.state, f"{type(exc).__name__}: {exc}")
            raise
        result = replace(result, processing_time_s=time.perf_counter() - started - self._input_wait_s)
        self.last_result = result
        return result

    def _step(self):
        waiting = time.perf_counter()
        states, snapshot, reason = self._read_inputs()
        self._input_wait_s = time.perf_counter() - waiting
        self.snapshot_valid = states is not None and snapshot is not None
        if snapshot is not None:
            self.last_snapshot = snapshot
        if states is None or snapshot is None:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            return CoordinatorResult(False, self.state, reason)
        stamps = tuple(state.state_receive_time_ns for state in states)
        if self.input_mode == "sim" and stamps[0] != stamps[1]:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            return CoordinatorResult(
                False, self.state, "mismatched_simulation_snapshots"
            )
        if self.input_mode == "sim" and stamps[0] is not None and stamps[0] == self._last_state_stamp:
            # A retry is not another policy tick. Resend the identical limited
            # commands without advancing LocoMode history or the task reference.
            for robot, state, (command, enable, control) in zip(
                self.robots, states, self._cached_commands
            ):
                robot.client.send(
                    command, enable=enable, state=state, sim_control=control
                )
            return CoordinatorResult(True, self.state, "simulation_snapshot_retry")
        buttons = {
            key: any(state.buttons.get(key, False) for state in states)
            for key in ("start", "B", "A", "stop")
        }
        if not self._buttons_initialized:
            self._previous_buttons = buttons.copy()
            self._buttons_initialized = True
        rises = {
            key: value and not self._previous_buttons.get(key, False)
            for key, value in buttons.items()
        }
        self._previous_buttons = buttons
        try:
            if buttons["stop"] or self.state == DeploymentState.STOPPED:
                return self._stop(states)
            # An upright standing threshold is not a motion-tracking failure criterion.
            # The reference may deliberately crouch and lean while grasping.
            if self.state == DeploymentState.LOCO_STANDING and self.max_tilt_rad is not None:
                for index, state in enumerate(states):
                    q = state.quat_wxyz / np.linalg.norm(state.quat_wxyz)
                    tilt = np.arccos(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))
                    if tilt > self.max_tilt_rad:
                        raise RuntimeError(f"robot {index} tilt {tilt:.3f} rad exceeds {self.max_tilt_rad:.3f}")
            if self.state == DeploymentState.ZERO_TORQUE:
                if rises["start"]:
                    self.state = DeploymentState.DEFAULT_POSE
                    self._transition_start = [state.q_lab.copy() for state in states]
                    self._transition_elapsed = 0
                    print(
                        "Start accepted: transitioning to DefaultPose; wait for completion, then press B/b",
                        flush=True,
                    )
                else:
                    zero = np.zeros(29, dtype=np.float32)
                    self._send(states, [PDCommand(zero, zero, zero)] * 2, "zero_torque")
                    return CoordinatorResult(True, self.state, "zero_torque_wait_s")
            elif (self.state == DeploymentState.DEFAULT_POSE and rises["B"]
                  and self._transition_elapsed >= self.transition_ticks):
                for loco, robot, state in zip(self.loco_modes, self.robots, states):
                    loco.reset()
                    robot.prepare(state)
                self.state = DeploymentState.LOCO_STANDING
                print(
                    "LocoMode standing: confirm both robots are stable, then press A/a",
                    flush=True,
                )
            elif (
                self.state == DeploymentState.LOCO_STANDING
                and rises["A"]
                and not self._task_finished
            ):
                self.last_geometry = self.policy.initialize(
                    snapshot,
                    max_partner_position_error_m=self.geometry_limits[0],
                    max_object_position_error_m=self.geometry_limits[1],
                    max_box_size_error_m=self.geometry_limits[2],
                    max_robot_orientation_error_rad=self.geometry_limits[3],
                    max_object_orientation_error_rad=self.geometry_limits[4],
                )
                self.reference_alignment_pose = snapshot.robot_a
                self.policy.reset_rollout()
                # Preserve the last issued target across A, as in the original
                # tracking path; measured q includes normal PD tracking error.
                self.state = DeploymentState.EXECUTING
                print(
                    "A accepted: executing dual ScaleBFM residual reference", flush=True
                )

            if self.state == DeploymentState.DEFAULT_POSE:
                self._transition_elapsed += 1
                alpha = min(1.0, self._transition_elapsed / max(1, self.transition_ticks))
                commands = [PDCommand(start * (1.0 - alpha) + self.default_command.target_pos * alpha,
                                      self.default_command.kp, self.default_command.kd)
                            for start in self._transition_start]
                self._send(states, commands, "default_pose")
                if self._transition_elapsed == max(1, self.transition_ticks):
                    print("DefaultPose ready: press B/b for LocoMode standing", flush=True)
                return CoordinatorResult(True, self.state, "default_pose_wait_b")
            if self.state == DeploymentState.LOCO_STANDING:
                commands = [
                    loco.compute(state) for loco, state in zip(self.loco_modes, states)
                ]
                self._send(states, commands, "loco_standing")
                return CoordinatorResult(True, self.state, "loco_mode_standing")
            if self.state == DeploymentState.EXECUTING:
                step = self.policy.compute(states, snapshot)
                commands = [
                    PDCommand(target, self.policy.kp, self.policy.kd)
                    for target in step.targets
                ]
                reference = self.policy.reference.frame(step.frame)[0].object_pos_w
                if self.task_safety is not None:
                    checked_reference = reference if self._last_reference_position is None else self._last_reference_position
                    checked_frame = step.frame if self._last_reference_position is None else self._last_reference_frame
                    delta = snapshot.object.position_w - checked_reference
                    if abs(float(delta[2])) > self.task_safety["object_position_z_error_m"]:
                        raise RuntimeError(f"object.position_z frame={checked_frame} error={abs(float(delta[2])):.3f}m")
                    if float(np.linalg.norm(delta)) > self.task_safety["object_position_xyz_error_m"]:
                        raise RuntimeError(f"object.position_xyz frame={checked_frame} error={float(np.linalg.norm(delta)):.3f}m")
                self._send(
                    states,
                    commands,
                    "executing",
                    frame=step.frame,
                    reference_position=reference,
                )
                if step.complete:
                    self._task_finished = True
                    for loco, robot, state in zip(self.loco_modes, self.robots, states):
                        loco.reset()
                        robot.prepare(state)
                    self.state = DeploymentState.LOCO_STANDING
                    print(
                        "Reference complete; returning to LocoMode standing. Press x to stop",
                        flush=True,
                    )
                return CoordinatorResult(True, self.state, "policy_target", step)
            raise RuntimeError(f"unexpected interactive state: {self.state}")
        except BaseException:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            raise
