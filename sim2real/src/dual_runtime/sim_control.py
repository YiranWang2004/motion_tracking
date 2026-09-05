"""Interactive dual simulation control; hardware coordinator remains independent."""

from pathlib import Path

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

    Phase metadata travels with both timestamped PD commands. No simulator key
    may release a root before both policies have produced the matching command.
    """

    def __init__(self, *args, loco_modes, default_command, **kwargs):
        super().__init__(*args, **kwargs)
        self.loco_modes = loco_modes
        self.default_command = default_command
        self.state = DeploymentState.ZERO_TORQUE
        self._previous_buttons = {}
        self._task_finished = False
        self._last_state_stamp = None
        self._cached_commands = []
        print(
            "ZERO TORQUE: press s for DefaultPose, b for LocoMode, a for task, x to stop",
            flush=True,
        )

    def _send(self, states, commands, phase, *, frame=-1, reference_position=None):
        control = {"phase": phase, "frame": frame}
        if reference_position is not None:
            control["reference_position_w"] = np.asarray(reference_position).tolist()
        self._cached_commands = []
        for robot, state, command, enable in zip(
            self.robots, states, commands, self.enable
        ):
            safe = robot.send_pd(command, state, enable=enable, sim_control=control)
            self._cached_commands.append((safe, enable, control))

        self._last_state_stamp = states[0].state_receive_time_ns

    def _stop(self, states):
        for robot, state in zip(self.robots, states):
            zero = np.zeros(29, dtype=np.float32)
            robot.send_pd(
                PDCommand(state.q_lab, zero, np.full(29, robot.config.damping_kd)),
                state,
                enable=0,
                sim_control={"phase": "stopped", "frame": -1},
            )
        self.state = DeploymentState.STOPPED
        return CoordinatorResult(True, self.state, "stop")

    def step(self):
        states, snapshot, reason = self._read_inputs()
        if states is None or snapshot is None:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            return CoordinatorResult(False, self.state, reason)
        stamps = tuple(state.state_receive_time_ns for state in states)
        if stamps[0] != stamps[1]:
            self._send_hold(states)
            self.state = DeploymentState.FAULT
            return CoordinatorResult(
                False, self.state, "mismatched_simulation_snapshots"
            )
        if stamps[0] is not None and stamps[0] == self._last_state_stamp:
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
        rises = {
            key: value and not self._previous_buttons.get(key, False)
            for key, value in buttons.items()
        }
        self._previous_buttons = buttons
        try:
            if buttons["stop"] or self.state == DeploymentState.STOPPED:
                return self._stop(states)
            if self.state == DeploymentState.ZERO_TORQUE:
                if rises["start"]:
                    self.state = DeploymentState.DEFAULT_POSE
                    print(
                        "DefaultPose active; roots locked. Press b for LocoMode standing",
                        flush=True,
                    )
                else:
                    zero = np.zeros(29, dtype=np.float32)
                    self._send(states, [PDCommand(zero, zero, zero)] * 2, "zero_torque")
                    return CoordinatorResult(True, self.state, "zero_torque_wait_s")
            elif self.state == DeploymentState.DEFAULT_POSE and rises["B"]:
                for loco, robot, state in zip(self.loco_modes, self.robots, states):
                    loco.reset()
                    robot.prepare(state)
                self.state = DeploymentState.LOCO_STANDING
                print(
                    "LocoMode standing: roots released with paired commands; confirm stability then press a",
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
                self.policy.reset_rollout()
                self.state = DeploymentState.EXECUTING
                print(
                    "A accepted: executing dual ScaleBFM residual reference", flush=True
                )

            if self.state == DeploymentState.DEFAULT_POSE:
                self._send(states, [self.default_command] * 2, "default_pose")
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
                self._send(
                    states,
                    commands,
                    "executing",
                    frame=step.frame,
                    reference_position=reference,
                )
                if step.complete:
                    self._task_finished = True
                    for loco in self.loco_modes:
                        loco.reset()
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
