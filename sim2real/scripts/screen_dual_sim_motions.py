#!/usr/bin/env python3
"""Screen full motions with production physics/controllers and in-process transport.

This does not validate UDP delivery or wall-clock deadlines; confirm winners using
run_dual_scalebfm_sim2sim.sh. No hardware sockets are opened. Safety limits remain
those in the supplied config. Each motion supplies its own box dimensions.
"""
import argparse
import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deploy_dual_scalebfm_residual import resolve_artifacts, resolve
from dual_runtime.scalebfm_residual_policy import DualScaleBFMResidualPolicy
from dual_runtime.reference import DualReferenceBundle
from dual_runtime.runtime_config import shared_control_settings
from dual_runtime.interactive_control import InteractiveDualCoordinator, load_default_command
from dual_runtime.robot_session import RobotSession, RobotSessionConfig
from dual_runtime.constants import POLICY_JOINT_NAMES
from dual_runtime.sim_pose_provider import DualSimulationPoseProvider
from dual_scalebfm_sim2sim import DualScaleBFMSim2Sim
from omnicontact.runtime import BridgeState, MotionBridgeClient
from dual_runtime.scalebfm_standing import DualScaleBFMStanding
from common.udp_transport import _command_payload


def run_motion(config_path, raw, policy, motion, standing_ticks=250):
    policy.reference = DualReferenceBundle(motion)
    policy.initialized = False
    policy.reset_rollout()
    raw['simulation']['box_half_extents'] = policy.reference.box_half_extents.tolist()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', dir=config_path.parent) as cfg:
        yaml.safe_dump(raw, cfg); cfg.flush()
        low = tuple(SimpleNamespace(send_state=lambda **kw: None, close=lambda: None) for _ in range(2))
        sim = DualScaleBFMSim2Sim(cfg.name, headless=True, reference_bundle=motion, transports=low, scalebfm_only=not policy.residual_enabled)
    provider = DualSimulationPoseProvider()
    commands = [None, None]
    clients = []
    for index, binding in enumerate(sim.bindings):
        client = MotionBridgeClient.__new__(MotionBridgeClient)
        client.pose_sink = None
        def send_command(index=index, **kwargs):
            commands[index] = sim.parse_command(_command_payload(**kwargs))
            return 1
        client.transport = SimpleNamespace(send_command=send_command, close=lambda: None)
        def read_next(timeout, binding=binding):
            q, dq, quat, gyro, _ = sim._robot_state(binding)
            return BridgeState(q_lab=q, dq_lab=dq, quat_wxyz=quat, gyro=gyro,
                buttons=sim._button_snapshot.copy(), state_receive_time_ns=sim._snapshot_id+1,
                packet_seq=sim._snapshot_id+1, packet_arrival_ns=time.monotonic_ns())
        client.read_next = read_next
        clients.append(client)
    settings = shared_control_settings(raw)
    assets = resolve(config_path.parent, settings['standing_asset_dir'])
    limits = yaml.safe_load(resolve(config_path.parent, raw['joint_limits_source']).read_text())
    sessions = [RobotSession(RobotSessionConfig(robot_id=name, udp={},
        lower=limits['joint_pos_lowerlimit_lab'], upper=limits['joint_pos_upperlimit_lab'],
        kp=policy.kp, kd=policy.kd, torque_limit=policy.torque_limit,
        damping_kd=settings['damping_kd'], max_target_delta=settings['phase_target_delta']['default_pose']),client=client)
        for name,client in zip(('a','b'),clients)]
    ticks = round(settings['default_pose_duration_s']*raw['control_frequency_hz'])
    pre = raw.get('preflight',{})
    coordinator = InteractiveDualCoordinator(*sessions, provider, policy, enable_a=True, enable_b=True,
        standing_policy=DualScaleBFMStanding(policy,load_default_command(assets)),
        default_command=load_default_command(assets), transition_ticks=ticks,
        phase_target_delta=settings['phase_target_delta'], require_button_release=True,
        max_tilt_rad=settings['max_tilt_rad'], task_safety=settings['task_safety'] if policy.residual_enabled else None, **pre)
    start_b = 50 + ticks + 50
    start_a = start_b + standing_ticks
    complete_tick = None
    report = {'motion':str(motion),'frames':policy.reference.frames,
              'box_half_extents':policy.reference.box_half_extents.tolist(),'passed':False,'last_frame':-1}
    began = time.monotonic()
    try:
        for i in range(start_a + policy.reference.frames + 300):
            key = {50:'s',start_b:'b',start_a:'a'}.get(i)
            if key: sim.key_callback(ord(key))
            sim.publish_state_pair(sim._snapshot_id+1)
            provider.ingest_bridge_state({'dual_sim_pose':sim.pose_payload()})
            result = coordinator.step()
            if not result.ok: raise RuntimeError(result.reason)
            if result.policy_step is not None:
                report['last_frame'] = result.policy_step.frame
                if result.policy_step.complete: complete_tick = i
            sim.step_policy_interval(tuple(commands)); sim._snapshot_id += 1
            if complete_tick is not None and i-complete_tick >= 150:
                report['passed'] = True
                report['reason'] = 'full_reference_then_3s_scalebfm_standing'
                break
        else: report['reason'] = 'did_not_complete'
    except Exception as exc:
        report['reason'] = f'{type(exc).__name__}: {exc}'
    finally:
        sim.close(); provider.stop()
    report['wall_seconds'] = time.monotonic()-began
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--motions',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--stop-on-pass',action='store_true')
    parser.add_argument('--scalebfm-only', action='store_true', help='skip residual checkpoint and actor')
    args=parser.parse_args()
    cfg=Path(args.config).resolve(); raw=yaml.safe_load(cfg.read_text())
    artifacts=resolve_artifacts(cfg,raw,include_residual=not args.scalebfm_only)
    policy=DualScaleBFMResidualPolicy(**artifacts, device='cpu', residual_enabled=not args.scalebfm_only,
        inference_precision=raw.get('inference_precision','fp32'),control_mode=raw['control_mode'],
        future_step=raw['future_step'],residual_scale=raw['residual_scale'],start_frame=raw['start_frame'],
        reference_alignment=raw['reference_alignment'],torch_num_threads=raw['torch_num_threads'])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('a') as results, output.with_suffix('.console.log').open('a') as console:
        for motion in sorted(Path(args.motions).glob('motion_*.npz'))[args.start:args.start+args.count]:
            try:
                with contextlib.redirect_stdout(console): report=run_motion(cfg,raw,policy,motion)
            except Exception as exc: report={'motion':str(motion),'passed':False,'reason':f'{type(exc).__name__}: {exc}'}
            report['policy_mode'] = 'scalebfm_only' if args.scalebfm_only else 'scalebfm_residual'
            line=json.dumps(report); print(line,flush=True);results.write(line+'\n');results.flush()
            if report['passed'] and args.stop_on_pass: break

if __name__=='__main__': main()
