"""Optional cross-interpreter check against the neighboring training checkout."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dual_runtime.constants import KEY_BODY_NAMES
from dual_runtime.observation import build_pelvis_residual_observation, projected_gravity
from dual_runtime.reference import ReferenceFrame
from omnicontact.contracts import ObjectPose, RobotPose
from omnicontact.runtime import BridgeState


def test_full_201_vector_matches_current_training_builder(tmp_path):
    training = Path(__file__).resolve().parents[3] / "Dual_G1_MJ"
    python = training / ".venv/bin/python"
    if not python.exists():
        pytest.skip("neighboring training Python environment not available")
    rng = np.random.default_rng(18)
    values = {}
    for name, shape in dict(q=(2, 29), dq=(2, 29), reference_q=(2, 29), reference_dq=(2, 29),
                            gyro=(2, 3), reference_omega=(2, 3), default=(29,),
                            previous=(2, 29), target=(2, 29), body_pos=(2, 14, 3),
                            box_pos=(3,)).items():
        values[name] = rng.normal(size=shape).astype(np.float32)
    for name, shape in dict(imu=(2, 4), body_quat=(2, 14, 4),
                            reference_quat=(2, 14, 4), box_quat=(4,)).items():
        value = rng.normal(size=shape).astype(np.float32)
        values[name] = value / np.linalg.norm(value, axis=-1, keepdims=True)
    values["gravity"] = np.stack([projected_gravity(q) for q in values["imu"]])
    input_file, output = tmp_path / "inputs.npz", tmp_path / "training.npy"
    np.savez(input_file, **values)
    # This process uses the training repo's actual class, math functions,
    # feature ordering and observation sanitation; it never creates a simulator.
    code = r'''
import sys
from types import SimpleNamespace as NS
import numpy as np
import torch
from dual_g1_mj.envs.dual_g1_scalebfm_residual_object_marl_env import DualG1ScaleBFMResidualObjectMarlEnv as Env
from dual_g1_mj.asset_zoo.robots.g1.g1_constants import KEY_BODY_NAMES
data=np.load(sys.argv[1]); v={k:torch.from_numpy(data[k]) for k in data.files}
env=Env.__new__(Env); env.scene=NS(num_envs=1); env.agents=['robot_0','robot_1']
env.cfg=NS(box_observation='actual', actor_observation='current', obs_clip=100.,
           debug_finite_steps=0, anchor_angular_velocity_frame='reference-anchor')
env.isaac_joint_perm=torch.arange(29); env.residual_joint_indices=torch.arange(29)
env.robot_states={}; env.robots={}; env.actions={}; env._sonic_target={}
anchor=KEY_BODY_NAMES.index('torso_link')
for i,a in enumerate(env.agents):
    env.robot_states[a]=NS(robot_body_pos_w=v['body_pos'][i:i+1], robot_body_quat_w=v['body_quat'][i:i+1],
        robot_joint_pos=v['q'][i:i+1], robot_joint_vel=v['dq'][i:i+1],
        joint_pos=v['reference_q'][i:i+1],joint_vel=v['reference_dq'][i:i+1],
        anchor_quat_w=v['reference_quat'][i:i+1,anchor], anchor_ang_vel_w=v['reference_omega'][i:i+1],
        base_ang_vel=v['gyro'][i:i+1])
    env.robots[a]=NS(data=NS(projected_gravity_b=v['gravity'][i:i+1],
        default_joint_pos=v['default'][None],default_joint_vel=torch.zeros(1,29)))
    env.actions[a]=v['previous'][i:i+1];env._sonic_target[a]=v['target'][i:i+1]
env.object_states=NS(object_body_pos_w=v['box_pos'][None],object_body_quat_w=v['box_quat'][None])
obs=env._build_residual_observations()
np.save(sys.argv[2],torch.cat([obs[a] for a in env.agents]).numpy())
'''
    subprocess.run([str(python), "-c", code, str(input_file), str(output)], check=True,
                   cwd=training, env={**os.environ, "PYTHONPATH": str(training / "src")},
                   capture_output=True, text=True)
    actual = []
    poses = [RobotPose(values["body_pos"][i, 0], values["body_quat"][i, 0][[1,2,3,0]], 0.) for i in range(2)]
    for i in range(2):
        ref = ReferenceFrame(values["reference_q"][i], values["reference_dq"][i],
            np.zeros((14,3)), values["reference_quat"][i], values["reference_omega"][i],
            values["box_pos"], values["box_quat"])
        state = BridgeState(values["q"][i], values["dq"][i], values["imu"][i],
                            values["gyro"][i], {}, None, 0, 0)
        actual.append(build_pelvis_residual_observation(state=state, reference=ref,
            own_pelvis=poses[i], partner_pelvis=poses[1-i],
            object_pose=ObjectPose(values["box_pos"], values["box_quat"][[1,2,3,0]], [.5,.15,.15],0.),
            scalebfm_target=values["target"][i], previous_residual=values["previous"][i],
            default_q=values["default"]))
    np.testing.assert_allclose(np.stack(actual), np.load(output), atol=2e-6, rtol=2e-6)
