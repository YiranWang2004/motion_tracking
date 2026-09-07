from types import SimpleNamespace
import numpy as np
from omnicontact.contracts import PDCommand, RobotPose
from dual_runtime.scalebfm_standing import DualScaleBFMStanding
from test_dual_scalebfm_deploy import state, fresh_snapshot
from test_dual_sim_control import runtime, tick
from dual_runtime.policy_coordinator import DeploymentState as S
from dual_runtime.interactive_control import InteractiveDualCoordinator


def test_stationary_default_reference_batch_and_independent_history():
    calls=[]
    class FK:
        def forward(self,q,pose):
            return SimpleNamespace(body_pos_w=pose.position_w[None]+q[0],
                body_quat_wxyz=pose.quaternion_xyzw[[3,0,1,2]][None])
    def infer(histories,pos,quat,live_pos,live_quat,**kwargs):
        calls.append((pos.copy(),quat.copy(),[h.arrays()[-1][-1].copy() for h in histories]))
        return np.full((2,29),.25),np.full((2,29),.7)
    task=SimpleNamespace(scalebfm=SimpleNamespace(infer_batch=infer),kinematics=(FK(),FK()),
        kp=np.full(29,20),kd=np.full(29,2),control_mode=2,time_offsets=np.arange(6),frame=17)
    standing=DualScaleBFMStanding(task,PDCommand(np.full(29,.1),np.full(29,100),np.full(29,4)))
    snapshot=fresh_snapshot(); states=(state(),state())
    standing.reset(states,snapshot)
    targets=standing.reference_pos.copy()
    first=standing.compute(states,snapshot)
    second=standing.compute(states,snapshot)
    assert task.frame==17
    np.testing.assert_allclose(first[0].target_pos,.25)
    np.testing.assert_array_equal(second[1].kp,task.kp)
    for j in range(6): np.testing.assert_array_equal(targets[:,j],targets[:,0])
    np.testing.assert_array_equal(calls[0][2],0)
    np.testing.assert_allclose(calls[1][2],.7)
    # Live drift cannot drag the fixed standing reference along with the robot.
    moved=SimpleNamespace(robot_a=RobotPose(snapshot.robot_a.position_w+1,snapshot.robot_a.quaternion_xyzw,0),
                          robot_b=RobotPose(snapshot.robot_b.position_w+1,snapshot.robot_b.quaternion_xyzw,0))
    standing.compute(states,moved)
    np.testing.assert_array_equal(calls[-1][0],targets)
    standing.reset(states,moved)
    np.testing.assert_allclose(standing.reference_pos[...,2],targets[...,2])
    np.testing.assert_allclose(standing.reference_pos[...,:2],targets[...,:2]+1)


def test_b_and_completion_use_batch_standing_controller(runtime):
    old,clients,policy,_=runtime
    class Standing:
        resets=0
        calls=0
        def reset(self,states,snapshot):self.resets+=1
        def compute(self,states,snapshot):
            self.calls+=1
            return [PDCommand(np.full(29,.1),np.full(29,20),np.full(29,2))]*2
    standing=Standing()
    coordinator=InteractiveDualCoordinator(*old.robots,old.pose_provider,policy,
        enable_a=True,enable_b=True,default_command=old.default_command,standing_policy=standing)
    rt=(coordinator,clients,policy,None)
    tick(rt,'start')
    assert tick(rt,'B').state==S.SCALEBFM_STANDING
    assert standing.resets==1 and standing.calls==1
    assert clients[0].sent[-1][1]['sim_control']['phase']=='scalebfm_standing'
    assert policy.calls==0
    tick(rt,'A')
    policy.complete=True
    assert tick(rt).state==S.SCALEBFM_STANDING
    assert standing.resets==2
    tick(rt)
    assert standing.calls==2
