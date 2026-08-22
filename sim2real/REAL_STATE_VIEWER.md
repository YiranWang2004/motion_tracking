# MuJoCo real-state viewer

`real_state_viewer.py` is a read-only digital twin for physical deployment. The
C++ G1 bridge mirrors every policy-state packet to UDP port `55003`; the viewer
uses that copy to update the same 29-DoF G1 model used by sim2sim. It never sends
a command and is shared by tracking and OmniContact deployments.

Start it in an extra terminal after starting `g1_udp_bridge`:

```bash
cd /home/yiranwang/TeleHuman/motion_tracking/sim2real
uv run src/real_state_viewer.py --robot g1
```

The 29 joint angles and floating-base orientation come from measured LowState
and IMU data. Floating-base translation stays at `(0, 0, 0.793)` because
LowState does not provide world position. To inspect joints without IMU tilt or
yaw, add `--no-imu`.

Closing or stalling the viewer does not affect the bridge or motor commands.
If the state stream is lost, the viewer freezes its last pose and prints a stale
warning.
