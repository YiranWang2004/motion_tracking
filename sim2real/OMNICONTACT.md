# OmniContact carry-box sim2real

This integration runs only the OmniContact `carrybox` skill. It uses the
existing `g1_udp_bridge` for Unitree DDS and does not modify the PMG
`src/deploy.py` entry point.

## Data flow

Two Vive Trackers must share one SteamVR space:

- the robot Tracker is rigidly attached to the pelvis;
- the object Tracker is rigidly attached to the box.

The calibrated poses, the box half extents and `goal_position_w` use one common
world frame. Joint state and IMU angular velocity come from the G1 bridge.
OmniContact inference and bridge commands run at 50 Hz.
The checked configuration targets a 29-DoF G1 in `mode_pr=0` and does not
command Dex3 finger joints.

The runner is safe by default: without `--act` it receives state and evaluates
the policy but sends no UDP command to the bridge. Motor output additionally
requires the literal confirmation `ENABLE_MOTORS`.

## Install

From `motion_tracking/sim2real`:

```bash
uv sync
```

On the workstation that reads SteamVR locally:

```bash
uv sync --extra vive
```

The policy, its original YAML and an FK-only derivative of the original G1 XML
are stored in `config/g1/omnicontact/`. Visual mesh geoms were removed from the
FK XML; the joint/body transform tree is unchanged. The exact carry-box
sim2sim scene, its robot/ghost XML includes and all referenced meshes are
vendored under `config/g1/assets/`; no sibling `OmniContact_sim2sim` checkout is
required.

## Run motion_tracking sim2sim

This mode runs the same `deploy_omnicontact.py`, policy adapter, safety limiter,
50 Hz state loop and UDP command path used for the robot.  Only the low-level
side is replaced: `sim2sim.py` acts as the C++/DDS bridge and advances the
original OmniContact carry-box MuJoCo scene.

Open terminal 1:

```bash
cd <repo>/sim2real
uv run src/sim2sim.py \
  --robot g1 \
  --bridge-config config/g1/bridge_omnicontact.yaml
```

Use `--headless` only when no MuJoCo window is wanted.

Open terminal 2:

```bash
cd <repo>/sim2real
uv run src/deploy_omnicontact.py \
  --robot g1 \
  --pose-source sim \
  --goal-position 1.0 1.0 0.15 \
  --max-target-delta 1.0 \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

Then operate in this order:

1. Wait for terminal 2 to print `ZERO TORQUE`.
2. In terminal 1 press `s`; the virtual robot is moved through the same
   default-pose preparation state as real deployment.
3. Wait for terminal 2 to print `Hold default pose`.
4. In terminal 1 press `a`; CFgen creates the carry-box reference and policy
   control starts. A visible crouch during grasping and placing is expected.
5. In terminal 1 press `x` to request Stop/damping. Use `Ctrl+C` as the final
   process stop if needed.

The simulator publishes robot pelvis and box poses in the same state packet as
joint/IMU data, so `--pose-source sim` does not use Vive. The checked config
sets the box center to `[1, 0, 0.15]`, the goal to the command-line value, the
physics loop to 200 Hz, and the deployment interface to 50 Hz. Its lockstep
option removes Python process scheduling jitter by pairing each policy command
with one state frame; it does not bypass UDP or the deployment safety path. Do
not run the real C++ bridge on UDP ports 55001/55002 at the same time.

The simulation command also carries the original runner's visualization-only
reference snapshot. The yellow wrist/torso/ankle markers, gray ghost robot,
orange ghost box, red/green contact colors and start/goal planes therefore
follow the same CFgen frame as the policy instead of remaining at XML defaults.

`--max-target-delta 1.0` is a sim-only override for policy parity: the original
OmniContact runner produces valid target changes up to about 0.96 rad per
policy tick, while the real-deployment default remains the more conservative
0.15 rad. Omitting the override is useful for testing that stricter real safety
setting, but it materially changes the trained closed loop and may fail the
carry in simulation. Joint-position and torque limits remain active in both
modes.

## Calibrate Vive

List Trackers:

```bash
uv run --extra vive python scripts/list_vive_trackers.py --seconds 5
```

Calibrate the SteamVR standing frame:

```bash
uv run --extra vive python scripts/calibrate_vive_world.py \
  --serial <TRACKER_SERIAL> \
  --distance-x 0.5 \
  --distance-y 0.5
```

Copy `config/g1/omnicontact_vive.example.json` to a local deployment file.
Measure all three transforms, the box half extents and the box-center goal.
Only then set `calibration_confirmed` to the JSON boolean `true`.

## Start the G1 bridge

Build the existing bridge as documented by the repository, then run it on the
wired G1 interface:

```bash
cd ../g1_sim2real
bash scripts/build.sh
G1_NET=<WIRED_INTERFACE> bash scripts/run_bridge.sh
```

The bridge and Python runner must use the same UDP ports and policy joint list.
The checked-in G1 bridge configuration enables a 200 ms command watchdog.
After its first valid command, a timeout sends damping and latches the bridge;
restart the bridge to re-arm it.

## Run the read-only sim2real twin

Run the twin as a separate process on the SteamVR workstation. Do not put
MuJoCo or OpenVR in `run_bridge.sh`: the C++ bridge's DDS and watchdog timing
must not depend on a GUI process.

With the checked-in local port layout, use three terminals:

```bash
# Terminal 1: real-time G1 bridge
cd <repo>/g1_sim2real
G1_NET=<WIRED_INTERFACE> bash scripts/run_bridge.sh

# Terminal 2: read-only MuJoCo twin
cd <repo>/sim2real
uv run --extra vive python scripts/real_omnicontact_viewer.py \
  --vive-config config/g1/omnicontact_vive.json

# Terminal 3: policy deployment (start observation-only first)
cd <repo>/sim2real
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --run-seconds 30
```

The window combines three independent read-only inputs in the same carry-box
XML used by sim2sim:

- local Vive poses drive the real pelvis, real box and Tracker markers;
- bridge mirror UDP `55003` drives the real robot's 29 measured joints;
- deployment UDP `55004` drives CFgen wrist/torso/ankle references, ghost
  robot, ghost box, contacts, and start/goal planes.

The deployment process still uses bridge state `55001` and sends motor commands
only to `55002`. Closing or stalling the twin cannot enter that command path.
Reference and ghost geometry is hidden if the `55004` stream is stale (default
`0.5` s). `--state-port 0` disables measured joints;
`--no-visualization` disables the policy overlay.

If the twin runs on another computer, set `udp.state_mirror_host` in
`g1_sim2real/config/g1_bridge.yaml` to that computer's wired IP, and pass the
same IP to deployment with `--visualization-host`. Bind the twin receivers with
`--state-host 0.0.0.0 --visualization-host 0.0.0.0`. `G1_NET` selects only the
Unitree DDS interface; it does not route these viewer streams.

## Local OpenVR mode

First run observation-only:

```bash
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --run-seconds 30
```

Only after checking the logged pelvis/object/goal coordinates, suspend or
secure the robot and explicitly enable commands:

```bash
uv run --extra vive python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source local \
  --act \
  --confirm-actuation ENABLE_MOTORS
```

The remote sequence is Start (zero torque to default-pose transition), then A
while both Tracker poses are fresh (start carry-box). Stop sends damping.

## Remote Vive publisher mode

Use the same random token on the workstation and policy host:

```bash
export OMNICONTACT_POSE_TOKEN='<LONG_RANDOM_TOKEN>'
```

On the policy host:

```bash
uv run python src/deploy_omnicontact.py \
  --vive-config config/g1/omnicontact_vive.json \
  --pose-source udp \
  --udp-bind 0.0.0.0 \
  --allowed-sender-ip <WORKSTATION_WIRED_IP> \
  --run-seconds 30
```

On the SteamVR workstation:

```bash
uv run --extra vive python scripts/publish_vive_poses.py \
  --vive-config config/g1/omnicontact_vive.json \
  --target-ip <POLICY_HOST_WIRED_IP> \
  --bind-ip <WORKSTATION_WIRED_IP>
```

The UDP pose packet carries an atomic robot/object pair and is authenticated
with HMAC-SHA256. Receiver-local arrival time determines freshness.

## Runtime safety behavior

- stale/missing Tracker pair: freeze the reference and hold measured joints;
- non-finite or wrong-shaped policy/state data: reject and exit to damping;
- policy target: clip to OmniContact joint limits and per-cycle target delta;
- bridge-state timeout or pose-provider failure: send damping and exit;
- C++ command watchdog timeout: latch damping until the bridge is restarted;
- shutdown/Stop: send damping;
- no `--act`: never send a command packet.

Before ground testing, validate joint order, gains, Tracker loss, bridge loss,
Stop, process termination and target limits with the robot suspended.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
