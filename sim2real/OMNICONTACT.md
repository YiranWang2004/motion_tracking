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
FK XML; the joint/body transform tree is unchanged.

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
