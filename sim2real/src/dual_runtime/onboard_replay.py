"""Convert paired onboard recordings to the existing dual viewer format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .visualization_replay import DualScaleBFMReplay


def select_pair(root, artifact_directory, run_id=None):
    recordings = {}
    for side in ('a', 'b'):
        records = {}
        for path in sorted((Path(root) / side).glob('*/metadata.json')):
            meta = json.loads(path.read_text())
            run = meta.get('run_id')
            if (meta.get('robot') != side or not run or not meta.get('ticks')
                    or not path.with_name('rollout.npz').is_file()
                    or meta.get('config', {}).get('artifact_directory') != artifact_directory):
                continue
            if run in records:
                raise ValueError(f'Duplicate {side} recording for run {run}')
            records[run] = path.parent
        recordings[side] = records
    common = recordings['a'].keys() & recordings['b'].keys()
    if run_id is not None:
        common &= {run_id}
    if not common:
        raise ValueError('No completed, nonempty A/B recordings with a common run_id')
    # Only compare dates from A: the robots can have different wall clocks.
    selected = max(common, key=lambda run: recordings['a'][run].name)
    return recordings['a'][selected], recordings['b'][selected]


def merge_arrays(a, b, *, max_skew_s=0.03):
    for data in (a, b):
        n = len(data['time_ns'])
        if n == 0 or np.any(np.diff(data['time_ns']) <= 0):
            raise ValueError('Recording must contain increasing, nonempty timestamps')
        for name, tail in {'q': (29,), 'target': (29,), 'poses': (3, 7),
                           'wall_time_ns': (), 'peer_clock_offset_s': (),
                           'peer_clock_uncertainty_s': (), 'frame': (), 'phase': ()}.items():
            if data[name].shape != (n, *tail):
                raise ValueError(f'Invalid onboard field shape: {name}')
            if name != 'phase' and not np.isfinite(data[name]).all():
                raise ValueError(f'Nonfinite onboard field: {name}')
    if not 0 < max_skew_s <= 0.1:
        raise ValueError('Time matching tolerance must be in (0, 0.1] seconds')
    origin = int(a['wall_time_ns'][0])
    ta = (a['wall_time_ns'] - origin) * 1e-9
    # B's recorded offset is A wall clock minus B wall clock.
    tb = (b['wall_time_ns'] - origin) * 1e-9 + b['peer_clock_offset_s']
    if np.any(np.diff(ta) <= 0) or np.any(np.diff(tb) <= 0):
        raise ValueError('Clock discontinuity: cannot safely align these recordings')
    right = np.searchsorted(tb, ta).clip(0, len(tb) - 1)
    left = (right - 1).clip(0, len(tb) - 1)
    ib = np.where(abs(tb[left] - ta) <= abs(tb[right] - ta), left, right)
    skew = tb[ib] - ta
    keep = ((ta >= tb[0]) & (ta <= tb[-1])
            & (abs(skew) + b['peer_clock_uncertainty_s'][ib] <= max_skew_s))
    ia = np.flatnonzero(keep)
    ib = ib[keep]
    if not len(ia):
        raise ValueError('No overlapping samples within the time matching tolerance')
    poses = a['poses'][ia]
    output = dict(
        time_ns=a['time_ns'][ia],
        q=np.stack((a['q'][ia], b['q'][ib]), axis=1),
        command_target=np.stack((a['target'][ia], b['target'][ib]), axis=1),
        robot_position_w=poses[:, :2, :3], robot_quat_xyzw=poses[:, :2, 3:],
        object_position_w=poses[:, 2, :3], object_quat_xyzw=poses[:, 2, 3:],
        frame=a['frame'][ia], frame_b=b['frame'][ib],
        deployment_state=np.asarray([f"a:{pa} b:{pb}" for pa, pb in
                                     zip(a['phase'][ia], b['phase'][ib])]),
        matched_b_skew_s=skew[keep], source_index_a=ia, source_index_b=ib,
    )
    if 'residual' in a and 'residual' in b:
        output['residual'] = np.stack((a['residual'][ia], b['residual'][ib]), axis=1)
    if 'recovery_mode' in a and 'recovery_mode' in b:
        output['deployment_state'] = np.asarray([
            f'a:{pa}/{ra} b:{pb}/{rb}' for pa, ra, pb, rb in
            zip(a['phase'][ia], a['recovery_mode'][ia], b['phase'][ib], b['recovery_mode'][ib])])
    return output


def convert(a_dir, b_dir, output_root, reference):
    paths = [Path(a_dir), Path(b_dir)]
    meta = [json.loads((p / 'metadata.json').read_text()) for p in paths]
    for side, m in zip(('a', 'b'), meta):
        if m.get('robot') != side:
            raise ValueError('Incorrect robot role in recording')
    for key in ('run_id', 'identity', 'calibration_id', 'pose_stream'):
        if not meta[0].get(key) or meta[0][key] != meta[1].get(key):
            raise ValueError(f'A/B recording mismatch: {key}')
    arrays = []
    for p in paths:
        with np.load(p / 'rollout.npz', allow_pickle=False) as data:
            arrays.append({k: data[k] for k in data.files})
    merged = merge_arrays(*arrays)
    # Use the source directory's safe filename, never an unchecked metadata path.
    directory = Path(output_root) / paths[0].name
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'rollout.npz', **merged)
    summary = dict(
        run_id=meta[0]['run_id'], reference=str(Path(reference).resolve()),
        reference_alignment='motion_world',
        start_frame=meta[0]['policy_config'].get('start_frame', 1),
        pose_source='onboard_recording', sources=[str(p.resolve()) for p in paths],
        source_metadata=meta, alignment='nearest B sample in A wall clock, using recorded B peer offset',
        pose_selection='A recorded common Vive snapshot; no raw tracker poses recorded',
        reference_frame_selection='A task frame for both reference ghosts; frame_b retained in NPZ',
        frames=len(merged['time_ns']), dropped_a_frames=len(arrays[0]['time_ns'])-len(merged['time_ns']),
        max_match_skew_ms=float(abs(merged['matched_b_skew_s']).max()*1000),
    )
    (directory / 'metadata.json').write_text(json.dumps(summary, indent=2))
    replay = DualScaleBFMReplay(directory / 'rollout.npz', reference_bundle=reference)
    for i in {0, replay.frame_count // 2, replay.frame_count - 1}:
        replay.visualization_packet(i)
    print(f"Run {summary['run_id']}: {replay.frame_count} frames, {replay.duration_s:.2f}s; "
          f"max pairing skew {summary['max_match_skew_ms']:.2f} ms", flush=True)
    print(f"A: {paths[0].name}\nB: {paths[1].name}", flush=True)
    for side, m in zip('AB', meta):
        print(f"{side} exit: {m.get('exit_reason')}", flush=True)
    return directory / 'rollout.npz'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downloads', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--artifact-directory', required=True)
    p.add_argument('--reference-bundle', required=True)
    p.add_argument('--run-id')
    p.add_argument('--result-file', required=True)
    args = p.parse_args()
    a, b = select_pair(args.downloads, args.artifact_directory, args.run_id)
    result = convert(a, b, args.output, args.reference_bundle)
    Path(args.result_file).write_text(str(result.resolve()))


if __name__ == '__main__':
    main()
