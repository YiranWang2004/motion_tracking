#!/usr/bin/python3
"""Download completed onboard logs, pair/convert the latest run, and open viewer."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import yaml

from open_onboard_terminals import ROOT, TASKS, ensure_connection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, default='lift')
    p.add_argument('--run-id', help='select a recorded run instead of the latest matched run')
    p.add_argument('--local-only', action='store_true', help='use already downloaded logs, no SSH')
    p.add_argument('--prepare-only', action='store_true', help='download/convert without opening GUI')
    p.add_argument('--replay-speed', type=float, default=1.0)
    p.add_argument('--replay-start-frame', type=int, default=0)
    p.add_argument('--replay-paused', action='store_true')
    args = p.parse_args()
    if os.geteuid() == 0:
        p.error('run as the desktop user, not with sudo')
    config_path = ROOT / TASKS[args.task]
    config = yaml.safe_load(config_path.read_text())
    downloads = ROOT / 'logs/onboard_downloads'
    if not args.local_only:
        topology = yaml.safe_load((config_path.parent / config['network']['wired_topology']).read_text())
        sockets = Path.home() / '.ssh/ctl'
        sockets.mkdir(parents=True, exist_ok=True, mode=0o700)
        for side in 'ab':
            host = config['network'][side].get('host', topology['robot_ip'])
            socket = sockets / f'unitree-g1-{side}-deploy'
            ensure_connection(side, host, topology[f'robot_{side}']['namespace'], socket)
            destination = downloads / side
            destination.mkdir(parents=True, exist_ok=True)
            transport = ['ssh', '-S', str(socket), '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false']
            subprocess.run(['rsync', '-a', '--checksum', '-e', shlex.join(transport),
                            f'unitree@{host}:/home/unitree/wyr/motion_tracking/sim2real/logs/onboard_scalebfm/',
                            str(destination) + '/'], check=True)
    reference = config_path.parent / config['artifact_directory'] / 'reference_bundle.npz'
    with tempfile.TemporaryDirectory(prefix='onboard-replay-') as tmp:
        result_file = Path(tmp) / 'result.txt'
        command = ['uv', 'run', 'python', '-m', 'dual_runtime.onboard_replay',
                   '--downloads', str(downloads), '--output', str(ROOT / 'logs/onboard_replay'),
                   '--artifact-directory', config['artifact_directory'],
                   '--reference-bundle', str(reference), '--result-file', str(result_file)]
        if args.run_id:
            command += ['--run-id', args.run_id]
        subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, check=True)
        replay_path = result_file.read_text().strip()
    command = ['bash', 'scripts/run_dual_scalebfm_viewer.sh', '--replay-log', replay_path,
               '--reference-bundle', str(reference), '--no-vive',
               '--replay-speed', str(args.replay_speed),
               '--replay-start-frame', str(args.replay_start_frame)]
    if args.replay_paused:
        command += ['--replay-paused']
    print('Replay command: ' + shlex.join(command), flush=True)
    if not args.prepare_only:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
