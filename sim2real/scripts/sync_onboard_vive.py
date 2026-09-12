#!/usr/bin/python3
"""Synchronize non-ignored repository files to both onboard computers."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import shutil
import time

import yaml

from open_onboard_terminals import ROOT, TASKS, ensure_connection


# Input travels over SSH stdin, never through shell interpolation. Resolve the
# destination from each robot's actual task config, not a guessed filename.
REMOTE = r'''
import base64, hashlib, json, os, shutil, sys, tempfile, time
from pathlib import Path
import yaml
request = json.load(sys.stdin)
root = Path(request['root']).resolve()
config = root / request['config']
raw = yaml.safe_load(config.read_text())
target = (config.parent / raw['vive_config']).resolve()
if root not in target.parents:
    raise RuntimeError('remote calibration must be inside the deployment repository')
old = target.read_bytes() if target.exists() else None
digest = lambda data: hashlib.sha256(data).hexdigest() if data is not None else None
before = digest(old)
result = dict(path=str(target), sha256=before, status='checked', backup=None)
if request['operation'] == 'sync':
    data = base64.b64decode(request['data'], validate=True)
    if digest(data) != request['sha256'] or not isinstance(json.loads(data), dict):
        raise RuntimeError('invalid calibration payload or checksum')
    if before != request['expected_old']:
        raise RuntimeError('remote calibration changed since preflight; retry synchronization')
    if before == request['sha256']:
        result['status'] = 'unchanged'
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if old is not None:
            backup = target.with_name(target.name + '.bak.' + str(time.time_ns()))
            with backup.open('xb') as stream:
                stream.write(old)
            shutil.copystat(target, backup)
        fd, name = tempfile.mkstemp(prefix=target.name+'.sync-', dir=str(target.parent))
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(name, (target.stat().st_mode & 0o777) if old is not None else 0o644)
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        result.update(status='updated', backup=str(backup) if backup else None)
    result['sha256'] = digest(target.read_bytes())
    if result['sha256'] != request['sha256']:
        raise RuntimeError('remote checksum verification failed')
print(json.dumps(result))
'''


def remote_request(socket, host, request):
    python = str(Path(request['root']) / 'sim2real/.venv/bin/python')
    result = subprocess.run(
        ['ssh', '-S', str(socket), '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false',
         f'unitree@{host}', shlex.join([python, '-c', REMOTE])],
        input=json.dumps(request), text=True, stdout=subprocess.PIPE, check=True)
    return json.loads(result.stdout)


def repository_files(root, exclusions=()):
    """Include tracked and untracked files, but honor ignores even for tracked files."""
    listing = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '--cached', '--others',
         '--exclude-standard', '-z'], check=True, stdout=subprocess.PIPE).stdout
    candidates = sorted(set(listing.split(b'\0')) - {b''})
    ignored = subprocess.run(
        ['git', '-C', str(root), 'check-ignore', '--no-index', '--verbose', '--stdin', '-z'],
        input=b'\0'.join(candidates) + b'\0', stdout=subprocess.PIPE)
    if ignored.returncode not in (0, 1):
        raise RuntimeError('git ignore evaluation failed')
    fields = ignored.stdout.split(b'\0')
    # Verbose output includes negated matches too; those are explicitly included.
    ignored_paths = {fields[i + 3] for i in range(0, len(fields) - 1, 4)
                     if not fields[i + 2].startswith(b'!')}
    result = []
    for name in candidates:
        path = Path(os.fsdecode(name))
        if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
            raise RuntimeError(f'unsafe repository path: {path}')
        if name in ignored_paths or any(
                path.as_posix() == item or path.as_posix().startswith(item.rstrip('/') + '/')
                for item in exclusions):
            continue
        source = root / path
        if source.is_symlink() or source.is_file():
            result.append(name)
        elif source.exists():
            raise RuntimeError(f'nested repository/submodule requires explicit handling: {path}')
    return b'\0'.join(result) + (b'\0' if result else b'')


def rsync_command(root, destination, *, ssh=None, check=False, backup=None):
    # Explicit leaf paths, no recursive directory scan, no deletion or symlink following.
    command = ['rsync', '-a', '--no-owner', '--no-group', '--checksum',
               '--from0', '--files-from=-', '--itemize-changes', '--protect-args']
    if check:
        command.append('--dry-run')
    elif backup:
        command += ['--backup', '--backup-dir=' + backup]
    if ssh:
        command += ['-e', shlex.join(ssh)]
    return command + [str(root) + '/', destination.rstrip('/') + '/']


def sync_repository(args, net, topology):
    root = ROOT.parent
    if not shutil.which('rsync') or not shutil.which('git'):
        raise RuntimeError('host requires git and rsync')
    if not Path(args.remote_root).is_absolute() or args.remote_root == '/':
        raise RuntimeError('--remote-root must be an absolute repository path, not /')
    files = repository_files(root, args.exclude)
    print(f'Repository: {root}; {files.count(bytes([0]))} non-ignored files', flush=True)
    print('No remote files are deleted. Existing overwritten files are backed up.', flush=True)
    if b'sim2real/pyproject.toml\0' in files or b'sim2real/uv.lock\0' in files:
        print('Includes pyproject.toml / uv.lock: robot-specific dependency settings will '
              'be replaced if different. Use --exclude to preserve them.', flush=True)
    directory = Path.home() / '.ssh/ctl'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    prepared = []
    for side in ('a', 'b'):
        host = net[side].get('host', topology['robot_ip'])
        socket = directory / f'unitree-g1-{side}-deploy'
        ensure_connection(side, host, topology[f'robot_{side}']['namespace'], socket)
        ssh = ['ssh', '-S', str(socket), '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false']
        # Preflight both machines before writing either. No remote Python dependencies.
        subprocess.run(ssh + [f'unitree@{host}',
                              'command -v rsync && test -d ' + shlex.quote(args.remote_root)],
                       check=True)
        prepared.append((side, host, ssh))
    stamp = str(time.time_ns())
    different = False
    for side, host, ssh in prepared:
        backup = args.remote_root.rstrip('/') + '.sync-backups/' + stamp
        print(f'Robot {side.upper()}: {"checking" if args.check else "synchronizing"}...', flush=True)
        if not args.check:
            print(f'  Overwritten-file backups: {backup}', flush=True)
        command = rsync_command(root, f'unitree@{host}:{args.remote_root}',
                                ssh=ssh, check=args.check, backup=backup)
        try:
            result = subprocess.run(command, input=files, stdout=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            print('Synchronization failed; already transferred files remain. Fix the error and '
                  'rerun. No processes were restarted.', file=sys.stderr)
            raise
        print(os.fsdecode(result.stdout), end='', flush=True)
        different |= bool(result.stdout.strip())
    if args.check:
        print('MISMATCH: run without --check to synchronize.' if different else
              'MATCH: selected repository files agree on both robots.')
        return int(different)
    print('Repository synchronization complete. No build, dependency installation or process '
          'restart was performed. Restart both policies after code/config updates; restart the '
          'host Vive publisher too after calibration updates. Rebuild TensorRT engines if '
          'their source models changed.', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--task', choices=TASKS, default='lift')
    choice.add_argument('--config', help='onboard YAML; relative paths are relative to sim2real/')
    parser.add_argument('--remote-root', default='/home/unitree/wyr/motion_tracking')
    parser.add_argument('--check', action='store_true', help='preview checksum differences without writing files')
    parser.add_argument('--calibration-only', action='store_true', help='use the original calibration-only sync')
    parser.add_argument('--exclude', action='append', default=[], metavar='PATH',
                        help='exclude a repository-relative file or directory; repeatable')
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error('run as the desktop user, not with sudo')
    path = (ROOT / (args.config or TASKS[args.task])).resolve()
    try:
        relative = path.relative_to(ROOT.parent)
    except ValueError:
        parser.error('--config must be inside the local motion_tracking repository')
    config = yaml.safe_load(path.read_text())
    net = config['network']
    if net.get('transport') != 'wired_namespace':
        parser.error('this script uses the wired_namespace SSH connections')
    topology = yaml.safe_load((path.parent / net['wired_topology']).read_text())
    if not args.calibration_only:
        return sync_repository(args, net, topology)
    if args.exclude:
        parser.error('--exclude applies only to repository synchronization')
    source = (path.parent / config['vive_config']).resolve()
    data = source.read_bytes()
    if not isinstance(json.loads(data), dict):
        parser.error('Vive calibration must be a JSON object')
    digest = hashlib.sha256(data).hexdigest()
    topology = yaml.safe_load((path.parent / net['wired_topology']).read_text())
    print(f'Host calibration: {source}\nSHA256: {digest}', flush=True)
    directory = Path.home()/'.ssh/ctl'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    prepared = []
    # Verify both SSH connections and both destination configs before writing.
    for side in ('a', 'b'):
        host = net[side].get('host', topology['robot_ip'])
        socket = directory/f'unitree-g1-{side}-deploy'
        print(f'Checking robot {side.upper()}...', flush=True)
        ensure_connection(side, host, topology[f'robot_{side}']['namespace'], socket)
        request = dict(root=args.remote_root, config=str(relative), operation='check')
        result = remote_request(socket, host, request)
        print(f"{side.upper()}: {result['path']}\n  SHA256: {result['sha256']}", flush=True)
        prepared.append((side, socket, host, request, result))
    if args.check:
        matches = all(item[4]['sha256'] == digest for item in prepared)
        print('MATCH: all three files agree.' if matches else 'MISMATCH: run without --check to synchronize.')
        return 0 if matches else 1
    for side, socket, host, request, old in prepared:
        request.update(operation='sync', data=base64.b64encode(data).decode(),
                       sha256=digest, expected_old=old['sha256'])
        try:
            result = remote_request(socket, host, request)
        except (subprocess.CalledProcessError, ValueError):
            print(f'{side.upper()} synchronization failed. Earlier successful updates remain in place; '
                  'fix the connection/error and rerun before restarting the deployment.', file=sys.stderr)
            raise
        print(f"{side.upper()}: {result['status']}; SHA256 verified: {result['sha256']}", flush=True)
        if result['backup']:
            print(f"  Backup: {result['backup']}", flush=True)
    if source.read_bytes() != data:
        raise RuntimeError('host calibration changed during synchronization; rerun before deployment')
    print('Synchronization complete. No processes were restarted.\n'
          'Restart the host Vive publisher and BOTH onboard policies to load this calibration.\n'
          'The bridge and wired relay processes do not need a calibration reload.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
