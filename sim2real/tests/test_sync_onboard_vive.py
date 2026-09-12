"""Exercise the remote update protocol locally; no SSH or robot mutation."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('configobj'), 'requires desktop Python dependencies')
class CalibrationSyncTest(unittest.TestCase):
    def setUp(self):
        scripts = Path(__file__).resolve().parents[1]/'scripts'
        sys.path.insert(0, str(scripts))
        import sync_onboard_vive
        self.code = sync_onboard_vive.REMOTE
        self.addCleanup(lambda: sys.path.remove(str(scripts)))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/'task.yaml').write_text('vive_config: calibration.json\n')
        self.target = self.root/'calibration.json'
        self.old = b'{"old": true}\n'
        self.new = b'{\n  "new": true\n}\n'
        self.target.write_bytes(self.old)
        self.request = dict(root=str(self.root), config='task.yaml', operation='sync',
                            data=base64.b64encode(self.new).decode(),
                            sha256=hashlib.sha256(self.new).hexdigest(),
                            expected_old=hashlib.sha256(self.old).hexdigest())

    def run_request(self):
        return subprocess.run([sys.executable, '-c', self.code], input=json.dumps(self.request),
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_atomic_update_backup_and_idempotence(self):
        result = self.run_request()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(self.target.read_bytes(), self.new)
        self.assertEqual(Path(report['backup']).read_bytes(), self.old)
        self.request['expected_old'] = self.request['sha256']
        report = json.loads(self.run_request().stdout)
        self.assertEqual(report['status'], 'unchanged')
        self.assertEqual(len(list(self.root.glob('*.bak.*'))), 1)

    def test_check_does_not_write(self):
        self.request['operation'] = 'check'
        result = self.run_request()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.target.read_bytes(), self.old)
        self.assertFalse(list(self.root.glob('*.bak.*')))

    def test_concurrent_edit_or_corruption_does_not_overwrite(self):
        self.target.write_bytes(b'{"edited": true}')
        self.assertNotEqual(self.run_request().returncode, 0)
        self.assertEqual(self.target.read_bytes(), b'{"edited": true}')
        self.target.write_bytes(self.old)
        self.request['sha256'] = 'wrong'
        self.assertNotEqual(self.run_request().returncode, 0)
        self.assertEqual(self.target.read_bytes(), self.old)
        self.assertFalse(list(self.root.glob('*.bak.*')))


@unittest.skipUnless(importlib.util.find_spec('configobj'), 'requires desktop Python dependencies')
class RepositorySyncTest(unittest.TestCase):
    def test_ignore_rules_and_real_transfer(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        sys.path.insert(0, str(scripts))
        self.addCleanup(lambda: sys.path.remove(str(scripts)))
        from sync_onboard_vive import repository_files, rsync_command
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            dest = Path(directory) / 'destination'
            root.mkdir()
            dest.mkdir()
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('*.bin\n!required.bin\nlogs/\n.venv/\n')
            for name in ['required.bin', 'ignored.bin', 'tracked.bin', 'file with spaces.txt']:
                (root / name).write_text('new')
            subprocess.run(['git', '-C', str(root), 'add', '-f', 'tracked.bin'], check=True)
            (root / 'nested').mkdir()
            (root / 'nested/.gitignore').write_text('secret\n')
            (root / 'nested/secret').write_text('ignored')
            (root / 'nested/code.py').write_text('code')
            (root / 'link').symlink_to('file with spaces.txt')
            (dest / 'required.bin').write_text('old')
            (dest / 'robot.engine').write_text('local engine')
            (dest / '.venv').mkdir()
            (dest / '.venv/local').write_text('local environment')
            files = repository_files(root)
            selected = files.split(b'\0')
            self.assertIn(b'required.bin', selected)
            self.assertIn(b'file with spaces.txt', selected)
            for name in [b'ignored.bin', b'tracked.bin', b'nested/secret']:
                self.assertNotIn(name, selected)
            self.assertNotIn(b'nested/code.py', repository_files(root, ['nested']).split(b'\0'))
            preview = subprocess.run(rsync_command(root, str(dest), check=True),
                                     input=files, stdout=subprocess.PIPE, check=True)
            self.assertTrue(preview.stdout)
            self.assertEqual((dest / 'required.bin').read_text(), 'old')
            backup = Path(directory) / 'backup'
            subprocess.run(rsync_command(root, str(dest), backup=str(backup)),
                           input=files, stdout=subprocess.PIPE, check=True)
            self.assertEqual((dest / 'required.bin').read_text(), 'new')
            self.assertEqual((backup / 'required.bin').read_text(), 'old')
            self.assertEqual((dest / 'robot.engine').read_text(), 'local engine')
            self.assertEqual((dest / '.venv/local').read_text(), 'local environment')
            self.assertTrue((dest / 'link').is_symlink())
            self.assertFalse((dest / '.git').exists())
            again = subprocess.run(rsync_command(root, str(dest), check=True),
                                   input=files, stdout=subprocess.PIPE, check=True)
            self.assertFalse(again.stdout)


if __name__ == '__main__':
    unittest.main()
