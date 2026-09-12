"""Desktop launcher checks; run with host Python (PyYAML + ConfigObj)."""
import argparse
import importlib.util
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("configobj"), "requires host ConfigObj")
class OnboardTerminalsTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("onboard_terminals", root / "scripts/open_onboard_terminals.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.args = argparse.Namespace(task="lift", remote_root="/home/unitree/a path's repo",
                                       robot_interface="eth0", actuate=False, preview=True)

    def test_policy_uses_paired_task_and_explicit_actuation(self):
        command = self.module.remote_command(self.args, "b", "policy")
        subprocess.run(["bash", "-n", "-c", command], check=True)
        self.assertIn("onboard_scalebfm_wired_lift.yaml --robot b", command)
        self.assertNotIn("--actuate", command)
        self.assertIn("sport = :55002", command)
        self.args.actuate = True
        command = self.module.remote_command(self.args, "a", "policy")
        self.assertIn("--actuate --confirm-actuation ENABLE_MOTORS", command)
        ssh = self.module.ssh_command("/tmp/socket name", "192.168.123.164", command)
        self.assertIn("ProxyCommand=false", ssh)
        self.assertEqual(shlex.split(ssh[-1]), ["bash", "-lc", command])

    def test_distinct_robot_interfaces_reach_bridge_commands(self):
        self.args.robot_interface_a = "enP8p1s0"
        self.args.robot_interface_b = "eth0"
        for side, interface in (("a", "enP8p1s0"), ("b", "eth0")):
            command = self.module.remote_command(self.args, side, "bridge")
            self.assertIn(f"G1_NET={interface} bash", command)
            subprocess.run(["bash", "-n", "-c", command], check=True)

    def test_preview_layout_contains_four_distinct_roles(self):
        from configobj import ConfigObj
        endpoints = {s: ("192.168.123.164", f"g1{s}", Path(f"/tmp/{s}")) for s in ("a", "b")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config"
            self.module.write_layout(path, self.args, endpoints)
            config = ConfigObj(str(path))
        self.assertEqual(config["global_config"]["broadcast_default"], "off")
        terminals = [v for v in config["layouts"]["onboard"].values() if v["type"] == "Terminal"]
        self.assertEqual(len(terminals), 4)
        roles = set()
        for terminal in terminals:
            command = shlex.split(terminal["command"])
            index = command.index("--pane")
            roles.add(tuple(command[index+1:index+3]))
            self.assertIn("--preview", command)
            self.assertNotIn("--actuate", command)
        self.assertEqual(roles, {(s,r) for s in ("a","b") for r in ("bridge","policy")})

    def test_reject_actuation_without_confirmation_before_any_ssh(self):
        with patch.object(self.module.sys, "argv", ["launcher", "--actuate"]), \
                patch.object(self.module.subprocess, "run") as run, \
                self.assertRaises(SystemExit) as result:
            self.module.main()
        self.assertEqual(result.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
