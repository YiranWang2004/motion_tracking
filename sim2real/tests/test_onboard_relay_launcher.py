"""Privilege handoff and process cleanup without touching real namespaces."""
import importlib.util
from pathlib import Path
import signal
import subprocess
from unittest.mock import Mock

import pytest


@pytest.fixture
def launcher(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "relay_launcher", root / "scripts/start_onboard_wired_relays.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "argv", [str(spec.origin), "--config",
                        str(root / "config/g1/onboard_scalebfm_wired.yaml")])
    return module


def test_authentication_precedes_detached_children(launcher, monkeypatch):
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1000)
    execute = Mock(side_effect=SystemExit(0))
    spawn = Mock()
    monkeypatch.setattr(launcher.os, "execvp", execute)
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    with pytest.raises(SystemExit):
        launcher.main()
    executable, command = execute.call_args.args
    assert executable == "sudo"
    assert command[:3] == ["sudo", "--", launcher.sys.executable]
    assert command[-2] == "--config"
    assert Path(command[-1]).is_absolute()
    spawn.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_privileged_supervision_and_cleanup(launcher, monkeypatch, failure):
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1000")
    monkeypatch.setenv("SUDO_GID", "1001")
    run = Mock()
    query = Mock(return_value="")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher.subprocess, "check_output", query)
    children = [Mock(pid=100+i) for i in range(3)]
    for child in children:
        child.poll.return_value = None
    if failure:
        children[1].poll.return_value = 1
    else:
        children[0].wait.side_effect = [subprocess.TimeoutExpired("relay", 5), 0]
    spawn = Mock(side_effect=children)
    kill = Mock()
    handlers = {}
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher.os, "killpg", kill)
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, fn: handlers.update({sig: fn}))
    monkeypatch.setattr(launcher.time, "sleep", lambda _: handlers[signal.SIGINT]())
    if failure:
        with pytest.raises(RuntimeError, match="onboard b relay exited \\(code 1\\)"):
            launcher.main()
    else:
        launcher.main()
    assert run.call_args.args[0][0] == "bash"
    assert "DUAL_NETWORK_CONFIG" in run.call_args.kwargs["env"]
    assert [c.args[0] for c in query.call_args_list] == [
        ["ip", "netns", "pids", "g1a"], ["ip", "netns", "pids", "g1b"]]
    assert spawn.call_args_list[0].args[0][:4] == ["ip", "netns", "exec", "g1a"]
    assert spawn.call_args_list[1].args[0][:4] == ["ip", "netns", "exec", "g1b"]
    assert spawn.call_args_list[2].kwargs == dict(
        start_new_session=True, user=1000, group=1001, extra_groups=[])
    assert all("sudo" not in call.args[0] for call in spawn.call_args_list)
    killed = [call.args for call in kill.call_args_list]
    assert (100, signal.SIGTERM) in killed
    assert (102, signal.SIGTERM) in killed
    if not failure:
        assert (100, signal.SIGKILL) in killed
    for child in children:
        child.wait.assert_called()
