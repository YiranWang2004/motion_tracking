import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dual_runtime import initial_scene
from test_dual_scalebfm_deploy import fresh_snapshot


@pytest.mark.parametrize('failure', [None, 'timeout', 'stale', 'start', 'missing'])
def test_capture_closes_reader_on_success_and_failure(monkeypatch, failure):
    snapshot = fresh_snapshot()
    if failure == 'stale':
        snapshot = replace(snapshot, robot_a=replace(snapshot.robot_a, stamp_s=time.monotonic()-1))
    calls = []
    def start():
        if failure == 'start':
            raise RuntimeError('reader failed')
    provider = SimpleNamespace(start=start, stop=lambda: calls.append('stop'),
        wait_until_ready=lambda timeout: failure != 'timeout', get_snapshot=lambda: None if failure == 'missing' else snapshot)
    monkeypatch.setattr(initial_scene.DualViveDeploymentConfig, 'load', lambda *a, **kw: None)
    def make(*args, **kwargs):
        assert kwargs['require_object'] is False
        return provider
    monkeypatch.setattr(initial_scene, 'DualVivePoseProvider', make)
    if failure:
        with pytest.raises(RuntimeError):
            initial_scene.capture_vive_initial_scene('fake.json', require_object=False)
    else:
        assert initial_scene.capture_vive_initial_scene('fake.json', require_object=False) is snapshot
    assert calls == ['stop']
