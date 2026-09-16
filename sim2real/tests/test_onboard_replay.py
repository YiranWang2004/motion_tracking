import json

import numpy as np
import pytest

from dual_runtime.onboard_replay import convert, merge_arrays, select_pair


def recording(times, offset=0.0):
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    poses = np.zeros((n, 3, 7))
    poses[:, :, 6] = 1
    return dict(time_ns=(times * 1e9).astype(np.int64),
                wall_time_ns=((times + 1_700_000_000) * 1e9).astype(np.int64),
                peer_clock_offset_s=np.full(n, offset),
                peer_clock_uncertainty_s=np.full(n, .001),
                q=np.repeat(np.arange(n)[:, None], 29, axis=1),
                target=np.zeros((n, 29)), poses=poses,
                frame=np.arange(n), phase=np.full(n, 'executing'))


def test_large_clock_offset_and_nearest_samples():
    a = recording([1, 1.02, 1.04, 1.06])
    b = recording([568.002, 568.022, 568.042, 568.062], offset=-567)
    result = merge_arrays(a, b)
    # First A sample predates B coverage: do not extrapolate a robot pose.
    np.testing.assert_array_equal(result['source_index_a'], [1, 2, 3])
    np.testing.assert_array_equal(result['source_index_b'], [1, 2, 3])
    np.testing.assert_allclose(result['matched_b_skew_s'], .002, atol=1e-6)
    assert result['q'].shape == (3, 2, 29)


def test_no_overlap_and_clock_jump_rejected():
    with pytest.raises(ValueError, match='No overlapping'):
        merge_arrays(recording([1, 1.02]), recording([4, 4.02]))
    b = recording([1, 1.02, 1.04])
    b['peer_clock_offset_s'][1] = -1
    with pytest.raises(ValueError, match='Clock discontinuity'):
        merge_arrays(recording([1, 1.02, 1.04]), b)


def test_uncertain_matches_are_not_used():
    b = recording([1, 1.02])
    b['peer_clock_uncertainty_s'][:] = .04
    with pytest.raises(ValueError, match='No overlapping'):
        merge_arrays(recording([1, 1.02]), b)


def save_meta(root, side, name, run):
    directory = root / side / name
    directory.mkdir(parents=True)
    (directory / 'metadata.json').write_text(json.dumps(dict(
        robot=side, run_id=run, ticks=2, config=dict(artifact_directory='lift'))))
    (directory / 'rollout.npz').touch()
    return directory


def test_pair_by_run_id_not_b_directory_time(tmp_path):
    save_meta(tmp_path, 'a', '001', 'old')
    newest = save_meta(tmp_path, 'a', '002', 'new')
    save_meta(tmp_path, 'b', '999', 'old')
    partner = save_meta(tmp_path, 'b', '888', 'new')
    assert select_pair(tmp_path, 'lift') == (newest, partner)
    with pytest.raises(ValueError, match='No completed'):
        select_pair(tmp_path, 'lift', 'unknown')


def test_conversion_rejects_mismatched_identity_before_reading_npz(tmp_path):
    paths = [save_meta(tmp_path, side, '001', 'run') for side in 'ab']
    for side, path in zip('ab', paths):
        p = path / 'metadata.json'
        meta = json.loads(p.read_text())
        meta['identity'] = side
        p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='mismatch: identity'):
        convert(*paths, tmp_path / 'out', tmp_path / 'reference.npz')
