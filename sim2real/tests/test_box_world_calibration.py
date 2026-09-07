import unittest

import numpy as np

from omnicontact.perception.vive_pose import RigidTransform
from scripts.calibrate_vive_box_world import solve_box_tracker_world
from scripts.set_vive_box_axes import configure_box_axes


class TestBoxWorldCalibration(unittest.TestCase):
    def test_fixed_axes_preserve_world_and_translation_and_do_not_accumulate(self):
        config = {
            "world_from_steamvr": {"position_m": [1, 2, 3], "quaternion_xyzw": [0, 0, 0, 1]},
            "object_tracker_to_object": {"position_m": [.01, .02, .15], "quaternion_xyzw": [1, 0, 0, 0]},
        }
        world = config["world_from_steamvr"].copy()
        configure_box_axes(config, -1)
        first = config["object_tracker_to_object"].copy()
        configure_box_axes(config, -1)
        self.assertEqual(config["object_tracker_to_object"], first)
        self.assertEqual(config["world_from_steamvr"], world)
        self.assertEqual(first["position_m"], [.01, .02, .15])
        tracker = RigidTransform([0, 0, .3], [1, 0, 0, 0])
        box = tracker.compose(RigidTransform.from_dict(first, "box"))
        x_endpoint = box.compose(RigidTransform([1, 0, 0], [0, 0, 0, 1])).position
        np.testing.assert_allclose(x_endpoint - box.position, [0, -1, 0], atol=1e-12)
        configure_box_axes(config, 0)
        np.testing.assert_allclose(config["object_tracker_to_object"]["quaternion_xyzw"], [1, 0, 0, 0])

    def test_small_calibration_tilt_is_separate_from_fixed_axes(self):
        from scipy.spatial.transform import Rotation
        tilt = Rotation.from_euler("y", 2, degrees=True)
        measured = tilt * Rotation.from_euler("x", 180, degrees=True)
        config = {
            "object_tracker_serial": "box",
            "object_tracker_to_object": {"position_m": [0, 0, .15], "quaternion_xyzw": [1, 0, 0, 0]},
            "box_world_calibration": {"tracker_serial": "box", "world_from_tracker_start": {
                "position_m": [0, 0, .3], "quaternion_xyzw": measured.as_quat().tolist()}},
        }
        configure_box_axes(config, -1)
        mounting = Rotation.from_quat(config["object_tracker_to_object"]["quaternion_xyzw"])
        np.testing.assert_allclose((measured * mounting).as_matrix(),
                                   Rotation.from_euler("z", -90, degrees=True).as_matrix(), atol=1e-12)
        # Later real motion must remain observable, not be re-zeroed.
        motion = Rotation.from_euler("z", 15, degrees=True)
        np.testing.assert_allclose((motion * measured * mounting).as_matrix(),
                                   Rotation.from_euler("z", -75, degrees=True).as_matrix(), atol=1e-12)
        config["box_world_calibration"]["world_from_tracker_start"]["quaternion_xyzw"] = (
            Rotation.from_euler("z", 90, degrees=True) * measured).as_quat().tolist()
        with self.assertRaisesRegex(ValueError, "exceeds 10 deg"):
            configure_box_axes(config, -1)

    def test_box_center_and_top_tracker_pose(self):
        # Raw SteamVR pose is intentionally non-zero to exercise inversion.
        raw_steamvr_from_tracker = RigidTransform(
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        world_from_steamvr, tracker_from_object = solve_box_tracker_world(
            raw_steamvr_from_tracker,
            np.array([1.0, 2.0, 0.15]),
            0.30,
        )

        world_from_tracker = world_from_steamvr.compose(raw_steamvr_from_tracker)
        np.testing.assert_allclose(world_from_tracker.position, [1.0, 2.0, 0.30])
        np.testing.assert_allclose(tracker_from_object.position, [0.0, 0.0, 0.15])
        np.testing.assert_allclose(tracker_from_object.quaternion_xyzw, [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()


def test_recalibration_preserves_independent_box_axes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import calibrate_vive_box_world as calibration

    config = {
        "object_tracker_serial": "box",
        "object_tracker_to_object": {"position_m": [0, 0, .15], "quaternion_xyzw": [1, 0, 0, 0]},
    }
    configure_box_axes(config, -1)
    path = tmp_path / "vive.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(calibration, "OpenVRTrackerReader", lambda serials: SimpleNamespace(
        start=lambda: {"box": 0}, stop=lambda: None))
    monkeypatch.setattr(calibration, "capture_pose", lambda *args: SimpleNamespace(
        position_s=np.array([2., 3., 4.]), quaternion_s_tracker_xyzw=np.array([0., 0., 0., 1.]),
        valid_samples=150, orientation_std_deg=.01))
    calibration.main(["--vive-config", str(path), "--output", str(path)])
    result = json.loads(path.read_text())
    assert result["object_frame_definition"]["box_local_z_quarter_turns"] == -1
    assert result["box_world_calibration"]["tracker_serial"] == "box"
    np.testing.assert_allclose(result["object_tracker_to_object"]["quaternion_xyzw"],
                               config["object_tracker_to_object"]["quaternion_xyzw"], atol=1e-12)
