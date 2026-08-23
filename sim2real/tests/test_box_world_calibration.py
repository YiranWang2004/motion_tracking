import unittest

import numpy as np

from omnicontact.perception.vive_pose import RigidTransform
from scripts.calibrate_vive_box_world import solve_box_tracker_world


class TestBoxWorldCalibration(unittest.TestCase):
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
