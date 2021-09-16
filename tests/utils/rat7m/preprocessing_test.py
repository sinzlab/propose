from tests.mock.cameras import create_mock_camera

from propose.poses import Rat7mPose
from propose.utils.rat7m.preprocessing import remove_nans

import numpy as np


def test_remove_nans():
    cameras = dict(Camera1=create_mock_camera())

    pose_matrix = np.random.random((10, 20, 3))

    pose_matrix[:3] = np.nan

    pose = Rat7mPose(pose_matrix)

    masked_pose, masked_cameras = remove_nans(pose, cameras)

    np.testing.assert_array_equal(masked_pose.pose_matrix, pose_matrix[3:])
    np.testing.assert_array_equal(cameras['Camera1'].frames, np.arange(3, 10))
