from propose.poses import Rat7mPose
import propose.preprocessing as pp

from tests.mock.cameras import create_mock_camera

import numpy as np


def test_normalize_std():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))

    pose = Rat7mPose(pose_matrix)

    std = pose_matrix.std()
    norm_pose_matrix = pose_matrix / std

    norm_pose = pp.normalize_std(pose)

    np.testing.assert_array_equal(norm_pose.pose_matrix, norm_pose_matrix)


def test_rotate_to_camera():
    rot_matrix = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])

    camera_1 = create_mock_camera(rot_matrix)
    camera_2 = create_mock_camera(np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]))

    pose_matrix = np.random.random(size=(10, 20, 3))
    pose = Rat7mPose(pose_matrix)

    rot_pose = pp.rotate_to_camera(pose, camera_1)
    np.testing.assert_array_equal(rot_pose.pose_matrix, pose.pose_matrix @ rot_matrix)

    rot_pose = pp.rotate_to_camera(pose, camera_2)
    np.testing.assert_array_equal(rot_pose.pose_matrix, pose.pose_matrix @ rot_matrix)
