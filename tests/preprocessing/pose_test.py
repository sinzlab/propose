from propose.poses import Rat7mPose
import propose.preprocessing as pp

from tests.mock.cameras import create_mock_camera

import numpy as np

from scipy.spatial.transform import Rotation as R


def test_normalize_std():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))

    pose = Rat7mPose(pose_matrix)

    std = pose_matrix.std()
    norm_pose_matrix = pose_matrix / std

    norm_pose = pp.normalize_std(pose)

    np.testing.assert_array_equal(norm_pose.pose_matrix, norm_pose_matrix)


def test_rotate_to_camera():
    gen_rot = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    r = R.from_matrix(gen_rot)
    theta = np.arctan(gen_rot[0, 1] / gen_rot[0, 0])

    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    camera_1 = create_mock_camera(rot_matrix)
    camera_2 = create_mock_camera(gen_rot)

    pose_matrix = np.random.random(size=(10, 20, 3))
    pose = Rat7mPose(pose_matrix)

    rot_pose = pp.rotate_to_camera(pose, camera_1)
    np.testing.assert_array_almost_equal(rot_pose.pose_matrix, pose.pose_matrix @ rot_matrix)

    rot_pose = pp.rotate_to_camera(pose, camera_2)
    np.testing.assert_array_almost_equal(rot_pose.pose_matrix, pose.pose_matrix @ rot_matrix)


def test_center_pose_multi_frame():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))
    pose = Rat7mPose(pose_matrix)

    spine_m = pose.SpineM.pose_matrix.copy()
    arm_r = pose.ArmR.pose_matrix.copy()

    centered_pose = pp.center_pose(pose)

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, spine_m, centered_pose.SpineM.pose_matrix)
    np.testing.assert_array_equal(centered_pose.SpineM.pose_matrix, np.zeros_like(centered_pose.SpineM.pose_matrix))
    np.testing.assert_array_equal(centered_pose.ArmR.pose_matrix, arm_r - spine_m)


def test_center_pose_single_frame():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(20, 3))
    pose = Rat7mPose(pose_matrix)

    spine_m = pose.SpineM.pose_matrix.copy()
    arm_r = pose.ArmR.pose_matrix.copy()

    centered_pose = pp.center_pose(pose)

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, spine_m, centered_pose.SpineM.pose_matrix)
    np.testing.assert_array_equal(centered_pose.SpineM.pose_matrix, np.zeros_like(centered_pose.SpineM.pose_matrix))
    np.testing.assert_array_equal(centered_pose.ArmR.pose_matrix, arm_r - spine_m)


def test_scale_pose():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(20, 3))
    pose = Rat7mPose(pose_matrix)

    scale = 2
    scaled_pose = pp.scale_pose(pose, scale)

    np.testing.assert_array_equal(scaled_pose, pose_matrix * scale)
