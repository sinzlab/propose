from tests.mock.cameras import create_mock_camera

from propose.poses import Rat7mPose
import propose.preprocessing.rat7m as pp

import numpy as np


def test_mask_nans():
    pose_matrix = np.random.random((10, 20, 3))

    pose_matrix[:3] = np.nan

    pose = Rat7mPose(pose_matrix)

    mask = pp.mask_nans(pose)

    np.testing.assert_array_equal(mask, np.array([True, True, True, False, False, False, False, False, False, False]))


def test_mask_marker_failure():
    np.random.seed(1)
    pose_matrix = np.random.normal(size=(10, 20, 3), loc=0, scale=1e3)

    markers = ['ElbowL', 'ShoulderL', 'ArmL', 'ElbowR', 'ShoulderR', 'ArmR']
    marker_idx = [Rat7mPose.marker_names.index(marker) for marker in markers]

    pose_matrix[5:7, marker_idx[0]] = np.zeros(3)
    pose_matrix[5:7, marker_idx[1]] = np.zeros(3)

    pose = Rat7mPose(pose_matrix)

    mask = pp.mask_marker_failure(pose)

    target_mask = np.zeros(10)
    target_mask[5:7] = 1
    target_mask = target_mask.astype(bool)

    np.testing.assert_array_equal(mask, target_mask)


def test_apply_mask_with_camera():
    cameras = dict(Camera1=create_mock_camera())

    pose_matrix = np.random.random((10, 20, 3))

    pose_matrix[:3] = np.nan

    pose = Rat7mPose(pose_matrix)

    mask = pp.mask_nans(pose)

    masked_pose, masked_cameras = pp.apply_mask(mask, pose, cameras)

    np.testing.assert_array_equal(masked_pose.pose_matrix, pose_matrix[3:])
    np.testing.assert_array_equal(cameras['Camera1'].frames, np.arange(0, 10))
    np.testing.assert_array_equal(masked_cameras['Camera1'].frames, np.arange(3, 10))


def test_apply_mask_without_camera():
    pose_matrix = np.random.random((10, 20, 3))

    pose_matrix[:3] = np.nan

    pose = Rat7mPose(pose_matrix)

    mask = pp.mask_nans(pose)

    masked_pose = pp.apply_mask(mask, pose)

    np.testing.assert_array_equal(masked_pose.pose_matrix, pose_matrix[3:])
    np.testing.assert_array_equal(pose.pose_matrix, pose_matrix)
