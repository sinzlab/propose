import numpy as np

import propose.preprocessing.rat7m as pp
from propose.poses import Rat7mPose


def test_switch_arms_legs():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))

    markers = ["ElbowL", "ArmL", "ElbowR", "ArmR"]
    marker_idx = [Rat7mPose.marker_names.index(marker) for marker in markers]

    pose_matrix[:, marker_idx[0]] = 0
    pose_matrix[:, marker_idx[1]] = 0

    pose_matrix[:5, marker_idx[0]] = 1
    pose_matrix[5:, marker_idx[1]] = 1

    pose_matrix[:, marker_idx[2]] = 0
    pose_matrix[:, marker_idx[3]] = 0

    pose_matrix[:5, marker_idx[2]] = 1
    pose_matrix[5:, marker_idx[3]] = 1

    pose = Rat7mPose(pose_matrix)

    pose = pp.switch_arms_elbows(pose)

    np.testing.assert_array_equal(pose.pose_matrix[:, marker_idx[0], 2], np.ones(10))
    np.testing.assert_array_equal(pose.pose_matrix[:, marker_idx[1], 2], np.zeros(10))

    np.testing.assert_array_equal(pose.pose_matrix[:, marker_idx[2], 2], np.ones(10))
    np.testing.assert_array_equal(pose.pose_matrix[:, marker_idx[3], 2], np.zeros(10))


def test_switch_arms_legs_single_frame():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(20, 3))

    markers = ["ElbowL", "ArmL", "ElbowR", "ArmR"]
    marker_idx = [Rat7mPose.marker_names.index(marker) for marker in markers]

    pose_matrix[marker_idx[0]] = 0
    pose_matrix[marker_idx[1]] = 0

    pose_matrix[marker_idx[0]] = 0
    pose_matrix[marker_idx[1]] = 1

    pose_matrix[marker_idx[2]] = 0
    pose_matrix[marker_idx[3]] = 0

    pose_matrix[marker_idx[2]] = 0
    pose_matrix[marker_idx[3]] = 1

    pose = Rat7mPose(pose_matrix)

    pose = pp.switch_arms_elbows(pose)

    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[0], 2], np.ones(1))
    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[1], 2], np.zeros(0))

    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[2], 2], np.ones(1))
    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[3], 2], np.zeros(0))

    # Inverted Case
    pose_matrix[marker_idx[0]] = 0
    pose_matrix[marker_idx[1]] = 0

    pose_matrix[marker_idx[0]] = 1
    pose_matrix[marker_idx[1]] = 0

    pose_matrix[marker_idx[2]] = 0
    pose_matrix[marker_idx[3]] = 0

    pose_matrix[marker_idx[2]] = 1
    pose_matrix[marker_idx[3]] = 0

    pose = Rat7mPose(pose_matrix)

    pose = pp.switch_arms_elbows(pose)

    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[0], 2], np.ones(1))
    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[1], 2], np.zeros(0))

    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[2], 2], np.ones(1))
    np.testing.assert_array_equal(pose.pose_matrix[marker_idx[3], 2], np.zeros(0))


def test_normalize_scaling():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))
    pose = Rat7mPose(pose_matrix)

    reference_edge = pose.SpineF - pose.SpineM
    test_edge = pose.ArmL - pose.ElbowL

    test_ratio = (
        np.linalg.norm(test_edge, axis=1)
        / np.linalg.norm(reference_edge, axis=1).mean()
    )

    norm_pose = pp.normalize_scale(pose)

    norm_reference_edge = norm_pose.SpineF - norm_pose.SpineM
    norm_test_edge = norm_pose.ArmL - norm_pose.ElbowL

    norm_test_ratio = (
        np.linalg.norm(norm_test_edge, axis=1)
        / np.linalg.norm(norm_reference_edge, axis=1).mean()
    )

    np.testing.assert_array_almost_equal(
        np.linalg.norm(norm_reference_edge, axis=1),
        np.linalg.norm(reference_edge, axis=1)
        / np.linalg.norm(reference_edge, axis=1).mean(),
    )
    np.testing.assert_array_almost_equal(test_ratio, norm_test_ratio)


def test_normalize_rotation():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))

    pose_matrix[4, Rat7mPose.marker_names.index("SpineM")] = np.array([0, 0, 0])
    pose_matrix[4, Rat7mPose.marker_names.index("SpineF")] = np.array([1, 1, 1])

    pose = Rat7mPose(pose_matrix)

    norm_pose = pp.normalize_rotation(pose)

    target_position = np.array([0, np.sqrt(2), 1])

    np.testing.assert_array_almost_equal(norm_pose.SpineF[4], target_position)
