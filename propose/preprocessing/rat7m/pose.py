from propose.poses import Rat7mPose

import numpy as np


def switch_arms_elbows(pose: Rat7mPose) -> Rat7mPose:
    """
    Detects were arms and elbows are switched and switches them back.
    Arms and elbows are considered as switched if the arm is above the elbow.
    :param pose: Rat7mPose instance
    :return: Rat7mPose instance with fixed arms and elbows
    """
    frames = np.arange(pose.shape[0])

    pose_matrix = pose.pose_matrix.copy()

    left_marker_idx = [Rat7mPose.marker_names.index(marker) for marker in ['ElbowL', 'ArmL']]
    right_marker_idx = [Rat7mPose.marker_names.index(marker) for marker in ['ElbowR', 'ArmR']]

    left_arm = pose_matrix[:, left_marker_idx]
    right_arm = pose_matrix[:, right_marker_idx]

    left_switch = np.int0(left_arm[:, 1, 2] > left_arm[:, 0, 2])
    right_switch = np.int0(right_arm[:, 1, 2] > right_arm[:, 0, 2])

    pose_matrix[:, left_marker_idx[0]] = left_arm[frames, left_switch]
    pose_matrix[:, left_marker_idx[1]] = left_arm[frames, 1 - left_switch]

    pose_matrix[:, right_marker_idx[0]] = right_arm[frames, right_switch]
    pose_matrix[:, right_marker_idx[1]] = right_arm[frames, 1 - right_switch]

    return Rat7mPose(pose_matrix)


def normalize_scale(pose: Rat7mPose) -> Rat7mPose:
    """
    Normalises the scale of the graph by dividing the graph by the length of the edge between SpineM and SpineF.
    As a result, all the edges in the graph have a length relative to the length of the edge between SpineM and SpineF.
    :param pose: Rat7mPose instance
    :return: Rat7mPose instance with normalised scale
    """
    reference_edge = pose.SpineF - pose.SpineM
    reference_edge_length = np.linalg.norm(reference_edge.pose_matrix, axis=1).mean()

    norm_pose_matrix = pose.pose_matrix / reference_edge_length

    return Rat7mPose(norm_pose_matrix)


def normalize_rotation(pose: Rat7mPose) -> Rat7mPose:
    """
    Rotate the pose such that SpineF and SpineM always lay on the Y-axis.
    :param pose: Rat7mPose instance
    :return: Rat7mPose instance with normalised rotation
    """
    reference_edge = pose.SpineF - pose.SpineM

    norm = np.linalg.norm(reference_edge.pose_matrix[:, :2], axis=1)
    n_frames = pose.shape[0]

    uy = reference_edge[:, 1] / norm
    ux = reference_edge[:, 0] / norm

    zeros = np.zeros(n_frames)
    ones = np.ones(n_frames)

    rotation_matrix = np.array([
        [uy, ux, zeros],
        [-ux, uy, zeros],
        [zeros, zeros, ones],
    ])
    rotation_matrix = np.moveaxis(rotation_matrix, -1, 0)

    rotated_pose_matrix = pose.pose_matrix @ rotation_matrix

    return Rat7mPose(rotated_pose_matrix)


def center_pose(pose: Rat7mPose) -> Rat7mPose:
    """
    Center the pose such that SpineM is always in [0, 0, 0]
    :param pose: Rat7mPose instance
    :return: Rat7mPose instance with pose centered around SpineF
    """
    return pose.copy() - pose.SpineM[:, np.newaxis]
