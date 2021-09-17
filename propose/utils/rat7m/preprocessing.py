from propose.cameras import Camera
from propose.poses import Rat7mPose

import numpy as np
import numpy.typing as npt

Mask = npt.NDArray[bool]  # bool array, where True means that frame is to be masked


def mask_nans(pose: Rat7mPose) -> Mask:
    """
    Generates a mask for frames which have at least one joint missing from the mocap data.
    :param pose: Rat7mPose instance
    :return: A boolean mask where True means that frame is to be masked
    """
    mask = np.isnan(pose).any(1).any(1)

    return mask


def mask_marker_failure(pose: Rat7mPose, log_distance_threshold: float = 2) -> Mask:
    """
    Generates a mask for frames where joints fail to be tracked and switch to another marker
    :param pose: Rat7mPose instance
    :param log_distance_threshold: (optional) a parameter for the mask to determine where the
    :return: A boolean mask where True means that frame is to be masked
    """
    markers = ['ElbowL', 'ShoulderL', 'ArmL', 'ElbowR', 'ShoulderR', 'ArmR']
    marker_idx = [pose.marker_names.index(marker) for marker in markers]

    selected_matrix = pose.pose_matrix[:, marker_idx]

    distance_matrix = np.zeros((len(pose.pose_matrix), len(markers), len(markers)))

    for i in range(len(markers)):
        for j in range(len(markers)):
            diff = selected_matrix[:, i, :] - selected_matrix[:, j, :]
            dist = diff.__pow__(2).sum(1).__pow__(0.5)

            distance_matrix[:, i, j] = dist

    # Log Distance matrix with large diagonal to not consider the diagonals in masking
    # as the diagonal has the distance between the same joints i.e. = 0
    log_distance_matrix = np.log(distance_matrix + 1) + np.eye(len(markers)) * 1e9

    return (log_distance_matrix < log_distance_threshold).any(1).any(1)


def apply_mask(mask: Mask, pose: Rat7mPose, cameras: dict[Camera] = None) -> tuple[Rat7mPose, dict[Camera]]:
    """
    Removes frames which have at least one joint missing from the mocap data.
    :param mask: boolean mask that is to be applied to pose and cameras, where True means that frame is to be masked
    :param pose: Rat7mPose instance
    :param cameras: dictionary of Camera instances (optional)
    :return: masked_pose and masked_cameras (if cameras were provided)
    """
    masked_pose = pose.copy()[np.invert(mask)]

    if cameras:
        masked_cameras = {}
        for camera_key in cameras:
            masked_cameras[camera_key] = cameras[camera_key].copy()
            masked_cameras[camera_key].frames = masked_cameras[camera_key].frames[np.invert(mask)]

        return masked_pose, masked_cameras

    return masked_pose


def switch_arms_elbows(pose):
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
