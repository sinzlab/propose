import numpy as np
import numpy.typing as npt

from propose.poses import Rat7mPose

Mask = npt.NDArray[bool]  # bool array, where True means that frame is to be masked


def mask_marker_failure(pose: Rat7mPose, log_distance_threshold: float = 2) -> Mask:
    """
    Generates a mask for frames where joints fail to be tracked and switch to another marker
    :param pose: Rat7mPose instance
    :param log_distance_threshold: (optional) a parameter for the mask to determine where the
    :return: A boolean mask where True means that frame is to be masked
    """
    markers = ["ElbowL", "ShoulderL", "ArmL", "ElbowR", "ShoulderR", "ArmR"]
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
