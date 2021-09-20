from propose.cameras import Camera
from propose.poses import Rat7mPose

import numpy as np
import numpy.typing as npt

from skimage.transform import rescale

Mask = npt.NDArray[bool]  # bool array, where True means that frame is to be masked
Image = npt.NDArray[float]


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


def square_crop_to_pose(image: Image, pose2D: Rat7mPose, width: int = 350) -> Image:
    """
    Crops a square from the input image such that the mean of the corresponding pose2D is in the center of the image.
    :param image: Input image to be cropped.
    :param pose2D: pose2D to find the center for cropping.
    :param width: width of the cropping (default = 350)
    :return: cropped image
    """
    mean_xy = pose2D.pose_matrix.mean(0).astype(int)

    padding = int(width // 2)

    x_slice = slice(mean_xy[0] - padding, mean_xy[0] + padding)
    y_slice = slice(mean_xy[1] - padding, mean_xy[1] + padding)

    return image[y_slice, x_slice]


def rescale_image(image: Image,
                  scale: float,
                  anti_aliasing: bool = True,
                  multichannel: bool = True,
                  **kwargs
                  ) -> Image:
    """
    Proxy for the skimage.transform.rescale function
    :param image: Image to be rescaled
    :param scale: float, by how much should the image be rescaled e.g. 0.5 means the image will be 2x smaller.
    :param anti_aliasing: (optional) Whether anti aliasing should be applied (default = True).
    :param multichannel: (optional) Whether the input should be considered as multi-channel (default = True).
    :param kwargs: additional parameters (see https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rescale)
    :return: rescaled Image.
    """
    return rescale(image, scale=scale, anti_aliasing=anti_aliasing, multichannel=multichannel, **kwargs)
