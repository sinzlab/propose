from propose.cameras import Camera
from propose.poses import Rat7mPose

import numpy as np


def remove_nans(pose: Rat7mPose, cameras: dict[Camera]) -> tuple[Rat7mPose, dict[Camera]]:
    """
    Removes frames which have at least one joint missing from the mocap data.
    :param pose: Rat7mPose instance
    :param cameras: dictionary of Camera instances
    :return:
    """
    masked_pose = pose.copy()
    masked_cameras = cameras.copy()

    mask = np.invert(np.isnan(masked_pose).any(1).any(1))

    masked_pose = masked_pose[mask]

    for camera_key in masked_cameras:
        masked_cameras[camera_key].frames = masked_cameras[camera_key].frames[mask]

    return masked_pose, masked_cameras
