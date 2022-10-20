import numpy as np
import numpy.typing as npt

from propose.cameras import Camera
from propose.poses import BasePose

Mask = npt.NDArray[bool]  # bool array, where True means that frame is to be masked


def mask_nans(pose: BasePose) -> Mask:
    """
    Generates a mask for frames which have at least one joint missing from the mocap data.
    :param pose: Rat7mPose instance
    :return: A boolean mask where True means that frame is to be masked
    """
    mask = np.isnan(pose).any(1).any(1)

    return mask


def apply_mask(
    mask: Mask, pose: BasePose, cameras: dict[Camera] = None
) -> tuple[BasePose, dict[Camera]]:
    """
    Removes frames which have at least one joint missing from the mocap data.
    :param mask: boolean mask that is to be applied to pose and cameras, where True means that frame is to be masked
    :param pose: A BasePose instance
    :param cameras: dictionary of Camera instances (optional)
    :return: masked_pose and masked_cameras (if cameras were provided)
    """
    masked_pose = pose.copy()[np.invert(mask)]

    if cameras:
        masked_cameras = {}
        for camera_key in cameras:
            masked_cameras[camera_key] = cameras[camera_key].copy()
            masked_cameras[camera_key].frames = masked_cameras[camera_key].frames[
                np.invert(mask)
            ]

        return masked_pose, masked_cameras

    return masked_pose
