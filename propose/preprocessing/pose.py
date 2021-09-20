from propose.poses import BasePose
from propose.cameras import Camera


def normalize_std(pose: BasePose) -> BasePose:
    """
    Normalizes the std of the poses in the dataset to be 1
    :param pose: set of Poses for which the normalization should occur.
    :return: pose: std normalized poses.
    """
    pose_matrix = pose.pose_matrix

    std = pose_matrix.std()

    pose_matrix /= std

    return pose.__class__(pose_matrix)


def rotate_to_camera(pose: BasePose, camera: Camera):
    """
    Rotates the pose to align with the camera. i.e. if the object faces the camera it is parallel to the X axis.
    :param pose: Pose to be rotated
    :param camera: Camera to rotate to.
    :return: rotated pose.
    """
    rot = camera.rotation_matrix.copy()

    # Clean rotation matrix to be a RotZ matrix
    rot[:, -1] = 0
    rot[-1, :] = 0
    rot[-1, -1] = 1

    rot_pose_matrix = pose.pose_matrix.dot(rot)

    return pose.__class__(rot_pose_matrix)
