import numpy as np

from propose.cameras import Camera
from propose.poses import BasePose


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
    This is achieved by deconstructing the general rotation matrix to the the yaw matrix (a.k.a RotZ matrix).
    The yaw matrix defined as
    [ cos(a) -sin(a) 0 ]
    [ sin(a)  cos(a) 0 ]
    [   0      0     1 ]
    where "a" is the rotation angle in the XY plane.
    "a" can be computed from the general rotation matrix (R) given it's defintion
    [cos(a)cos(b) ... ...]
    [sin(a)cos(b) ... ...]
    [     ...     ... ...]
    (Some of the terms have been omitted for brevity.)
    Thus a = arctan( R_10 / R_00 ).
    :param pose: Pose to be rotated
    :param camera: Camera to rotate to.
    :return: rotated pose.
    """
    rot = camera.rotation_matrix.copy()

    tan = rot[1, 0] / rot[0, 0]
    cos = 1 / np.sqrt(1 + tan**2)  # Cos expressed in terms of tan
    sin = cos * tan

    yaw = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    rot_pose_matrix = pose.pose_matrix.dot(yaw)

    return pose.__class__(rot_pose_matrix)


def center_pose(pose: BasePose, center_marker_name: str = "SpineM") -> BasePose:
    """
    Center the pose such that the selected marker is always in [0, 0, 0]
    :param center_marker_name: Marker to which the pose should be centered to.
    :param pose: BasePose instance
    :return: BasePose instance with pose centered around SpineF
    """
    if len(pose.shape) == 3:
        return pose.copy() - pose[center_marker_name][:, np.newaxis]

    return pose.copy() - pose[center_marker_name]


def scale_pose(pose: BasePose, scale: float) -> BasePose:
    """
    Scales the pose by a scale parameter
    :param pose: Pose to be scaled
    :param scale: float determining the scaling given as (p * s), where p is pose and s is the scaling parameter.
    :return: scaled pose.
    """
    return pose.__class__(pose.pose_matrix * scale)
