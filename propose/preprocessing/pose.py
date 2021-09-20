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
