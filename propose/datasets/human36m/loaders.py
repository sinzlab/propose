import cdflib
import numpy as np


def load_poses(path, every_n_frame: int = 4):
    """
    Loads the poses from the .cdf file.
    :param path: Path to the .cdf file.
    :param every_n_frame: How many frames to skip.
    :return: A numpy array with the poses. (N, 17, 3)
    """
    file = cdflib.CDF(path)

    poses_3d = file[0].squeeze()
    assert (
        poses_3d.shape[1] == 96
    ), f"Wrong number of joints, expected 96, got {poses_3d.shape[1]}"

    joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    poses_3d = poses_3d.reshape(-1, 32, 3)[:, joints]
    poses_3d = poses_3d.swapaxes(1, 2).reshape(-1, 17, 3)

    indices = np.arange(3, len(poses_3d), every_n_frame)

    poses_3d = poses_3d[indices, :]

    return poses_3d
