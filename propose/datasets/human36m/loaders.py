import os
import cdflib
import numpy as np

from propose.poses.utils import load_data_ids


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

    dirname = os.path.dirname(__file__)
    metadata_path = os.path.join(dirname, "../../poses/metadata/human36m.yaml")
    joints = load_data_ids(metadata_path)

    poses_3d = poses_3d.reshape(-1, 32, 3)[:, joints]
    poses_3d = poses_3d.swapaxes(1, 2).reshape(-1, 17, 3)

    indices = np.arange(3, len(poses_3d), every_n_frame)

    poses_3d = poses_3d[indices, :]

    return poses_3d
