from propose.poses import Rat7mPose
import propose.preprocessing as pp

import numpy as np


def test_normalize_std():
    np.random.seed(1)
    pose_matrix = np.random.random(size=(10, 20, 3))

    pose = Rat7mPose(pose_matrix)

    std = pose_matrix.std()
    norm_pose_matrix = pose_matrix / std

    norm_pose = pp.normalize_std(pose)

    np.testing.assert_array_equal(norm_pose.pose_matrix, norm_pose_matrix)
