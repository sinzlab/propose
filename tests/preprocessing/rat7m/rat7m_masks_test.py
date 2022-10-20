import numpy as np

import propose.preprocessing.rat7m as pp
from propose.poses import Rat7mPose


def test_mask_marker_failure():
    np.random.seed(1)
    pose_matrix = np.random.normal(size=(10, 20, 3), loc=0, scale=1e3)

    markers = ["ElbowL", "ShoulderL", "ArmL", "ElbowR", "ShoulderR", "ArmR"]
    marker_idx = [Rat7mPose.marker_names.index(marker) for marker in markers]

    pose_matrix[5:7, marker_idx[0]] = np.zeros(3)
    pose_matrix[5:7, marker_idx[1]] = np.zeros(3)

    pose = Rat7mPose(pose_matrix)

    mask = pp.mask_marker_failure(pose)

    target_mask = np.zeros(10)
    target_mask[5:7] = 1
    target_mask = target_mask.astype(bool)

    np.testing.assert_array_equal(mask, target_mask)
