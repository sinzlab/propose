import os
from unittest import TestCase

import numpy as np
import numpy.testing as npt
from propose.poses import YamlPose

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../mock/data/mock_pose.yaml")


class YamlPoseTest(TestCase):
    @staticmethod
    def test_yaml_marker_names():
        shape = (10, 17, 3)
        pose_matrix = np.zeros(shape)

        pose = YamlPose(pose_matrix, path)

        assert len(pose.marker_names) == 6
        assert pose.marker_names == ["RHip", "RFoot", "RKnee", "LHip", "LKnee", "LFoot"]

    @staticmethod
    def test_yaml_adjacency_matrix():
        shape = (10, 17, 3)
        pose_matrix = np.zeros(shape)

        pose = YamlPose(pose_matrix, path)

        adjacency_matrix = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ]
        )

        npt.assert_array_equal(pose.adjacency_matrix, adjacency_matrix)

    @staticmethod
    def test_yaml_edge_groups():
        shape = (10, 6, 3)
        pose_matrix = np.zeros(shape)

        pose = YamlPose(pose_matrix, path)

        assert len(pose.edge_groups.keys()) == 2
        assert set(pose.edge_groups.keys()) == {"leg_r", "leg_l"}
