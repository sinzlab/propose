import os
from unittest import TestCase

from propose.poses.utils import load_data_ids, yaml_pose_loader

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../mock/data/mock_pose.yaml")


class UtilsTest(TestCase):
    @staticmethod
    def test_yaml_pose_loader():
        marker_names, named_edges, named_group_edges = yaml_pose_loader(path)

        assert len(marker_names) == 6
        assert marker_names == ["RHip", "RFoot", "RKnee", "LHip", "LKnee", "LFoot"]

        assert len(named_edges) == 4
        assert set(named_edges) == {
            ("RHip", "RKnee"),
            ("RKnee", "RFoot"),
            ("LHip", "LKnee"),
            ("LKnee", "LFoot"),
        }

        assert len(named_group_edges.keys()) == 2
        assert set(named_group_edges.keys()) == {"leg_r", "leg_l"}

    @staticmethod
    def test_load_data_ids():
        data_ids = load_data_ids(path)

        assert len(data_ids) == 6

        assert data_ids == [0, 1, 2, 7, 4, 6]
