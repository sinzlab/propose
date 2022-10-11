from unittest import TestCase

import numpy as np
from propose.poses import Human36mPose


class Human36mPoseTest(TestCase):
    @staticmethod
    def test_smoke():
        shape = (10, 17, 3)
        pose_matrix = np.zeros(shape)

        Human36mPose(pose_matrix)

    @staticmethod
    def test_marker_names():
        shape = (10, 17, 3)
        pose_matrix = np.zeros(shape)

        pose = Human36mPose(pose_matrix)

        assert pose.marker_names == [
            "Hip",
            "RHip",
            "RKnee",
            "RFoot",
            "LHip",
            "LKnee",
            "LFoot",
            "Spine",
            "Thorax",
            "Neck",
            "Head",
            "LShoulder",
            "LElbow",
            "LWrist",
            "RShoulder",
            "RElbow",
            "RWrist",
        ]
