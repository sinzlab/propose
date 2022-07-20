from propose.poses.base import YamlPose
import os


class Human36mPose(YamlPose):
    """
    Pose Class for the Human3.6M dataset.
    """

    def __init__(self, pose_matrix, **kwargs):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "metadata/human36m.yaml")

        super().__init__(pose_matrix, path)


data = """
spine:
    Hip:
      id: 0
      data_id: 0
      parent_id: -1

    Spine:
      id: 7
      data_id: 12
      parent_id: 0

    Thorax:
      id: 8
      data_id: 13
      parent_id: 7

    Neck:
      id: 9
      data_id: 14
      parent_id: 8

head:
    Head:
      id: 10
      data_id: 15
      parent_id: 9

leg_r:
    RHip:
      id: 1
      data_id: 1
      parent_id: 0

    RKnee:
      id: 2
      data_id: 2
      parent_id: 1

    RFoot:
      id: 3
      data_id: 3
      parent_id: 2

leg_l:
    LHip:
      id: 4
      data_id: 6
      parent_id: 0

    LKnee:
      id: 5
      data_id: 7
      parent_id: 4

    LFoot:
      id: 6
      data_id: 8
      parent_id: 5


arm_l:
    LShoulder:
      id: 11
      data_id: 17
      parent_id: 8

    LElbow:
      id: 12
      data_id: 18
      parent_id: 11

    LWrist:
      id: 13
      data_id: 19
      parent_id: 12


arm_r:
    RShoulder:
      id: 14
      data_id: 25
      parent_id: 8

    RElbow:
      id: 15
      data_id: 26
      parent_id: 14

    RWrist:
      id: 16
      data_id: 27
      parent_id: 15
"""
