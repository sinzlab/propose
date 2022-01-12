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
