import numpy as np


class BasePose(object):
    def __init__(self, **kwargs):
        self.markers_dict = kwargs
        self.marker_names = list(kwargs.keys())
        self.marker_positions = np.array(list(kwargs.values()))


class PoseSet(object):
    def __init__(self, poses):
        self.poses = poses
        self.pose_class = poses[0].__class__

        self.pose_positions = np.array([pose.marker_positions for pose in poses])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, item):
        return self.poses[item]

    @property
    def shape(self):
        return self.pose_positions.shape
