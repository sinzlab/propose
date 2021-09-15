import numpy as np


class BasePose(object):
    def __init__(self, **kwargs):
        self.marker_names = list(kwargs.keys())

        for marker_name in self.marker_names:
            setattr(self, marker_name, kwargs[marker_name])

    @property
    def marker_positions(self):
        positions = []
        for name in self.marker_names:
            positions.append(getattr(self, name))

        return np.array(positions)

    @property
    def shape(self):
        return self.marker_positions.shape

    def proj2D(self, camera):
        marker_dict = {}
        for name in self.marker_names:
            marker_pos = getattr(self, name)
            marker_dict[name] = camera.proj2D(marker_pos).squeeze()

        return self.__class__(**marker_dict)

    def plot(self, ax):
        raise NotImplementedError('The plot functionality has not been defined for this pose')


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
