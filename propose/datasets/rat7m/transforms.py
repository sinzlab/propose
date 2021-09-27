import propose.preprocessing.rat7m as pp
from propose.poses.rat7m import Rat7mPose

import numpy.typing as npt
from collections import namedtuple

Image = npt.NDArray[float]


class ScalePose(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = x.poses
        pose_matrix = pose.pose_matrix

        pose_matrix *= self.scale

        key_vals['poses'] = pose.__class__(pose_matrix)

        return x.__class__(**key_vals)


class CenterPose(object):
    def __init__(self, center_marker_name='SpineM'):
        self.center_marker_name = center_marker_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals['poses'] = key_vals['poses'] - key_vals['poses'][self.center_marker_name]

        return x.__class__(**key_vals)


class CropImage(object):
    def __init__(self, width=350):
        self.width = width

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = key_vals['poses']
        image = key_vals['images']
        camera = key_vals['cameras']

        pose2D = Rat7mPose(camera.proj2D(pose))

        key_vals['images'] = pp.square_crop_to_pose(image=image, pose2D=pose2D, width=self.width)

        return x.__class__(**key_vals)


class RotateToCamera(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        pose = key_vals['poses']
        camera = key_vals['cameras']

        key_vals['poses'] = pp.rotate_to_camera(pose=pose, camera=camera)

        return x.__class__(**key_vals)


class ToGraph(object):
    def __init__(self):
        self.graph_data_point = namedtuple("GraphDataPoint", ('pose_matrix', 'adjacency_matrix', 'image'))

    def __call__(self, x):
        return self.graph_data_point(
            pose_matrix=x.poses.pose_matrix,
            adjacency_matrix=x.poses.adjacency_matrix,
            image=x.images)
