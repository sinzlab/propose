import propose.preprocessing.rat7m as pp
from propose.poses.rat7m import Rat7mPose

from collections import namedtuple


class ScalePose(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals['poses'] = pp.scale_pose(pose=x.poses, scale=self.scale)

        return x.__class__(**key_vals)


class CenterPose(object):
    def __init__(self, center_marker_name='SpineM'):
        self.center_marker_name = center_marker_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}

        key_vals['poses'] = pp.center_pose(pose=key_vals['poses'], center_marker_name=self.center_marker_name)

        return x.__class__(**key_vals)


class CropImageToPose(object):
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


class RotatePoseToCamera(object):
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


class SwitchArmsElbows(object):
    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        pose = key_vals['poses']

        key_vals['poses'] = pp.switch_arms_elbows(pose=pose)

        return x.__class__(**key_vals)
