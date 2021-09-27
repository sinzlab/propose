from propose.poses import Rat7mPose
import propose.datasets.rat7m.transforms as tr
from tests.mock.cameras import create_mock_camera

from unittest.mock import MagicMock, patch

from collections import namedtuple

import numpy as np


@patch('propose.datasets.rat7m.transforms.pp')
def test_ScalePose(pp_mock):
    np.random.seed(1)
    data = namedtuple('Data', ['poses'])

    pose_matrix = np.random.random(size=(10, 20, 3))

    pose = Rat7mPose(pose_matrix)

    scale = 0.5
    tr.ScalePose(scale=scale)(data(poses=pose))
    pp_mock.scale_pose.assert_called_once_with(pose=pose, scale=scale)


@patch('propose.datasets.rat7m.transforms.pp')
def test_CenterPose(pp_mock):
    np.random.seed(1)
    data = namedtuple('Data', ['poses'])

    pose_matrix = np.random.random(size=(10, 20, 3))

    pose = Rat7mPose(pose_matrix)

    tr.CenterPose(center_marker_name='SpineM')(data(poses=pose[0]))

    pp_mock.center_pose.assert_called_once_with(pose=pose[0], center_marker_name='SpineM')


@patch('propose.datasets.rat7m.transforms.pp')
def test_CropImage(pp_mock):
    np.random.seed(1)
    camera = create_mock_camera()
    data = namedtuple('Data', ['poses', 'images', 'cameras'])

    pose_matrix = np.random.uniform(0, 100, size=(20, 3))
    pose = Rat7mPose(pose_matrix)

    pose2D = Rat7mPose(camera.proj2D(pose))

    image = MagicMock()

    width = 10
    tr.CropImageToPose(width=width)(data(poses=pose, images=image, cameras=camera))

    pp_mock.square_crop_to_pose.assert_called_once_with(image=image, pose2D=pose2D, width=width)


@patch('propose.datasets.rat7m.transforms.pp')
def test_RotateToCamera(pp_mock):
    np.random.seed(1)
    camera = create_mock_camera()
    data = namedtuple('Data', ['poses', 'cameras'])

    pose_matrix = np.random.uniform(0, 100, size=(20, 3))
    pose = Rat7mPose(pose_matrix)

    tr.RotatePoseToCamera()(data(poses=pose, cameras=camera))

    pp_mock.rotate_to_camera.assert_called_once_with(pose=pose, camera=camera)


def test_ToGraph():
    np.random.seed(1)
    data = namedtuple('Data', ['poses', 'images'])

    pose_matrix = np.random.uniform(0, 100, size=(20, 3))
    pose = Rat7mPose(pose_matrix)

    images = np.zeros((100, 100, 3))

    res = tr.ToGraph()(data(poses=pose, images=images))

    np.testing.assert_array_equal(res.pose_matrix, pose.pose_matrix)
    np.testing.assert_array_equal(res.adjacency_matrix, pose.adjacency_matrix)
    np.testing.assert_array_equal(res.image, images)


@patch('propose.datasets.rat7m.transforms.pp')
def test_SwitchArmsElbows(pp_mock):
    np.random.seed(1)
    camera = create_mock_camera()
    data = namedtuple('Data', ['poses', 'cameras'])

    pose_matrix = np.random.uniform(0, 100, size=(20, 3))
    pose = Rat7mPose(pose_matrix)

    res = tr.SwitchArmsElbows()(data(poses=pose, cameras=camera))

    pp_mock.switch_arms_elbows.assert_called_once_with(pose=pose)

    assert isinstance(res, data)


@patch('propose.datasets.rat7m.transforms.pp')
def test_ScalePixelRange(pp_mock):
    np.random.seed(1)
    data = namedtuple('Data', ['images'])

    image = MagicMock()

    res = tr.ScalePixelRange()(data(images=image))

    pp_mock.scale_pixel_range.assert_called_once_with(image=image)

    assert isinstance(res, data)
