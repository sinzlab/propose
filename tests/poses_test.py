from propose.poses import Rat7mPose, BasePose
import numpy as np
from .mock import create_mock_camera

from unittest.mock import Mock


def test_pose_init():
    shape = (10, 20, 3)
    pose_matrix = np.zeros(shape)

    pose = BasePose(pose_matrix)

    np.testing.assert_array_equal(pose_matrix, pose.pose_matrix)

    assert pose.shape == shape


def create_pose():
    pose_matrix = np.zeros((10, 20, 3))

    pose = Rat7mPose(
        pose_matrix
    )

    return pose


def test_rat7m_init():
    pose = create_pose()

    assert isinstance(pose, BasePose)

    assert list(pose.edge_groups.keys()) == ['head', 'spine', 'leg_l', 'leg_r', 'arm_l', 'arm_r']

    assert len(pose.edge_groups_flat) == 20


def test_rat7m_plot():
    pose = create_pose()

    ax = Mock()
    ax.plot = Mock(return_value=[1])

    actors = pose.plot(ax)
    assert len(actors) == 20


def test_pose_can_be_proj2D():
    pose = create_pose()
    camera = create_mock_camera()

    pose2D = camera.proj2D(pose)

    np.testing.assert_array_equal(camera.proj2D(pose.pose_matrix), pose2D)



