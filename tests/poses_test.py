from propose.poses import Rat7mPose, BasePose
import numpy as np
from .mock import create_mock_camera

from unittest.mock import Mock, patch, call


def test_pose_init():
    shape = (10, 20, 3)
    pose_matrix = np.zeros(shape)

    try:
        BasePose(pose_matrix)
    except NotImplementedError as e:
        assert str(e) == 'Adjacency matrix has not been setup'


def create_pose(pose_matrix=np.zeros((10, 20, 3))):
    pose = Rat7mPose(
        pose_matrix
    )

    return pose


def test_rat7m_init():
    shape = (10, 20, 3)
    pose_matrix = np.zeros(shape)
    pose = create_pose(pose_matrix)

    assert isinstance(pose, BasePose)

    assert list(pose.edge_groups.keys()) == ['head', 'spine', 'leg_l', 'leg_r', 'arm_l', 'arm_r']

    assert len(pose.edge_groups_flat) == 20

    np.testing.assert_array_equal(pose_matrix, pose.pose_matrix)

    assert pose.shape == shape


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


def test_get_by_marker_name():
    pose_matrix = np.zeros((10, 20, 3))
    pose = create_pose(pose_matrix)

    np.testing.assert_array_equal(pose.HeadF, pose[:, 0])
    np.testing.assert_array_equal(pose['HeadF'], pose[:, 0])

    np.testing.assert_array_equal(pose.HeadB, pose[:, 1])
    np.testing.assert_array_equal(pose['HeadB'], pose[:, 1])


def test_pose_comparison():
    pose_matrix = np.zeros((10, 20, 3))
    pose1 = create_pose(pose_matrix)

    pose_matrix = np.zeros((10, 20, 3))
    pose2 = create_pose(pose_matrix)

    pose_matrix = np.ones((10, 20, 3))
    pose3 = create_pose(pose_matrix)

    np.testing.assert_array_equal(pose1, pose1)
    np.testing.assert_array_equal(pose1, pose2)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, pose1, pose3)


@patch('propose.poses.base.np')
def test_save(numpy_mock):
    pose = create_pose()

    numpy_mock.load = Mock(return_value=pose.pose_matrix)
    numpy_mock.save = Mock()

    pose.save('path')
    loaded_pose = Rat7mPose.load('path')

    assert numpy_mock.save.mock_calls[0] == call('path', pose.pose_matrix)
    assert numpy_mock.load.mock_calls[0] == call('path')
    assert isinstance(loaded_pose, Rat7mPose)


def test_adjacency():
    pose = create_pose()

    np.testing.assert_array_equal(pose.adjacency_matrix, np.array(
        [[1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., ],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ]]))
