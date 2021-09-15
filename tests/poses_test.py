from propose.poses import Rat7mPose, BasePose, PoseSet
import numpy as np
from .mock import create_mock_camera

from unittest.mock import Mock

def test_pose_init():
    a = np.zeros(3)
    b = np.zeros(3)
    c = np.zeros(3)
    d = np.zeros(3)

    markers = dict(a=a, b=b, c=c, d=d)
    pose = BasePose(**markers)

    np.testing.assert_array_equal(pose.marker_positions, np.zeros((4, 3)))

    assert pose.marker_names == ['a', 'b', 'c', 'd']


def create_pose():
    HeadF = np.zeros(3)
    HeadB = np.zeros(3)
    HeadL = np.zeros(3)
    SpineF = np.zeros(3)
    SpineM = np.zeros(3)
    SpineL = np.zeros(3)
    HipL = np.zeros(3)
    KneeL = np.zeros(3)
    ShinL = np.zeros(3)
    HipR = np.zeros(3)
    KneeR = np.zeros(3)
    ShinR = np.zeros(3)
    ElbowL = np.zeros(3)
    ArmL = np.zeros(3)
    ShoulderL = np.zeros(3)
    ElbowR = np.zeros(3)
    ArmR = np.zeros(3)
    ShoulderR = np.zeros(3)
    Offset1 = np.zeros(3)
    Offset2 = np.zeros(3)

    pose = Rat7mPose(
        HeadF,
        HeadB,
        HeadL,
        SpineF,
        SpineM,
        SpineL,
        HipL,
        KneeL,
        ShinL,
        HipR,
        KneeR,
        ShinR,
        ElbowL,
        ArmL,
        ShoulderL,
        ElbowR,
        ArmR,
        ShoulderR,
        Offset1,
        Offset2
    )

    return pose


def test_rat7m_init():
    pose = create_pose()

    assert isinstance(pose, BasePose)

    assert list(pose.edge_groups.keys()) == ['head', 'spine', 'leg_l', 'leg_r', 'arm_l', 'arm_r']


def test_rat7m_plot():
    pose = create_pose()

    ax = Mock()
    ax.plot = Mock(return_value=[1])

    actors = pose.plot(ax)
    assert len(actors) == 20


def test_pose_can_be_proj2D():
    pose = create_pose()
    camera = create_mock_camera()

    pose2D = pose.proj2D(camera)

    assert pose != pose2D
    np.testing.assert_array_equal(camera.proj2D(pose.marker_positions), pose2D.marker_positions)


def test_PoseSet_init():
    poses = [create_pose() for _ in range(10)]
    pose_set = PoseSet(poses)

    assert len(pose_set) == 10

    assert pose_set.pose_positions.shape == (10, 20, 3)

    assert isinstance(pose_set[0], BasePose)


def test_PoseSet_slice_selection():
    poses = [create_pose() for _ in range(10)]
    pose_set = PoseSet(poses)

    sliced_pose_set = pose_set[3:5]
    assert isinstance(sliced_pose_set, PoseSet)
    assert sliced_pose_set.shape == (2, 20, 3)


def test_PoseSet_can_be_proj2D():
    poses = [create_pose() for _ in range(10)]
    pose_set = PoseSet(poses)

    camera = create_mock_camera()
    pose2D_set = pose_set.proj2D(camera)

    assert pose_set != pose2D_set
    assert pose2D_set.shape == (10, 20, 2)


