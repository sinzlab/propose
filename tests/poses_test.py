from propose.poses import Rat7mPose, BasePose, PoseSet
import numpy as np


def test_pose_init():
    a = np.zeros(3)
    b = np.zeros(3)
    c = np.zeros(3)
    d = np.zeros(3)

    markers = dict(a=a, b=b, c=c, d=d)
    pose = BasePose(**markers)

    np.testing.assert_array_equal(pose.marker_positions, np.zeros((4, 3)))

    assert pose.marker_names == ['a', 'b', 'c', 'd']
    assert pose.markers_dict == markers


def test_rat7m_init():
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

    assert isinstance(pose, BasePose)


def test_PoseSet_init():
    poses = []
    for i in range(10):
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

        poses.append(Rat7mPose(
            HeadF=HeadF,
            HeadB=HeadB,
            HeadL=HeadL,
            SpineF=SpineF,
            SpineM=SpineM,
            SpineL=SpineL,
            HipL=HipL,
            KneeL=KneeL,
            ShinL=ShinL,
            HipR=HipR,
            KneeR=KneeR,
            ShinR=ShinR,
            ElbowL=ElbowL,
            ArmL=ArmL,
            ShoulderL=ShoulderL,
            ElbowR=ElbowR,
            ArmR=ArmR,
            ShoulderR=ShoulderR,
            Offset1=Offset1,
            Offset2=Offset2
        ))

    pose_set = PoseSet(poses)

    assert len(pose_set) == 10

    assert pose_set.pose_positions.shape == (10, 20, 3)


def test_PoseSet_item_is_Pose():
    poses = []
    for i in range(10):
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

        poses.append(Rat7mPose(
            HeadF=HeadF,
            HeadB=HeadB,
            HeadL=HeadL,
            SpineF=SpineF,
            SpineM=SpineM,
            SpineL=SpineL,
            HipL=HipL,
            KneeL=KneeL,
            ShinL=ShinL,
            HipR=HipR,
            KneeR=KneeR,
            ShinR=ShinR,
            ElbowL=ElbowL,
            ArmL=ArmL,
            ShoulderL=ShoulderL,
            ElbowR=ElbowR,
            ArmR=ArmR,
            ShoulderR=ShoulderR,
            Offset1=Offset1,
            Offset2=Offset2
        ))

    pose_set = PoseSet(poses)

    assert isinstance(pose_set[0], BasePose)




