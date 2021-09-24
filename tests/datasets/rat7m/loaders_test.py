from propose.cameras import Camera
from propose.poses import Rat7mPose
from propose.datasets.rat7m.loaders import load_mocap, load_cameras

path = './tests/mock/data/mocap-mock.mat'


def test_rat7m_mocap_loaded():
    mocap = load_mocap(path)

    assert isinstance(mocap, Rat7mPose)

    assert mocap.shape[1] == 20
    assert mocap.shape[2] == 3


def test_rat7m_camera_loaded():
    cameras = load_cameras(path)

    assert list(cameras.keys()) == ['Camera1', 'Camera2', 'Camera4', 'Camera5', 'Camera3', 'Camera6']
    assert isinstance(cameras['Camera1'], Camera)
