import os

from propose.datasets.rat7m.Rat7mDataset import Rat7mDataset
from neuralpredictors.data.datasets import TransformDataset

from unittest.mock import MagicMock, patch, call

import numpy as np


@patch('propose.datasets.rat7m.Rat7mDataset.Rat7mPose')
@patch('propose.datasets.rat7m.Rat7mDataset.pickle')
@patch('builtins.open')
def test_is_a_transform_dataset(open_mock, pickle_mock, Rat7mPose_mock):
    dataset = Rat7mDataset(
        'images',
        'poses',
        'cameras',
        dirname='',
        data_key='',
        transforms=[]
    )

    assert isinstance(dataset, TransformDataset)


@patch('propose.datasets.rat7m.Rat7mDataset.Rat7mPose')
@patch('propose.datasets.rat7m.Rat7mDataset.pickle')
@patch('builtins.open')
def test_pose_and_cameras_loaded(open_mock, pickle_mock, Rat7mPose_mock):
    Rat7mPose_mock.load = MagicMock()

    dirname = ''
    data_key = ''

    dataset = Rat7mDataset(
        'images',
        'poses',
        'cameras',
        dirname=dirname,
        data_key=data_key,
        transforms=[]
    )

    assert dataset.poses == Rat7mPose_mock.load()
    assert dataset.cameras == pickle_mock.load()
    assert Rat7mPose_mock.load.mock_calls[0] == call(f'{dirname}/poses/{data_key}.npy')
    assert open_mock.mock_calls[0] == call(f'{dirname}/cameras/{data_key}.pickle')


@patch('propose.datasets.rat7m.Rat7mDataset.Rat7mPose')
@patch('propose.datasets.rat7m.Rat7mDataset.pickle')
@patch('builtins.open')
def test_len(open_mock, pickle_mock, Rat7mPose_mock):
    poses = np.zeros((10, 20, 3))
    cameras = {
        'Camera1': MagicMock(),
        'Camera2': MagicMock(),
        'Camera3': MagicMock(),
    }

    Rat7mPose_mock.load = MagicMock(return_value=poses)
    pickle_mock.load = MagicMock(return_value=cameras)

    dirname = ''
    data_key = ''

    dataset = Rat7mDataset(
        'images',
        'poses',
        'cameras',
        dirname=dirname,
        data_key=data_key,
        transforms=[]
    )

    assert len(dataset) == len(poses) * len(cameras)


@patch('propose.datasets.rat7m.Rat7mDataset.Rat7mPose')
@patch('propose.datasets.rat7m.Rat7mDataset.pickle')
@patch('builtins.open')
@patch('propose.datasets.rat7m.Rat7mDataset.imageio')
def test_getitem(imageio_mock, open_mock, pickle_mock, Rat7mPose_mock):
    poses = np.zeros((7000, 20, 3))
    cameras = {
        'Camera1': MagicMock(),
        'Camera2': MagicMock(),
        'Camera3': MagicMock(),
    }

    cameras['Camera1'].frames.__getitem__ = MagicMock(side_effect=lambda x: x)
    cameras['Camera2'].frames.__getitem__ = MagicMock(side_effect=lambda x: x)

    Rat7mPose_mock.load = MagicMock(return_value=poses)
    pickle_mock.load = MagicMock(return_value=cameras)

    imageio_mock.imread = MagicMock(return_value=np.zeros((100, 100, 3)))

    dirname = ''
    data_key = ''

    dataset = Rat7mDataset(
        'images',
        'poses',
        'cameras',
        dirname=dirname,
        data_key=data_key,
        transforms=[]
    )

    # Test Cases

    # Selecting object for first camera
    camera, pose, image = dataset[0]

    assert camera == cameras['Camera1']
    assert imageio_mock.imread.mock_calls[0] == call(f'{dirname}/images/{data_key}/{data_key}-camera1-0.pickle')
    np.testing.assert_array_equal(pose, poses[0])

    # Selecting object for second camera
    camera, pose, image = dataset[7000 * 1]

    assert camera == cameras['Camera2']
    assert imageio_mock.imread.mock_calls[1] == call(f'{dirname}/images/{data_key}/{data_key}-camera2-0.pickle')
    np.testing.assert_array_equal(pose, poses[0])

    # Selecting object second pose for second camera
    camera, pose, image = dataset[7000 * 1 + 1]

    assert camera == cameras['Camera2']
    assert imageio_mock.imread.mock_calls[2] == call(f'{dirname}/images/{data_key}/{data_key}-camera2-0.pickle')
    np.testing.assert_array_equal(pose, poses[1])

    # Selecting object from second chunk for first camera
    camera, pose, image = dataset[3500]

    assert camera == cameras['Camera1']
    assert imageio_mock.imread.mock_calls[3] == call(f'{dirname}/images/{data_key}/{data_key}-camera1-3500.pickle')
    np.testing.assert_array_equal(pose, poses[0])
