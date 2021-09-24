import propose.preprocessing.image as pp

from unittest.mock import MagicMock, patch

from propose.poses import Rat7mPose

import numpy as np


@patch('propose.preprocessing.image.rescale')
def test_rescale_image(skimage_rescale):
    np.random.seed(1)
    img = np.random.uniform(0, 1, size=(100, 100, 3))

    pp.rescale_image(img, 0.1)

    skimage_rescale.assert_called_once_with(img, scale=0.1, anti_aliasing=True, multichannel=True)


def test_square_crop_to_pose():
    np.random.seed(1)
    pose_matrix = np.random.uniform(0, 100, size=(20, 2))
    pose2D = Rat7mPose(pose_matrix)

    mean = pose_matrix.mean(0).astype(int)

    image = MagicMock()

    pp.square_crop_to_pose(image, pose2D, width=10)

    assert image.mock_calls[0][1][0][0] == slice(mean[1] - 5, mean[1] + 5, None)
    assert image.mock_calls[0][1][0][1] == slice(mean[0] - 5, mean[0] + 5, None)