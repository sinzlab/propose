from unittest.mock import patch

import propose.preprocessing.image as pp

import numpy as np


@patch('propose.preprocessing.image.rescale')
def test_rescale_image(skimage_rescale):
    np.random.seed(1)
    img = np.random.uniform(0, 1, size=(100, 100, 3))

    pp.rescale_image(img, 0.1)

    skimage_rescale.assert_called_once_with(img, scale=0.1, anti_aliasing=True, multichannel=True)