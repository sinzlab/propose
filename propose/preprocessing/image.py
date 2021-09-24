from propose.poses import BasePose

import numpy.typing as npt

from skimage.transform import rescale

Image = npt.NDArray[float]


def rescale_image(image: Image,
                  scale: float,
                  anti_aliasing: bool = True,
                  multichannel: bool = True,
                  **kwargs
                  ) -> Image:
    """
    Proxy for the skimage.transform.rescale function
    :param image: Image to be rescaled
    :param scale: float, by how much should the image be rescaled e.g. 0.5 means the image will be 2x smaller.
    :param anti_aliasing: (optional) Whether anti aliasing should be applied (default = True).
    :param multichannel: (optional) Whether the input should be considered as multi-channel (default = True).
    :param kwargs: additional parameters (see https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rescale)
    :return: rescaled Image.
    """
    return rescale(image, scale=scale, anti_aliasing=anti_aliasing, multichannel=multichannel, **kwargs)


def square_crop_to_pose(image: Image, pose2D: BasePose, width: int = 350) -> Image:
    """
    Crops a square from the input image such that the mean of the corresponding pose2D is in the center of the image.
    :param image: Input image to be cropped.
    :param pose2D: pose2D to find the center for cropping.
    :param width: width of the cropping (default = 350)
    :return: cropped image
    """
    mean_xy = pose2D.pose_matrix.mean(0).astype(int)

    padding = int(width // 2)

    x_slice = slice(mean_xy[0] - padding, mean_xy[0] + padding)
    y_slice = slice(mean_xy[1] - padding, mean_xy[1] + padding)

    return image[y_slice, x_slice]
