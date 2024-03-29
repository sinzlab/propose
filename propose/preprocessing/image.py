import numpy.typing as npt

from propose.poses import BasePose

Image = npt.NDArray[float]


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


def scale_pixel_range(image: Image) -> Image:
    """
    Scales image from pixel range [0, 255] range to pixel range [0, 1].
    :param image: Image to be scaled with pixels in range [0, 255]
    :return: Scaled image with pixel in range [0, 1]
    """
    return image / 255
