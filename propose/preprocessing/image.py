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
