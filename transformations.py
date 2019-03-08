"""
Image transformations.
"""
import cv2
import numpy as np

from skimage.color.adapt_rgb import adapt_rgb
from skimage.color.adapt_rgb import hsv_value, each_channel


def gaussian_blur(im=None, k=5, sigma=1):
    kwargs = {'ksize': k,
              'sigmaX': sigma}

    def blur(image):
        return cv2.GaussianBlur(image, **kwargs)

    if im is None:
        return blur
    else:
        return blur(im)


def normalise(im=None, max_val=1, log_scale=False, adaptation='rgb'):
    """

    Parameters
    ----------
    im
    max_val: int
    adaptation

    log_scale: bool

    Returns
    -------

    """
    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):
        if log_scale:
            image = np.log10(np.clip(image, 0.001, 10 ** 10))

        return max_val * (image - image.min())/(image.max() - image.min())

    if im is None:
        return f
    else:
        return f(im)


def gradient(im=None, blur=None, adaptation='rgb'):

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):

        if blur is not None:
            image = blur(image)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        return np.sqrt((sobelx ** 2) + (sobely ** 2))

    if im is None:
        return f
    else:
        return f(im)


def laplacian(im=None, blur=None, adaptation='rgb'):

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):

        if blur is not None:
            image = blur(image)

        limg = cv2.Laplacian(image, cv2.CV_64F)
        limg = (limg - limg.min()) / (limg.max() - limg.min())

        return limg

    if im is None:
        return f
    else:
        return f(im)


def median(im=None, disk_size=2, adaptation='rgb'):

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):
        return cv2.medianBlur(image, disk_size)

    if im is None:
        return f
    else:
        return f(im)
