"""
Image transformations.
"""
import cv2
import numpy as np

from skimage.color.adapt_rgb import adapt_rgb
from skimage.color.adapt_rgb import hsv_value, each_channel


def compose(func_list):

    def func(images, seed):
        # probs = np.random.rand(len(func_list), seed)
        # for image in images
        #   for func, prob in zip(func_list, probs):
        #       if random.rand(0,1, rand) < prob:
        #          image = func(image)
        #       else:
        #           image
        pass


def gaussian_blur(im=None, k=5, sigma=1):
    kwargs = {'ksize': k,
              'sigmaX': sigma}

    def blur(image, prob=1.0):
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


def downsample(im=None, size=None, adaptation='rgb', interpolation='linear'):
    """

    Parameters
    ----------
    im
    size
    adaptation
    interpolation: str
        One of ['linear', 'cubic', 'nearest'].

    Returns
    -------

    """
    assert size is None, "Must specify reshape size as tuple."

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    intp = 'INTER_LINEAR' if interpolation == 'linear' else \
           'INTER_CUBIC' if interpolation == 'cubic' else \
           'INTER_NEAREST'

    @adapt_rgb(adapt)
    def f(image):
        return cv2.resize(image, size, interpolation=intp)

    if im is None:
        return f
    else:
        return f(im)


def bin_image(bins=(0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255)):

    def f(image):
        # Set the first bin to zero
        return np.digitize(image, bins) - 1

    return f


def one_hot_image(index):

    def f(image):

        h, w, _ = image.shape
        oh_image = np.empty((h, w))

        for i in range(h):
            for j in range(w):
                key = tuple(image[i, j, :])
                oh_image[i, j] = index[key]

        return oh_image

    return f


def cantor(l):

    while len(l) > 1:
        a = l[0]
        b = l[1]

        del l[0]

        c = 0.5 * (a + b) * (a + b + 1) + b

        l[0] = c

    return l[0]
