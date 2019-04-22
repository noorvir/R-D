"""
Image transformations.
"""
import cv2
import numpy as np

from math import inf
from skimage.color.adapt_rgb import adapt_rgb
from skimage.color.adapt_rgb import hsv_value, each_channel


def select_2d_idx(shape, num):
    """

    Parameters
    ----------
    shape
    num

    Returns
    -------

    """
    rows = np.arange(shape[0])
    cols = np.arange(shape[1])
    num_pixels = shape[0] * shape[1]
    xy = np.empty((shape[0], shape[1], 2), dtype=int)
    xy[:, :, 0] = rows[:, None]
    xy[:, :, 1] = cols
    xy_flat = xy.reshape(-1, 2)
    idx = np.random.choice(np.arange(num_pixels),
                           size=num, replace=False)

    return xy_flat[idx]


def compose(func_list):
    """
    Create pipeline to apply a series of transformations to an image/images.

    Parameters
    ----------
    func_list

    Returns
    -------

    """
    assert type(func_list) is list, ("Argument to compose function"
                                         "must be a list of functions.")

    def f(images, seed):
        if type(images) is not list:
            images = [images]

        transf_images = []
        np.random.seed(seed)
        probs = np.random.rand(len(func_list)).tolist()

        for image in images:
            for func, prob in zip(func_list, probs):
                if 0.5 < prob:
                    image = func(image)

            transf_images.append(image)

        return transf_images if len(transf_images) > 1 else transf_images[0]

    return f


def random_noise(im=None, frac=0.01, scale=0.05, dist='normal'):
    """

    Parameters
    ----------
    im
    frac:
    scale:
    dist:

    Returns
    -------

    """
    assert frac <= 1.0, "frac must be <= 1.0"

    def f(image):
        shape = np.shape(image)
        im_max = np.max(image)
        num_pixels = shape[0] * shape[1]
        num_noise_pixels = int(num_pixels * frac)
        noise_image = np.zeros(shape)

        noise_idx = select_2d_idx(shape, num_noise_pixels)
        noise_idx_h = noise_idx[:, 0]
        noise_idx_w = noise_idx[:, 1]

        outshape = num_noise_pixels

        if len(shape) > 2:
            outshape = (num_noise_pixels, shape[2])

        if dist == 'normal':
            noise = np.random.normal(size=outshape) * scale * im_max
        else:
            noise = np.random.rand(*outshape) * scale * im_max

        noise_image[noise_idx_h, noise_idx_w] = noise
        noise_image = noise_image.reshape(shape)

        ret = image + noise_image
        ret = np.clip(ret, 0, im_max)

        return ret

    if im is None:
        return f
    else:
        return f(im)


def gaussian_blur(im=None, k=5, sigma=1):
    kwargs = {'ksize': (k, k),
              'sigmaX': sigma}

    def blur(image, prob=1.0):
        return cv2.GaussianBlur(image, **kwargs)

    if im is None:
        return blur
    else:
        return blur(im)


def random_dropout(im=None, dropout_prob=0.3):

    def dropout(image):
        pass

    if im is None:
        return dropout
    else:
        return dropout(im)


def scale_min_max(im=None, max_val=1, log_scale=False, adaptation='rgb'):
    """
    Normalise to range [0, 1].

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


def normalise(im=None, mean=None, std_dev=None):
    """
    Standard score normalisation.

    Parameters
    ----------
    im
    mean
    std_dev

    Returns
    -------

    """

    assert mean is not None, "Must provide data-set mean for normalisation.\n"
    assert std_dev is not None, "Must provide data-set std_dev for normalisation.\n"

    def f(image):
        return (image - mean)/std_dev

    if im is None:
        return f
    else:
        return f(im)


def flip_vertical(im=None):
    def f(image):
        return cv2.flip(image, flipCode=0)

    if im is None:
        return f
    else:
        return f(im)


def flip_horizontal(im=None):
    def f(image):
        return cv2.flip(image, flipCode=1)

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
