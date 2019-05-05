"""
Image transformations.
"""
import cv2
import torch
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
        rands = np.random.rand(len(func_list)).tolist()

        for image in images:
            for func, rand in zip(func_list, rands):
                prob = 0.5
                prob = func.prob if func.prob is not None else prob

                if rand < prob:
                    image = func(image)

            transf_images.append(image)

        return transf_images if len(transf_images) > 1 else transf_images[0]

    return f


def type_converter(im=None, dtype=np.float32):

    def f(image):
        if type(dtype) == torch.dtype:
            return torch.tensor(image, dtype=dtype)
        return image.astype(dtype=dtype)

    f.prob = 1.0
    if im is None:
        return f
    else:
        return f(im)


def NHWC_to_NCHW(im=None, prob=1.0):
    """Convert to pytorch image convention"""

    def f(image):
        shape = np.shape(image)
        if len(shape) == 3:
            if shape[2] < shape[0] and shape[2] < shape[1]:
                return np.transpose(image, (2, 0, 1))
        return image

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def random_noise(im=None, frac=0.01, scale=0.05, dist='normal', prob=None):
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

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def gaussian_blur(im=None, k=5, sigma=1, prob=None):
    kwargs = {'ksize': (k, k),
              'sigmaX': sigma}

    def blur(image, prob=1.0):
        return cv2.GaussianBlur(image, **kwargs)

    blur.prob = prob

    if im is None:
        return blur
    else:
        return blur(im)


def random_dropout(im=None, dropout_prob=0.3, prob=None):

    def f(image):
        pass

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def scale_min_max(im=None, max_val=1, log_scale=False, adaptation='rgb', prob=None):
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

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def normalise(im=None, mean=None, std_dev=None, prob=None):
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

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def depth_to_proximity(mean=None, prob=1.0):

    # f.prob = prob

    pass


def flip_vertical(im=None, prob=None):
    def f(image):
        return cv2.flip(image, flipCode=0)

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def flip_horizontal(im=None, prob=None):
    def f(image):
        return cv2.flip(image, flipCode=1)

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def gradient(im=None, blur=None, adaptation='rgb', prob=None):

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):

        if blur is not None:
            image = blur(image)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        return np.sqrt((sobelx ** 2) + (sobely ** 2))

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def laplacian(im=None, blur=None, adaptation='rgb', prob=None):

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):

        if blur is not None:
            image = blur(image)

        limg = cv2.Laplacian(image, cv2.CV_64F)
        limg = (limg - limg.min()) / (limg.max() - limg.min())

        return limg

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def median(im=None, disk_size=2, adaptation='rgb', prob=None):

    adapt = hsv_value if adaptation == 'hsv' else each_channel

    @adapt_rgb(adapt)
    def f(image):
        return cv2.medianBlur(image, disk_size)

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def downsample(im=None, size=None, adaptation='rgb', interpolation='linear', prob=None):
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

    f.prob = prob

    if im is None:
        return f
    else:
        return f(im)


def bin_image(bins=(0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255), prob=None):

    def f(image):
        # Set the first bin to zero
        return np.digitize(image, bins) - 1

    return f


def one_hot_image(index, prob=None):

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
