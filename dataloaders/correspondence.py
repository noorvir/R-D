import torch
import numpy as np


def sample_from_mask(mask, num_samples):
    """
    Randomly sample pixels coordinates from a mask.

    Parameters
    ----------
    mask: torch.ByteTensor
        Binary image mask
    num_samples: int
        Number of samples

    Returns
    -------
    Shape [num_samples, 2] torch.Tensor containing sampled pixel coordinates.
    """
    non_zero_pix = torch.nonzero(mask)
    high = non_zero_pix.shape[0]
    idx = torch.randint(0, high, (num_samples,))

    return torch.index_select(non_zero_pix, dim=0, index=idx)


def sample_outside_mask(mask, num_samples):
    """
    Randomly sample pixels coordinates where mask is zero.

    Parameters
    ----------
    mask: torch.ByteTensor
        Binary image mask
    num_samples: int
        Number of samples

    Returns
    -------
    Shape [num_samples, 2] torch.Tensor containing sampled pixel coordinates.
    """
    outside_mask_pix = torch.nonzero(~mask)
    high = outside_mask_pix.shape[0]
    idx = torch.randint(0, high, (num_samples,))

    return torch.index_select(outside_mask_pix, dim=0, index=idx)


def get_image_values(image, pixel_idx):

    shape = image.shape
    if len(image.shape) == 3:
        h, w, c = shape
    else:
        h, w = shape

    image = torch.as_tensor(np.ascontiguousarray(image))
    image = image.type(torch.LongTensor)

    pixel_idx = torch.as_tensor(np.ascontiguousarray(pixel_idx))
    pixel_idx = pixel_idx.type(torch.LongTensor)

    flat_image = image.view(h * w, -1)
    flat_pix_idx = pixel_idx[:, 0] * w + pixel_idx[:, 1]

    return torch.index_select(flat_image, 0, flat_pix_idx), flat_pix_idx


def flat_to_2d_image(pix_vals, pix_idx, shape):
    """

    Parameters
    ----------
    pix_vals
    pix_idx
    shape

    Returns
    -------

    """
    h = shape[0]
    w = shape[1]
    c = -1 if len(shape) == 2 else shape[2]

    zeros = torch.zeros(h * w, c).type(torch.LongTensor)
    zeros[pix_idx] = pix_vals
    sparse_im = zeros.view(h, w, c)

    return sparse_im


def find_correspondences(material_mask, object_mask,
                         frac_correspondences=0.1,
                         non_correspondences_per_match=100,
                         frac_object_correspondences=0,
                         device='CPU'):
    """

    Parameters
    ----------
    material_mask
    object_mask
    frac_correspondences
    non_correspondences_per_match
    frac_object_correspondences
    device

    Returns
    -------

    """
    h, w = material_mask.shape

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    if device == 'GPU':
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    if type(material_mask) is np.ndarray:
        material_mask = torch.as_tensor(np.ascontiguousarray(material_mask))
        material_mask = material_mask.type(dtype_long)

    if type(object_mask) is np.ndarray:
        object_mask = torch.as_tensor(np.ascontiguousarray(object_mask))
        object_mask = object_mask.type(dtype_long)

    material_mask_unique_vals = torch.unique(material_mask)

    ones = torch.ones(h, w).type(dtype_float)
    zeros = torch.zeros(h, w).type(dtype_float)

    correspondence_list = []

    # TODO: could get rid of this for loop by flattening images somehow
    for mval in material_mask_unique_vals:
        mask = torch.where(material_mask == mval, ones, zeros)
        num_matches = int(torch.sum(mask) * frac_correspondences)
        num_non_matches = num_matches * non_correspondences_per_match

        # Get matches from the same material index. For match, sample
        # non_correspondences_per_match pixels outside the material mask.
        matched_pixels = sample_from_mask(mask, num_matches)
        non_matched_pixels = sample_outside_mask(mask, num_non_matches)

        # For every matched pixel, sample num_obj_matches from the object
        # mask that the pixel (material) belongs to.
        idx = w * matched_pixels[0, 1] + matched_pixels[0, 0]
        oval = object_mask[idx]
        omask = torch.where(object_mask == oval, ones, zeros)
        num_obj_matches = int(torch.sum(omask) * frac_object_correspondences) * num_matches
        obj_matched_pixels = sample_from_mask(omask, num_obj_matches)

        correspondence_list.append((matched_pixels, non_matched_pixels, obj_matched_pixels))

    # TODO: Considerations
    # - Sample close to thin material regions
    # - Think about proximity clustering

    return correspondence_list
