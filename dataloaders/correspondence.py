import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.structures import DataTypes


def sample_from_mask(mask, num_samples, dtype):
    """
    Randomly sample pixels coordinates from a mask.

    Parameters
    ----------
    mask: torch.ByteTensor
        Binary image mask
    num_samples: int
        Number of samples
    dtype: utils.structures.DataTypes

    Returns
    -------
    Shape [num_samples, 2] torch.Tensor containing sampled pixel coordinates.
    """
    non_zero_pix = torch.nonzero(mask).type(dtype.long)
    high = non_zero_pix.shape[0]
    idx = torch.randint(0, high, (num_samples,)).type(dtype.long)

    return torch.index_select(non_zero_pix, dim=0, index=idx)


def sample_outside_mask(mask, num_samples, dtype):
    """
    Randomly sample pixels coordinates where mask is zero.

    Parameters
    ----------
    mask: torch.ByteTensor
        Binary image mask
    num_samples: int
        Number of samples
    dtype: utils.structures.DataTypes


    Returns
    -------
    Shape [num_samples, 2] torch.Tensor containing sampled pixel coordinates.
    """
    outside_mask_pix = torch.nonzero(~mask).type(dtype.long)
    high = outside_mask_pix.shape[0]
    idx = torch.randint(0, high, (num_samples,)).type(dtype.long)

    return torch.index_select(outside_mask_pix, dim=0, index=idx)


def get_image_values(image, pixel_idx, dtype):

    shape = image.shape
    if len(image.shape) == 3:
        h, w, c = shape
    else:
        h, w = shape

    image = torch.as_tensor(np.ascontiguousarray(image))
    image = image.type(dtype.long)

    pixel_idx = torch.as_tensor(np.ascontiguousarray(pixel_idx))
    pixel_idx = pixel_idx.type(dtype.long)

    flat_image = image.view(h * w, -1)
    flat_pix_idx = pixel_idx[:, 0] * w + pixel_idx[:, 1]

    return torch.index_select(flat_image, 0, flat_pix_idx), flat_pix_idx


def flat_to_2d_image(pix_vals, pix_idx, shape, dtype):
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

    zeros = torch.zeros(h * w, c).type(dtype.long)
    zeros[pix_idx] = pix_vals
    sparse_im = zeros.view(h, w, c)

    return sparse_im


def find_correspondences(material_mask, object_mask,
                         frac_correspondences=0.1,
                         non_correspondences_per_match=100,
                         frac_object_correspondences=0.0,
                         min_pixels_pruning_threshold=50,
                         device='CPU'):
    """

    Parameters
    ----------
    material_mask
    object_mask
    frac_correspondences: float
    non_correspondences_per_match: int
    frac_object_correspondences: float
    min_pixels_pruning_threshold: int
        Discard material (values) that have fewer pixels belonging to them
        than this threshold.
    device: str

    Returns
    -------

    """
    h, w = material_mask.shape
    dtype = DataTypes(device)

    ones = torch.ones(h, w).type(dtype.byte)
    zeros = torch.zeros(h, w).type(dtype.byte)

    if type(material_mask) is np.ndarray:
        material_mask = torch.as_tensor(np.ascontiguousarray(material_mask))
        material_mask = material_mask.type(dtype.long)

    if type(object_mask) is np.ndarray:
        object_mask = torch.as_tensor(np.ascontiguousarray(object_mask))
        object_mask = object_mask.type(dtype.long)

    # Get unique mask values a prune to remove spurious values
    material_mask_unique_vals, counts = torch.unique(material_mask, return_counts=True)
    idx_mask = torch.where(counts > min_pixels_pruning_threshold,
                           torch.ones(counts.shape).type(dtype.byte),
                           torch.zeros(counts.shape).type(dtype.byte))

    material_mask_unique_vals = torch.masked_select(material_mask_unique_vals, idx_mask)

    correspondence_list = []

    # TODO: could get rid of this for loop by flattening images somehow
    for mval in material_mask_unique_vals:
        mask = torch.where(material_mask == mval, ones, zeros).type(dtype.byte)
        num_matches = max((torch.sum(mask).float() * frac_correspondences).int().item(), 1)
        num_non_matches = num_matches * non_correspondences_per_match

        # Get matches from the same material index. For each match, sample
        # non_correspondences_per_match pixels outside the material mask.
        matched_pixels = sample_from_mask(mask, num_matches, dtype)

        oval = object_mask[matched_pixels[0, 0]]
        omask = torch.where(object_mask == oval, ones, zeros)

        if frac_object_correspondences == 0:
            non_matched_pixels = sample_outside_mask(mask, num_non_matches, dtype)
        else:
            non_matched_pixels = sample_outside_mask(omask, num_non_matches, dtype)

        # For every matched pixel, sample num_obj_matches from the object
        # mask that the pixel (material) belongs to.
        # idx = w * matched_pixels[0, 1] + matched_pixels[0, 0]
        num_obj_matches = (torch.sum(omask).float() * frac_object_correspondences).int().item() * num_matches
        obj_matched_pixels = sample_from_mask(omask, num_obj_matches, dtype)

        correspondence_list.append((matched_pixels, non_matched_pixels, obj_matched_pixels))

    # TODO: Considerations
    # - Sample close to thin material regions
    # - Think about proximity clustering

    return correspondence_list


if __name__ == "__main__":

    from tools.dataio import read_pfm
    mat_path = "/home/noorvir/Documents/projects/geometricOS/research_and_dev/data/material.pfm"
    obj_path = "/home/noorvir/Documents/projects/geometricOS/research_and_dev/data/object.pfm"

    m1, _ = read_pfm(mat_path)
    o1, _ = read_pfm(obj_path)

    l = find_correspondences(m1, o1, device='GPU')

    print(l)