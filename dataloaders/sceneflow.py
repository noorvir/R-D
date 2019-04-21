import re
import os
import h5py
import tarfile
import logging
import webp
import numpy as np
from time import time, gmtime, strftime

from os.path import join
from math import inf, floor
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from utils.dataio import read_pfm
from utils import transformations as transf
from utils.dataset_tools import encode_webp, decode_webp

# N.B. The baseline value is made up. The data-set does not provide it.
BASELINE = 0.5
FOCAL_LENGTH = 0.035

logging.getLogger().setLevel(logging.INFO)


def create_rdmo_dataset(rgb_path, disparity_path, material_path,
                        object_path, hdf5_path, dataset_name, chunk_size=1,
                        resize_factor=1.0, compression=5, num_points=inf):
    """

    Parameters
    ----------
    rgb_path: str
    disparity_path: str
    material_path: str
    object_path: str
    hdf5_path: str
    dataset_name: str
    chunk_size: int
    resize_factor: float
    compression: int
    num_points: int

    Returns
    -------

    """
    with h5py.File(rgb_path, 'r') as rgb_archive, \
            h5py.File(disparity_path, 'r') as disparity_archive, \
            h5py.File(material_path, 'r') as material_archive, \
            h5py.File(object_path, 'r') as object_archive, \
            h5py.File(hdf5_path, 'w') as h5f:

        def func(name):
            if dataset_name in name: return True

        for archive in [rgb_archive, disparity_archive, material_archive,
                        object_archive]:

            assert archive.visit(func), ("Dataset %s does not exisit in " 
                                         "archive %s. Exiting!" %
                                         (dataset_name, archive.name))

        logging.info("Starting HDF5 dataset creation...\n")
        start_time = time()

        # 1. Configure dataset properties
        dataset_name = dataset_name if dataset_name[0] != "/" else dataset_name[1:]

        rgb = rgb_archive[join(list(rgb_archive.keys())[0], dataset_name)]
        material = material_archive[join(list(material_archive.keys())[0], dataset_name)]
        obj = object_archive[join(list(object_archive.keys())[0], dataset_name)]
        disparity = disparity_archive[join(list(disparity_archive.keys())[0], dataset_name)]
        total_points = len(rgb)

        if num_points == inf:
            num_points = total_points

        shape2d = disparity.shape[1:3]
        shape2d = (int(shape2d[0] * resize_factor), int(shape2d[1] * resize_factor))

        if rgb.attrs['format'] == '.webp':
            rgb_shape = ()
        else:
            rgb_shape = shape2d + (3,)

        mask_shape = (num_points,) + shape2d

        # 2. Create new HDF5 data-sets
        logging.info("Creating data-set of %d points with chunk size %d.\n" %
                     (num_points, chunk_size))

        h5f.create_dataset('rgb', (num_points,) + rgb_shape, chunks=(chunk_size,) + rgb_shape,
                           compression=rgb.compression, compression_opts=rgb.compression_opts,
                           dtype=rgb.dtype)
        h5f['rgb'].attrs['format'] = rgb.attrs['format']

        h5f.create_dataset('depth', mask_shape, chunks=(chunk_size,) + shape2d,
                           compression=disparity.compression, compression_opts=compression,
                           dtype=disparity.dtype)
        h5f['depth'].attrs['format'] = disparity.attrs['format']

        h5f.create_dataset('material', mask_shape, chunks=(chunk_size,) + shape2d,
                           compression=material.compression, compression_opts=compression,
                           dtype=material.dtype)
        h5f['material'].attrs['format'] = material.attrs['format']

        h5f.create_dataset('object', mask_shape, chunks=(chunk_size,) + shape2d,
                           compression=obj.compression, compression_opts=compression,
                           dtype=obj.dtype)
        h5f['object'].attrs['format'] = obj.attrs['format']

        # 3. Convert disparity to depth
        depth = FOCAL_LENGTH * BASELINE / disparity[:num_points]

        # 4. Write to data-set - resize if specified
        if resize_factor != 1:

            for data, name in zip([rgb, depth, material, obj], ['rgb', 'depth', 'material', 'obj']):
                resize_func = transf.downsample() if name == 'rgb' else \
                    transf.downsample(interpolation='nearest')

                logging.info("Resizing and writing %d %s images to shape: %s"
                             % (num_points, name, (data.shape,)))

                for i in range(num_points):
                    if data.attrs['format'] == ".webp":
                        np_img = decode_webp(data[i])                                # decode
                        resized_img = resize_func(np_img)                            # resize
                        bytes_img = encode_webp(resized_img, ret="bytes")            # encode
                        h5f[name][i] = np.frombuffer(bytes_img, dtype=np.int8)       # assign
                    else:
                        h5f[name][i] = resize_func(data[i])

                    if i % 100 == 0:
                        logging.info("Processing %s datapoint number %d.\n" % (name, i))
        else:
            for data, name in zip([rgb, depth, material, obj], ['rgb', 'depth', 'material', 'object']):
                logging.info("Writing %d %s images" % (num_points, name))

                for i in range(num_points):
                    h5f[name][i] = data[i]

                    if i % 100 == 0:
                        logging.info("Processing %s datapoint number %d.\n" % (name, i))

        end_time = time() - start_time
        logging.info("\n\n*****Finished writing HDF5 dataset*****\n\n"
                     "HDF5 file saved at: %s\n"
                     "Data-points written: %d\n"
                     "Time taken: %s" % (hdf5_path,
                                         num_points,
                                         strftime("%H:%M:%S", gmtime(end_time))))


class RDMODataset(Dataset):

    def __init__(self, dataset_path, rgb_transforms, depth_transforms,
                 co_transforms, split='train', clip_size=inf):

        self.dataset_path = dataset_path
        self.rgb_transform = transf.compose(rgb_transforms)
        self.depth_transform = transf.compose(depth_transforms)
        self.co_transform = transf.compose(co_transforms)

    def __len__(self):
        with h5py.File(self.dataset_path, 'r') as ds:
            return len(ds['rgb'])

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as ds:
            rgb = ds['rgb'][idx]
            depth = ds['depth'][idx]
            material = ds['material'][idx]
            obj = ds['object'][idx]

        seed = np.random.randint(10000)
        # TODO: there might be some correlation b/w noise in rgb and depth
        rgb = self.rgb_transform(rgb, seed)
        depth = self.depth_transform(depth, seed)
        rgb, depth, material, obj = self.co_transform([rgb, depth, material, obj], seed)

        return rgb, depth, material, obj

    def visualse(self, idx):

        pass

# TODO:
# - Implement pyTorch dataset to access HDF5
# - implement triplet loss
# - train
#
# ###############################################################################
# # TESTS
# ###############################################################################
#
# def __test_tar_to_hdf5():
#
#     # make temp directory if it doesn't exisit
#     #
#     tar_to_hdf5()
#     pass