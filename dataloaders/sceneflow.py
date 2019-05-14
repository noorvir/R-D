import re
import os
import h5py
import tarfile
import logging
import webp
import numpy as np

from os.path import join
from math import inf, floor
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from time import time, gmtime, strftime
from torchvision.transforms import Compose

from tools.dataio import read_pfm
from tools import transformations as tfs
from tools.dataset_tools import encode_webp, decode_webp, ImageDatasetStats

# N.B. The baseline value is made up. The data-set does not provide it.
BASELINE = 0.5
FOCAL_LENGTH = 1050     # 0.035 mm

logging.getLogger().setLevel(logging.INFO)


def create_rdmo_dataset(rgb_path, disparity_path, material_path,
                        object_path, hdf5_path, dataset_name, chunk_size=1,
                        resize_factor=1.0, compression=5, num_points=inf,
                        train_val_split=(0.8, 0.2), seed=140693):
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
    train_val_split: tuple(float, float)
    seed: int

    Returns
    -------

    """
    assert np.sum(train_val_split) == 1, "Train/val split must add up to 1.0."

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

        # Stats
        dataset_stats = {'rgb': ImageDatasetStats(),
                         'depth': ImageDatasetStats(),
                         'material': ImageDatasetStats(),
                         'obj': ImageDatasetStats()}

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

        h5f.create_dataset('obj', mask_shape, chunks=(chunk_size,) + shape2d,
                           compression=obj.compression, compression_opts=compression,
                           dtype=obj.dtype)
        h5f['obj'].attrs['format'] = obj.attrs['format']

        # 3. Convert disparity to depth
        depth = FOCAL_LENGTH * BASELINE / disparity[:num_points]

        for data, name in zip([rgb, depth, material, obj], ['rgb', 'depth', 'material', 'obj']):

            if name == 'depth':
                dtype = np.float32
            else:
                dtype = data.attrs['format']

            resize_func = tfs.downsample() if name == 'rgb' else \
                tfs.downsample(interpolation='nearest')

            logging.info("Writing %d %s images to shape: %s" % (num_points, name, (data.shape,)))

            # 4. Iterate and add to data-set
            for i in range(num_points):
                curr_data = data[i]

                if dtype == ".webp":
                    curr_data = decode_webp(data[i])                        # decode

                if resize_factor != 1:
                    curr_data = resize_func(curr_data)                      # resize

                dataset_stats[name].update(curr_data)

                if dtype == ".webp":
                    curr_data = encode_webp(curr_data, ret="bytes")         # encode
                    h5f[name][i] = np.frombuffer(curr_data, dtype=np.int8)  # assign
                else:
                    h5f[name][i] = curr_data

                if i % 100 == 0:
                    logging.info("Processing %s datapoint number %d.\n" % (name, i))

            h5f[name].attrs['mean'] = dataset_stats[name].mean
            h5f[name].attrs['std_dev'] = dataset_stats[name].std_dev
            h5f[name].attrs['var'] = dataset_stats[name].var
            h5f[name].attrs['min'] = dataset_stats[name].min
            h5f[name].attrs['max'] = dataset_stats[name].max

        # 6. Create train/val/test indices
        idx = np.arange(num_points)
        np.random.seed(seed)
        np.random.shuffle(idx)
        split_idx = int(train_val_split[0] * num_points)
        train_idx, val_idx = idx[: split_idx], idx[split_idx:]
        h5f.attrs['train_idx'] = train_idx
        h5f.attrs['val_idx'] = val_idx

        end_time = time() - start_time
        logging.info("\n\n*****Finished writing HDF5 dataset*****\n\n"
                     "HDF5 file saved at: %s\n"
                     "Data-points written: %d\n"
                     "Time taken: %s" % (hdf5_path,
                                         num_points,
                                         strftime("%H:%M:%S", gmtime(end_time))))


# *************************************************************************************************
# RGB, Depth, Material, Object - PyTorch data-set access
# *************************************************************************************************

class RDMODataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.rgb_transform = tfs.compose([])
        self.depth_transform = tfs.compose([])
        self.co_transform = tfs.compose([])
        self.dataset_stats = self._get_dataset_stats()

        with h5py.File(self.dataset_path, 'r') as ds:
            self.train_idx = ds.attrs['train_idx']
            self.val_idx = ds.attrs['val_idx']

    def __len__(self):
        with h5py.File(self.dataset_path, 'r') as ds:
            return len(ds['rgb'])

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as ds:
            rgb = ds['rgb'][idx]
            depth = ds['depth'][idx]
            material = ds['material'][idx]
            obj = ds['obj'][idx]

        rgb = decode_webp(rgb)

        # TODO: there might be some correlation b/w noise in rgb and depth
        rgb = self.rgb_transform(rgb, seed=np.random.randint(10000))
        depth = self.depth_transform(depth, seed=np.random.randint(10000))
        rgb, depth, material, obj = self.co_transform([rgb, depth, material, obj],
                                                      seed=np.random.randint(10000))

        return (rgb, depth), (material, obj)

    def _get_dataset_stats(self):
        """

        Returns
        -------

        """
        stats = {}
        names = ['rgb', 'depth', 'material', 'obj']

        with h5py.File(self.dataset_path, 'r') as ds:
            for name in names:
                stats[name] = ImageDatasetStats()
                stats[name].mean = ds[name].attrs['mean']
                stats[name].std_dev = ds[name].attrs['std_dev']
                stats[name].var = ds[name].attrs['var']
                stats[name].min = ds[name].attrs['min']
                stats[name].max = ds[name].attrs['max']

        return stats

    def setup_transformations(self, rgb_transforms, depth_transforms, co_transforms):
        self.rgb_transform = tfs.compose(rgb_transforms)
        self.depth_transform = tfs.compose(depth_transforms)
        self.co_transform = tfs.compose(co_transforms)

    def visualse(self, idx):
        images = self.__getitem__(idx)
        fig, axes = plt.subplots(2, 2)

        for image, axe in zip(images, axes.reshape(-1)):
            axe.imshow(image)

        plt.show()

# TODO:
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