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

from utils.dataio import read_pfm
from utils import transformations as transf
from utils.dataset_tools import encode_webp, decode_webp

# N.B. The baseline value is made up. The data-set does not provide it.
BASELINE = 0.5
FOCAL_LENGTH = 0.035

logging.getLogger().setLevel(logging.INFO)


def shift_dataset(dataset):

    if type(dataset) ==  h5py._hl.group.Group or \
            type(dataset) ==  h5py._hl.files.File:

        keys = dataset.keys()

        for key in keys:
            shift_dataset(dataset[key])
    else:
        dataset[0:-1] = dataset[1:]
        dataset.resize(len(dataset) - 1, axis=0)
        logging.info("Shifted data group %s" % dataset.name)
        logging.info("New size %s" % (dataset.shape,))


def tar_to_hdf5(tar_path, hdf5_path, max_size=5000, compression=9):
    """
    Convert tarfile to HDF5 database which allows indexing.
    Parameters
    ----------
    tar_path: str
    hdf5_path: str
    max_size: int
    compression: int

    Returns
    -------

    """
    start_time = time()
    total_datapoints = 0

    with tarfile.open(tar_path, 'r') as archive, \
            h5py.File(hdf5_path, 'w') as h5f:

        datasets = {}

        while True:
            member = archive.next()

            if member is None:
                break

            name = member.name
            substring, suffix = os.path.splitext(name)

            if suffix == "":
                continue

            data = archive.extractfile(member)
            dataset_name = re.split('\d+(?=\.)', name)[0]
            datapoint_idx = int(re.search('\d+(?=\.)', name).group(0)) - 1

            if suffix == '.pfm':
                data, scale = read_pfm(data)
            elif suffix == '.webp':
                data = data.read()
                data = np.frombuffer(data, dtype=np.int8)
                compression = 0
            else:
                data = imread(data.read())

            # If first image in group, create new data-set
            if dataset_name not in datasets:

                if suffix == '.webp':
                    data_shape = ()
                    data_type = h5py.special_dtype(vlen=data.dtype)
                else:
                    data_shape = data.shape
                    data_type = data.dtype

                shape = (max_size,) + data_shape
                chunk_shape = (1,) + data_shape
                logging.info("Creating subgroup: %s of shape: %s \n\n"
                             % (dataset_name, (shape,)))
                datasets[dataset_name] = [h5f.create_dataset(dataset_name,
                                                             shape,
                                                             chunks=chunk_shape,
                                                             compression='gzip',
                                                             compression_opts=compression,
                                                             dtype=data_type),
                                          0]
                datasets[dataset_name][0].attrs['format'] = suffix

            # Get index into dataset
            dataset, max_idx = datasets[dataset_name]
            dataset[datapoint_idx] = data

            # Update the max data index seen so far for pruning later
            datasets[dataset_name][1] = datapoint_idx if datapoint_idx > max_idx else max_idx
            total_datapoints += 1

            if total_datapoints % 100 == 0:
                logging.info("Processing datapoint number %d.\n" % total_datapoints)

        # Prune
        for key, val in datasets.items():
            dataset, max_idx = val
            datasets[key] = dataset.resize(max_idx + 1, axis=0)
            logging.info("Pruned dataset %s to size %d points.\n" % (key, max_idx + 1))

    end_time = time() - start_time
    logging.info("*****Finished converting tarfile to HDF5 dataset*****\n\n"
                 "HDF5 file saved at: %s\n"
                 "Tar to HDF5 conversion done in %s" %
                 (hdf5_path, strftime("%H:%M:%S", gmtime(end_time))))


def create_thingnet_dataset(rgb_path, disparity_path, material_path,
                            object_path, hdf5_path, dataset_name, chunk_size,
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

        shape2d = disparity.shape
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
            for data, name in zip([rgb, depth, material, obj], ['rgb', 'depth', 'material', 'obj']):
                logging.info("Writing %d %s images" % (num_points, name))

                for i in range(num_points):
                    h5f[name][i] = data[i]

                    if i % 100 == 0:
                        logging.info("Processing %s datapoint number %d.\n" % (name, i))

        # 5. Add meta-data
        # TODO: add h5py attributes

        end_time = time() - start_time
        logging.info("*****Finished writing HDF5 dataset*****\n\n"
                     "HDF5 file saved at: %s\n"
                     "Data-points written: %d\n"
                     "Time taken: %s" % (hdf5_path,
                                         num_points,
                                         strftime("%H:%M:%S", gmtime(end_time))))


class DataPoint:
    def __init__(self, d, p, i):
        self.__dict__ = {}
        self.photo = d
        self.depth = p
        self.instance = i


class SceneFlowDataset(Dataset):

    def __init__(self, root_path, split='train', use_shortcut=False,
                 clip_size=inf):

        self.dataset_path = ""
        self.root = os.path.join(root_path, split)

    def __len__(self):
        with h5py.File(self.dataset_path, 'r') as ds:
            return len(ds['rgb'])

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as ds:
            rgb = ds['rgb']
            depth = ds['depth']
            material = ds['material']
            obj = ds['obj']

        sample = {'images': image, 'labels': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

#
# # Loading from h5py
# class DeephomographyDataset(Dataset):
#
#
# def __init__(self, hdf5file, imgs_key='images', labels_key='labels',
#              transform=None):
#     self.hdf5file = hdf5file
#
#     self.imgs_key = imgs_key
#     self.labels_key = labels_key
#     self.transform = transform
#
#
# def __len__(self):
#     # return len(self.db[self.labels_key])
#     with h5py.File(self.hdf5file, 'r') as db:
#         lens = len(db[self.labels_key])
#     return lens
#
#
# def __getitem__(self, idx):
#     with h5py.File(self.hdf5file, 'r') as db:
#         image = db[self.imgs_key][idx]
#         label = db[self.labels_key][idx]
#     sample = {'images': image, 'labels': label}
#     if self.transform:
#         sample = self.transform(sample)
#     return sample
#
# # TODO:
# # - Implement pyTorch dataset to access HDF5
# # - implement triplet loss
# # - train
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