import re
import os
import h5py
import tarfile
import logging
import numpy as np
from time import time, gmtime, strftime

from math import inf, floor
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset

from utils.dataio import read_pfm
from utils import transformations as transf

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
            datasets[key] = dataset.resize(max_idx, axis=0)
            logging.info("Pruned dataset %s to size %d points.\n" % (key, max_idx + 1))

    end_time = time() - start_time
    logging.info("Tar to HDF5 conversion done in %s" %
                 strftime("%H:%M:%S", gmtime(end_time)))


def create_thingnet_dataset(rgb_path, disparity_path, material_path,
                            object_path, hdf5_path, resize_factor=1.0,
                            compression=5, num_points=inf):
    """

    Parameters
    ----------
    rgb_path: str
    disparity_path: str
    material_path: str
    object_path: str
    hdf5_path: str
    resize_factor: float
    compression: int
    num_points: int

    Returns
    -------

    """
    with tarfile.open(rgb_path, 'r') as rgb_archive, \
            tarfile.open(disparity_path, 'r') as disparity_archive, \
            tarfile.open(material_path, 'r') as material_archive, \
            tarfile.open(object_path, 'r') as object_archive, \
            h5py.File(hdf5_path, 'w') as h5f:

        # Get member names and filter
        names = rgb_archive.getnames()
        names = [name for name in names if ('15mm_focallength' not in name) and
                                           ('scene_backwards' not in name) and
                                           ('slow' not in name) and
                                           ('right' not in name) and
                                           (os.path.splitext(name)[-1] != "")]

        if len(names) == 0:
            return

        # Image details
        _member = rgb_archive.getmember(names[0])
        _data = rgb_archive.extractfile(_member)
        image = imread(_data.read())
        h, w, c = image.shape
        h_out = floor(resize_factor * h)
        w_out = floor(resize_factor * w)

        h5_datasets = [h5f.create_dataset("rgb", (len(names), h_out, w_out, c),
                                          compression="gzip",
                                          compression_opts=compression),
                       h5f.create_dataset("depth", (len(names), h_out, w_out),
                                          compression="gzip",
                                          compression_opts=compression),
                       h5f.create_dataset("material", (len(names), h_out, w_out),
                                          compression="gzip",
                                          compression_opts=compression),
                       h5f.create_dataset("object", (len(names), h_out, w_out),
                                          compression="gzip",
                                          compression_opts=compression)]

        datapoint_idx = 0
        logging.info("Starting tarfile to HDF5 conversion\n")
        start_time = time()

        for name in names:
            substring, suffix = os.path.splitext(name)
            common_substring = re.split('^([a-z_]*(?=/))', substring)[2]

            # Get names
            disparity_member_name = 'disparity' + common_substring
            material_member_name = 'material_idx' + common_substring
            object_member_name = '' + common_substring

            # Get members from names
            rgb_member = rgb_archive.getmember(name)
            disparity_member = disparity_archive.getmember(disparity_member_name)
            material_member = material_archive.getmember(material_member_name)
            object_member = object_archive.getmember(object_member_name)

            member_num = 0

            # Read and add to HDF5
            for member, archive in [(rgb_member, rgb_archive),
                                    (disparity_member, disparity_archive),
                                    (material_member, material_archive),
                                    (object_member, object_archive)]:

                data = archive.extractfile(member)

                if suffix == '.pfm':
                    data, scale = read_pfm(data)
                else:
                    data = imread(data.read())  # read bytes

                if member_num == 0:
                    data = transf.downsample(data, (h_out, w_out))
                else:
                    data = transf.downsample(data, (h_out, w_out),
                                             interpolation='nearest')

                if member_num == 1:
                    # Convert to depth
                    data = FOCAL_LENGTH * BASELINE / data

                h5_datasets[member_num][datapoint_idx] = data
                member_num += 1

            if (datapoint_idx + 1) % 100 == 0:
                logging.info("Adding datapoint %d of %d" % (datapoint_idx + 1,
                                                            len(names)))

            datapoint_idx += 1

            if datapoint_idx >= num_points:
                logging.info("Existing after %d points" % num_points)
                break

        end_time = time() - start_time
        logging.info("*****Finished converting tarfile to HDF5 dataset*****\n\n"
                     "HDF5 file saved at: %s\n"
                     "Data-points written: %d\n"
                     "Time taken: %s" % (hdf5_path,
                                         datapoint_idx,
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

        self.root = os.path.join(root_path, split)
        self.data_paths = []

        data_point_count = 0

        if use_shortcut:
            logging.warning("Creating database index assuming that every "
                            "trajectory contains 300 datapoints and the names "
                            "of datapoints are multiples of 25: 0, 25, 50 etc.")

            num_subsets = len(os.listdir(self.root))

            for subset in range(num_subsets):
                num_seqs = len(os.listdir(os.path.join(self.root, str(subset))))
                for seq in range(num_seqs):
                    for j in range(0, 7500, 25):
                        self.data_paths += [[os.path.join(self.root,
                                                          str(subset),
                                                          str(seq)), j]]

        else:
            for subset_dir in os.listdir(self.root):
                subset_dir = os.path.join(self.root, subset_dir)

                for seq_dir in os.listdir(subset_dir):
                    seq_dir = os.path.join(subset_dir, seq_dir)

                    data_dir = os.path.join(seq_dir, "depth")
                    datapoint_names = os.listdir(data_dir)
                    datapoint_paths = [[seq_dir, name[:-4]]
                                       for name in datapoint_names]

                    data_point_count += len(datapoint_names)

                    if data_point_count > clip_size:
                        eidx = data_point_count - clip_size
                        datapoint_paths = datapoint_paths[:-eidx]
                        self.data_paths += datapoint_paths
                        return

                    self.data_paths += datapoint_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, item):

        ppath = os.path.join(self.data_paths[item][0], 'photo',
                             str(self.data_paths[item][1]) + ".jpg")

        dpath = os.path.join(self.data_paths[item][0], 'depth',
                             str(self.data_paths[item][1]) + ".png")

        ipath = os.path.join(self.data_paths[item][0], 'instance',
                             str(self.data_paths[item][1]) + ".png")

        pimage = Image.open(ppath)
        dimage = Image.open(dpath)
        iimage = Image.open(ipath)

        return DataPoint(pimage, dimage, iimage)

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