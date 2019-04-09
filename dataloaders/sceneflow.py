import re
import os
import h5py
import tarfile
import logging
from time import time, gmtime, strftime

from math import inf
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset

from utils.dataio import read_pfm

# N.B. The baseline value is made up. The data-set does not provide it.
BASELINE = 0.5
FOCAL_LENGTH = 0.035


def tar_to_hdf5(rgb_path, disparity_path, material_path, object_path,
                compression=5):
    """

    Parameters
    ----------
    rgb_path
    disparity_path
    material_path
    object_path
    compression

    Returns
    -------

    """
    with tarfile.open(rgb_path, 'r') as rgb_archive, \
            tarfile.open(disparity_path, 'r') as disparity_archive, \
            tarfile.open(material_path, 'r') as material_archive, \
            tarfile.open(object_path, 'r') as object_archive, \
            h5py.File('', 'w') as h5f:

        # Get member names and filter
        names = rgb_archive.getnames()
        names = [name for name in names if ('15mm_focallength' not in name) and
                                           ('scene_backwards' not in name) and
                                           ('slow' not in name) and
                                           ('right' not in name)]

        if len(names) == 0:
            return

        h5_datasets = [h5f.create_dataset("rgb", (len(names), 0, 0, 3) ,
                                          "gzip", compression),
                       h5f.create_dataset("depth", (len(names), 0, 0),
                                          "gzip", compression),
                       h5f.create_dataset("material", (len(names), 0, 0),
                                          "gzip", compression),
                       h5f.create_dataset("object", (len(names), 0, 0),
                                          "gzip", compression)]

        datapoint_idx = 0
        logging.info("Starting tarfile to HDF5 conversion\n")
        start_time = time()

        for name in names:
            substring, suffix = os.path.splitext(name)
            common_substring = re.split('^([a-z_]*(?=/))', substring)[2]

            # Get names
            disparity_member_name = '' + common_substring
            material_member_name = '' + common_substring
            object_member_name = '' + common_substring

            # Get members from names
            rgb_member = disparity_archive.getmember(name)
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

                # Convert to depth
                if member_num == 1:
                    data = FOCAL_LENGTH * BASELINE / data

                h5_datasets[member_num][datapoint_idx] = data
                member_num += 1

            if (datapoint_idx + 1) % 100 == 0:
                logging.info("Adding datapoint %d of %d" % (datapoint_idx + 1,
                                                            len(names)))

            datapoint_idx += 1

        end_time = time() - start_time
        logging.info("*****Finished converting tarfile to HDF5 dataset*****\n\n"
                     "Data-points written: %d\n"
                     "Time taken: %s" % (datapoint_idx,
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
