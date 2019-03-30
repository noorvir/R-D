import os
import logging
from math import inf
from PIL import Image
from torch.utils.data import Dataset

from utils.nn.transformations import


class DataPoint:
    def __init__(self, d, p, i):
        self.__dict__ = {}
        self.photo = d
        self.depth = p
        self.instance = i


class SceneNetDataset(Dataset):

    def __init__(self, root_path, split='train', use_shortcut=False,
                 transform=None, clip_size=inf):

        self.data_paths = []
        self.transform = transform
        self.root = os.path.join(root_path, split)

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

        if self.transform:

            # pass seed to ensure same randomness for both depth and rgb image
            # apply flip to target bot not other transformations

            pimage, dimage, iimage = self.transform(inputs=[pimage, dimage],
                                                    targets=[iimage])

        return DataPoint(pimage, dimage, iimage)
