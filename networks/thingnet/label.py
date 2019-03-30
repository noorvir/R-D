import os
import re
import copy
import pickle
import numpy as np
import logging
import tarfile

from utils.dataio import read_pfm


class LabelIndex(dict):
    def __init__(self):
        super(LabelIndex, self).__init__()
        self._reverse_dict = {}

    def __setitem__(self, key, value):

        if (type(value) is list) or (type(value) is np.ndarray):
            value = tuple(value)

        assert key != value, ("Key and value cannot be the same in a two-way"
                              "index")

        # Remove any previous connections with these values
        if key in self:
            del self[key]

        dict.__setitem__(self, key, value)
        self._reverse_dict.__setitem__(value, key)

    def __delitem__(self, key):
        value = self[key]
        dict.__delitem__(self, key)
        del self._reverse_dict[value]

    def get_reverse(self, key):
        return self._reverse_dict[key]


def create_thing_index(vecs, save=False, cache_dir=None):

    """

    Parameters
    ----------
    vecs: list
        list of lists/tuples
    save: bool

    cache_dir: str

    Returns
    -------

    """
    num_labels = len(vecs)
    label_idx = LabelIndex()

    for i, vec in enumerate(vecs):

        if (type(vec) is list) or (type(vec) is np.ndarray):
            vec = tuple(vec)

        if vec in label_idx:
            logging.warning("Label already exists! Overwriting!")

        one_hot_vec = np.zeros(num_labels)
        one_hot_vec[i] = 1

        label_idx[vec] = copy.copy(one_hot_vec)

    if save:
        try:
            with open(cache_dir) as f:
                pickle.dump(label_idx, f)

            logging.info("Label Index saved as pickle at %s" % cache_dir)

        except TypeError as e:
            logging.error("Must pass a string for argument 'cache_dir' if "
                          "'save = True'\n %s" % e)

    return label_idx


def relabel_from_tar(output_dir, suffix='.pfm'):

    tf = ""
    archive = tarfile.open(tf, 'r')
    names = archive.getnames()
    members = archive.getmembers()

    for name, member in zip(names, members):

        if suffix not in name:
            continue

        f = archive.extractfile(member)

        if suffix == ".pfm":
            data, scale = read_pfm(f)

        # Open corresponding color image

        # Get all unique values

        # get indices of all unique indices

        # get pixel values from rgb image for all indices



if __name__ == '__main__':

    # parse command line paths to label data
    # load images
    # turn pixels into class ids
    #   for each datapoint
    #       load both color and material image
    #       get average color for masks in material image
    #       lookup class id for the color
    #       replace masks in material image with new class ids
    #
    # save as uint16 compressed npz

    pass

batch_size = 1
c, h, w = 1, 10, 10
nb_classes = 3
x = torch.randn(batch_size, c, h, w)
target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)

model = nn.Conv2d(c, nb_classes, 3, 1, 1)
criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, target)
loss.backward()