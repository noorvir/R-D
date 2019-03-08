import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import transformations

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ThingNetDataset(Dataset):

    def __init__(self):

        # Create an index of the entire dataset
        #   Visit every folder and add its path to list
        #   When accessing the item (through __getitem__) index into this list
        pass

    def __len__(self):
        return 10

    def __getitem__(self, item):
        return np.random.rand(1)[0]




dataset = ThingNetDataset()
print(len(dataset))

for i in range(len(dataset)):
    sample = dataset[i]
    print(sample)

transforms.Compose([
    transformations.gradient(adaptation='hsv'),
    transforms.ToTensor()
])
