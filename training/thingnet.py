"""

data_transforms = transforms.Compose([
    rescale,
    random_flip,
    normalise
])






"""
import numpy as np

from training import BaseTrainer


class Compose:
    def __init__(self):
        pass

    def __call__(self, inputs, targets):

        for image in inputs:
            pass

        for image in targets:
            pass

        return inputs, targets


class ThingNetTrainer(BaseTrainer):

        pass

