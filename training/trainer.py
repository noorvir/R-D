import torch
import torchvision

import numpy as np

__all__ = ['BaseTrainer']


class BaseTrainer:
    def __init__(self):
        pass

    def get_loss(self):
        pass

    def get_optimiser(self):
        pass

    def log(self):
        pass

    def plot(self):
        pass
