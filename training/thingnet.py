"""

data_transforms = transforms.Compose([
    rescale,
    random_flip,
    normalise
])

"""
import torch
import torchvision
import logging
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import transformations as tfs
from dataloaders.sceneflow import RDMODataset


logging.getLogger().setLevel(logging.INFO)


class ThingNetTrainer:

    def __init__(self):
        self.model = None
        self.optimiser = None
        self.loss = None
        self.scheduler = None
        self.batch_size = 10
        self.dataloader = self._setup_data()

        self.device = 'GPU'
        self.val_freq = 100

    def _setup_data(self):
        """Set-up transformations, sampler and returns Dataloader object"""
        dset = RDMODataset("")

        # 1. Transformations
        rgb_transformations = [tfs.gaussian_blur(), tfs.random_noise(),
                               tfs.normalise(mean=dset['rgb'].mean, std_dev=dset['rgb'].std_dev)]
        depth_transformations = [tfs.gaussian_blur(), tfs.random_noise()
                                 tfs.normalise(mean=dset['rgb'].mean, std_dev=dset['rgb'].std_dev)]
        co_transformations = [tfs.flip_horizontal(), tfs.flip_vertical()]

        dset.setup_transformations(rgb_transformations, depth_transformations, co_transformations)

        # 2. Sampler
        train_sampler = SubsetRandomSampler(dset.train_idx)
        val_sampler = SubsetRandomSampler(dset.val_idx)
        dataloader = {'train': torch.utils.data.Dataloader(dset, batch_size=self.batch_size,
                                                           sampler=train_sampler, num_workers=6),
                      'val': torch.utils.data.Dataloader(dset, batch_size=self.batch_size,
                                                         sampler=val_sampler, num_workers=6)}
        return dataloader

    def get_loss(self, output, mask_batch, obj_batch):

        # Find correspondences
        # evaluate triplet loss at correspondences
        loss = torch.Tensor([0.0], requires_grad=True)
        return loss

    def train(self):

        step = 0

        for epoch in range(100):
            logging.info("Epoch %d of %d" % (epoch, 100))

            phase = 'val' if step % self.val_freq == 0 else 'train'

            if phase == 'train':
                self.scheduler.step()
                self.model.train()
            else:
                self.model.eval()

            for data in self.dataloader[phase]:
                inputs = data[:2]
                masks = data[2:]

                inputs.to(self.device)
                masks.to(self.device)

                self.optimiser.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = self.model(inputs)
                    loss = self.get_loss(output, masks[0], masks[1])

                    if phase == 'train':
                        loss.backward()
                        self.optimiser.step()

                # TODO: logging
                # TODO: save stats
                # TODO: plotting
                # TODO: saving model checkpoints

    def visualise(self):
        # Visualise model predictions
        pass


