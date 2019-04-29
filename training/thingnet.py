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

from networks.thingnet import ThingNet
from utils import transformations as tfs
from dataloaders.sceneflow import RDMODataset


logging.getLogger().setLevel(logging.INFO)


class ThingNetTrainer:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ThingNet(4, 10, False).to(self.device)
        self.optimiser = None
        self.loss = None
        self.scheduler = None
        self.batch_size = 1
        self.dataloader = self._setup_data()

        self.val_freq = 100

    def _setup_data(self):
        """Set-up transformations, sampler and returns Dataloader object"""
        dset = RDMODataset("/home/noorvir/Documents/data/SceneFlow/thing_net.hdf5")
        stats = dset.dataset_stats

        # 1. Transformations
        rgb_transformations = [tfs.gaussian_blur(), tfs.random_noise(),
                               tfs.normalise(mean=stats['rgb'].mean, std_dev=stats['rgb'].std_dev)]
        depth_transformations = [tfs.gaussian_blur(), tfs.random_noise(),
                                 tfs.normalise(mean=stats['rgb'].mean, std_dev=stats['rgb'].std_dev)]
        co_transformations = [tfs.flip_horizontal(), tfs.flip_vertical()]

        dset.setup_transformations(rgb_transformations, depth_transformations, co_transformations)

        # 2. Sampler
        train_sampler = SubsetRandomSampler(dset.train_idx)
        val_sampler = SubsetRandomSampler(dset.val_idx)
        dataloader = {'train': DataLoader(dset, batch_size=self.batch_size, sampler=train_sampler,
                                          num_workers=6),
                      'val': DataLoader(dset, batch_size=self.batch_size, sampler=val_sampler,
                                        num_workers=6)}
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

    def infer(self, inputs):
        self.model.eval()
        return self.model(inputs)

    def visualise(self, num=None):
        if num is None:
            num = self.batch_size
        elif num >= self.batch_size:
            num = self.batch_size


        # Visualise model predictions
        pass


if __name__ == "__main__":
    trainer = ThingNetTrainer()
    trainer.train()
# TODO Next
# Allocate variables/Tensors onto the correct device
# Setup number of inputs as a variable and load network weights accordingly
# Pass the inputs through the model and make sure it runs
# Set-up correspondence finder
# compute loss using the output of the correspondence finder
# Update weights
# Implement config parser
# Configure model and training from config file