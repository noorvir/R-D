"""

data_transforms = transforms.Compose([
    rescale,
    random_flip,
    normalise
])

"""
import torch
import torch.nn as nn
import torchvision
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks.thingnet import ThingNet
from networks.fcn import FCN
from tools import transformations as tfs
from tools.structures import DataTypes

from dataloaders.correspondence import find_correspondences
from dataloaders.sceneflow import RDMODataset


logging.getLogger().setLevel(logging.INFO)


def clustering_loss(matches, non_matches):
    """

    Parameters
    ----------
    matches:
    non_matches:

    Returns
    -------

    """


class ThingNetTrainer:

    def __init__(self, device=None):

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
            self.dtypes = DataTypes('cuda') if torch.cuda.is_available() else DataTypes('cpu')
        else:
            self.device = torch.device(device)
            self.dtypes = DataTypes(device)

        # self.model = ThingNet(4, 10, False)#.to(self.device)
        self.batch_size = 10
        self.model = FCN((self.batch_size, 4, 540, 960), 10).to(self.device)
        self.optimiser = self.get_optimiser(self.model.parameters(), 0.0003, 1.0e-4)
        self.loss = None
        self.scheduler = None
        self.dataloader = self._setup_data()
        self._data_iter = {'train': iter(self.dataloader['train']),
                           'val': iter(self.dataloader['val'])}

        self.val_freq = 10

    def _setup_data(self):
        """Set-up transformations, sampler and returns Dataloader object"""
        dset = RDMODataset("/home/noorvir/Documents/data/SceneFlow/thing_net.hdf5")
        stats = dset.dataset_stats

        # 1. Transformations
        rgb_transformations = [tfs.gaussian_blur(), tfs.random_noise(),
                               tfs.normalise(mean=stats['rgb'].mean, std_dev=stats['rgb'].std_dev)]
        depth_transformations = [tfs.gaussian_blur(), tfs.random_noise(),
                                 tfs.normalise(mean=stats['depth'].mean, std_dev=stats['depth'].std_dev)]

        co_transformations = [tfs.flip_horizontal(), tfs.flip_vertical(), tfs.NHWC_to_NCHW(),
                              tfs.type_converter(dtype=torch.float32)]

        dset.setup_transformations(rgb_transformations, depth_transformations, co_transformations)

        # 2. Sampler
        train_sampler = SubsetRandomSampler(dset.train_idx)
        val_sampler = SubsetRandomSampler(dset.val_idx)
        dataloader = {'train': DataLoader(dset, batch_size=self.batch_size, sampler=train_sampler,
                                          num_workers=6),
                      'val': DataLoader(dset, batch_size=self.batch_size, sampler=val_sampler,
                                        num_workers=6)}
        return dataloader

    def _get_next_batch(self, phase):
        try:
            inputs, masks = next(self._data_iter[phase])
        except StopIteration:
            logging.debug("Resetting iterator for phase %s" % phase)
            self._data_iter[phase] = iter(self.dataloader[phase])
            inputs, masks = next(self._data_iter[phase]
                                 )
        inputs = [ip.type(self.dtypes.float) for ip in inputs]
        masks = [msk.type(self.dtypes.float) for msk in masks]

        # Expand dims to make inputs stackable
        inputs = [ip.unsqueeze(dim=1) if len(ip.shape) == 3 else ip for ip in inputs]
        inputs = torch.cat(inputs, dim=1)
        return inputs, masks

    def get_loss(self, output, mat_mask_batch, obj_mask_batch):

        # Find correspondences
        # evaluate triplet loss at correspondences

        total_loss = 0

        for mat_mask, obj_mask in zip(mat_mask_batch, obj_mask_batch):
            # compute loss over element from batch
            # add to total loss

            clist = find_correspondences(mat_mask, obj_mask, self.dtypes)

            variances_list = []
            means_list = []
            similar_non_matches_loss_list = []

            for material in clist:
                # get matches
                match_idx = material[0]
                non_match_idx = material[1]
                obj_match_idx = material[2]

                matches = output[:, :, match_idx[:, 0], match_idx[:, 1]]
                non_matches = output[:, :, non_match_idx[:, 0], non_match_idx[:, 1]]
                obj_matches = output[:, :, obj_match_idx[:, 0], obj_match_idx[:, 1]]

                variances_list.append(torch.var(matches))
                means_list.append(torch.mean(matches))

                # Compute pixel intensity (dis)similarity
                mshape = matches.shape # (1, 5, 121)
                nmshape = non_matches.shape
                rand_idx = torch.randint(0, nmshape[-1], (mshape[-1],)).type(self.dtypes.long)
                similar_non_matches = non_matches[:, :, rand_idx]
                similar_non_matches_loss = torch.mean((matches - similar_non_matches).pow(2))
                similar_non_matches_loss_list.append(similar_non_matches_loss)

            variances_tensor = torch.tensor(variances_list).type(self.dtypes.float)
            similarity_tensor = torch.tensor(similar_non_matches_loss_list).type(self.dtypes.float)

            mean_loss = 0
            var_loss = torch.mean(variances_tensor)
            similarity_loss = torch.mean(similarity_tensor)

            # Compute loss b/w each mean and all the rest.
            for i in range(len(means_list)):
                means_tensor = torch.tensor(means_list[:i] + means_list[i + 1:]).type(self.dtypes.float)
                mean_loss += torch.mean((means_list[i] - means_tensor).pow(2))

            total_loss += (var_loss + 1/similarity_loss + 1/mean_loss)

        return total_loss

    def get_optimiser(self, params, lr, weight_decay):
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def train(self):

        for epoch in range(100):
            logging.info("\n***** Epoch %d of %d *****" % (epoch + 1, 100))

            for step in tqdm(range(len(self.dataloader['train']))):

                # -----------
                # 1. Train
                # -----------
                # self.scheduler.step()
                self.model.train()
                self.optimiser.zero_grad()

                inputs, masks = self._get_next_batch('train')

                with torch.set_grad_enabled(True):
                    output = self.model(inputs).type(self.dtypes.float)
                    loss = self.get_loss(output, masks[0], masks[1]).type(self.dtypes.float)
                    loss.backward()
                    self.optimiser.step()

                # --------------
                # 2. Evaluate
                # --------------
                if step % self.val_freq == 0:
                    self.model.eval()
                    inputs, masks = self._get_next_batch('val')

                    with torch.set_grad_enabled(False):
                        output = self.model(inputs)
                        vloss = self.get_loss(output, masks[0], masks[1])

                    logging.info("\n\nVal loss: %0.5f, \t Train loss: %0.5f\n" %
                                 (vloss.item(), loss.item()))

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
    trainer = ThingNetTrainer(device='cuda')
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
# Change depth to disparity ( easier to deal with)
