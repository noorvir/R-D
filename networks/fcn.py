import torch
import torch.nn as nn
import torch.nn.functional as F


class config:
    __dict__ = {}


config.bn_eps = 100
config.bn_momentum = 0.5


def get_padding(i, k, s, ptype='same'):

    if type(i) == tuple or type(i) == list:
        py = int(0.5 * (i[0] * (s - 1) + k - s))
        px = int(0.5 * (i[1] * (s - 1) + k - s))
    else:
        px = py = int(0.5 * (i * (s - 1) + k - s))

    return py, px


class FCN(nn.Module):
    def __init__(self, input_shape, descriptor_dim,
                 norm_layer=nn.BatchNorm2d):
        """

        Parameters
        ----------
        input_shape
        descriptor_dim
        norm_layer
        """
        super().__init__()
        self._channels = input_shape[1]
        self.descriptor_dim = descriptor_dim

        ips = input_shape[2:]
        self.conv1 = nn.Conv2d(self._channels, 16, 5, padding=get_padding(ips, 5, 1))
        self.conv2 = nn.Conv2d(16, 64, 7, padding=get_padding(ips, 7, 1))
        self.conv3 = nn.Conv2d(64, 32, 5, padding=get_padding(ips, 5, 1))
        self.conv4 = nn.Conv2d(32, 32, 3, padding=get_padding(ips, 3, 1))
        self.conv5 = nn.Conv2d(32, self.descriptor_dim, 1, padding=get_padding(ips, 1, 1))

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        return x


if __name__ == "__main__":
    x_in_shape = (1, 3, 500, 500)
    x_in = torch.rand(x_in_shape)
    fcn = FCN(x_in_shape, 10)
    print(fcn)

    print(fcn(x_in).shape)
