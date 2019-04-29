import torch
import torch.nn as nn

from time import time
from utils.logger import get_logger
from utils.nn.network_tools import load_model

logger = get_logger()

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, channels=3, norm_layer=nn.BatchNorm2d,
                 bn_eps=1e-5, bn_momentum=0.1, deep_stem=False, stem_width=32,
                 inplace=True):
        """

        Parameters
        ----------
        block
        layers
        norm_layer
        bn_eps
        bn_momentum
        deep_stem
        stem_width
        inplace
        """
        super(ResNet, self).__init__()

        self.inplanes = stem_width * 2 if deep_stem else 64

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),)

        layers = [block(self.inplanes, planes, stride, norm_layer, bn_eps,
                        bn_momentum, downsample, inplace)]

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)

        return blocks


def adapt_weights(model_conv, weights_conv):

    m_shape = model_conv.shape
    w_shape = weights_conv.shape

    diff = w_shape[1] - m_shape[1]

    if diff < 0:
        new_wts = weights_conv[:, :1, :, :].repeat(1, abs(diff), 1, 1)
        wts = torch.cat((weights_conv, new_wts), dim=1)
    else:
        wts = weights_conv[:, :-diff, :, :]

    return wts


def resnet(num_layers=18, channels=3, pretrained_model=None, **kwargs):
    """

    Parameters
    ----------
    num_layers
    channels
    pretrained_model
    kwargs

    Returns
    -------

    """
    if num_layers == 18:
        layers = [2, 2, 2, 2]
    elif num_layers == 34:
        layers = [3, 4, 6, 3]
    elif num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]
    elif num_layers == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Incorrect number of layers specified for Resnet")

    model = ResNet(BasicBlock, layers, channels, **kwargs)

    if pretrained_model is None:
        return model

    # Adapt first conv-layer shape
    if isinstance(pretrained_model, str):
        logger.info("Loading model weights from %s." % pretrained_model)

        state_dict = torch.load(pretrained_model)

        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = pretrained_model

    conv_name = list(model.state_dict().keys())[0]      # dw it's an ordered dict

    if conv_name not in state_dict.keys():
        logger.warning("Can't load weights! Key '%s' not found. This could be due"
                       "to a architecture mismatch in the model and the weights "
                       "file. NOT LOADING WEIGHTS!!" % conv_name)
        return model

    model_conv = model.state_dict()[conv_name]
    pt_conv = state_dict[conv_name]

    if model_conv.shape != pt_conv.shape:

        logger.warning("Shape mismatch between model and weights file. Duplicating "
                       "weights in layer '%s' to match model shape %s " %
                       (conv_name, (model_conv.shape,)))

        temp_weight = adapt_weights(model_conv, pt_conv)
        state_dict[conv_name] = temp_weight

    return load_model(model, state_dict)


def resnet18(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet34(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet50(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet152(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
