import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .custom_layers import GaussianNoiseVariant


class BaseModelAutoEncoder(nn.Module):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelAutoEncoder, self).__init__()

        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.encoder_conv = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 13x13x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 27x27x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 55x55x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 111x111x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),  # 224x224x3
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.encode(x)
        decoded = self.decode(encoded).view(input_shape)
        return encoded, decoded


class BaseModelVAE(BaseModelAutoEncoder):
    """
    Base Class for VAE family
    """

    def __init__(self):
        super(BaseModelVAE, self).__init__()

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (th.Tensor)
        :param logvar: (th.Tensor)
        """
        if self.training:
            # logvar = \log(\sigma^2) = 2 * \log(\sigma)
            # \sigma = \exp(0.5 * logvar)
            std = logvar.mul(0.5).exp_()
            # Sample \epsilon from normal distribution
            # use std to create a new tensor, so we don't have to care
            # about running on GPU or not
            eps = std.new(std.size()).normal_()
            # Then multiply with the standard deviation and add the mean
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z).view(input_shape)
        return decoded, mu, logvar


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
