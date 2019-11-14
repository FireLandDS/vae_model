from .models import *


class CNNVAE(BaseModelVAE):
    """
    Custom convolutional VAE network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3):
        super(CNNVAE, self).__init__()
        self.encoder_fc1 = nn.Linear(6 * 6 * 64, state_dim)
        self.encoder_fc2 = nn.Linear(6 * 6 * 64, state_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, 6 * 6 * 64)
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

    def decode(self, z):
        """
        :param z: (th.Tensor)
        :return: (th.Tensor)
        """
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 64, 6, 6)
        return self.decoder_conv(z)
