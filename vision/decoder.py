
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim_size):
        super().__init__()

        # We don't take in a distribution, we take in an actual point/latent vector
        self.decoder_linear = nn.Linear(latent_dim_size, 16384)

        # We essentially just flip the encoder's convolution layers
        # to do a deconvolution

        self.decoder_deconv_layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Sigmoid at the end, because we need to squash the numbers between 0 and 1,
        # since the resulting numbers should be pixel intensity values

    def decode(self, z):
        output = self.decoder_linear(z)
        output = output.view(-1, 128, 8, 16)
        output = self.decoder_deconv_layers(output)
        return output
