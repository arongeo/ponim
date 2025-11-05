
from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # We essentially just flip the encoder's convolution layers
        # to do a deconvolution

        self.decoder_deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Sigmoid at the end, because we need to squash the numbers between 0 and 1,
        # since the resulting numbers should be pixel intensity values

    def decode(self, z):
        return self.decoder_deconv_layers(z)
