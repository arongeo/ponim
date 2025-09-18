
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim_size):
        super().__init__()

        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # Very interesting convolutional things going on to end up at 16384
        # We add paddings, so we can have emphasis on the corners as well
        # So our rows look like 64 pixels + 2 padding pixels
        # Then we have a 3x3 kernel first
        # That 3x3 kernel needs to go through the 66 pixels horizontally
        # It can't quite do that, cause in the end it'd be indexing into the 67th and 68th index
        # So we do 66 - 3, which is 63
        # At first we don't set a stride, because we want to go through every pixel
        # to be sure we find the ball, later we use stride 2, so we divide by 2
        # Finally we add 1, tbh I don't really get why, but we do
        # Do this for the height as well, also through all the layers
        # So we end up after the final one with 8x16
        # Multiply that by the output channels, so 8x16x128, and we have 16384

        # Two different linear layers to get to the mean and standard divergence
        self.encoder_linear_mean = nn.Linear(16384, latent_dim_size)
        self.encoder_linear_std_logvar = nn.Linear(16384, latent_dim_size)
        
    def encode(self, input_frame):
        output = self.flatten(self.encoder_conv_layers(input_frame))
        mean = self.encoder_linear_mean(output)
        std_logvar = self.encoder_linear_std_logvar(output)

        return mean, std_logvar
