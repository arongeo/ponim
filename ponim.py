import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import math
import random

class VAE(nn.Module):
    def __init__(self, latent_dimension) -> None:
        super().__init__()

        # ENCODER
        
        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
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
        self.encoder_linear_mean = nn.Linear(16384, latent_dimension)
        self.encoder_linear_std_logvar = nn.Linear(16384, latent_dimension)

        # DECODER

        # We don't take in a distribution, we take in an actual point/latent vector
        self.decoder_linear = nn.Linear(latent_dimension, 16384)

        # We essentially just flip the encoder's convolution layers
        # to do a deconvolution

        self.decoder_deconv_layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Sigmoid at the end, because we need to squash the numbers between 0 and 1,
        # since the resulting numbers should be pixel intensity values

    def encode(self, input_frame):
        output = self.flatten(self.encoder_conv_layers(input_frame))
        mean = self.encoder_linear_mean(output)
        std_logvar = self.encoder_linear_std_logvar(output)

        return mean, std_logvar

    # Reparameterize or sample, aka get a valid latent vector
    # from the returned latent space
    def reparameterize(self, mean, std_logvar):
        # note to self: what is logvar and why to use it?
        # logvar is used for a very simple reason:
        # neural networks return whatever number their heart pleases,
        # it can be negative or positive, or whatever they want.
        # The standard deviation shouldn't be negative though.
        # So we say the NN's returned value isn't the standard deviation
        # it's ln(std^2), so then we can simply do e^(0.5 * ln(std^2)), which
        # returns us the standard deviation, which is never negative

        # We use a random epsilon, to get a random point in the latent space

        std = torch.exp(0.5 * std_logvar) 
        eps = torch.randn_like(std) # Just gets a random point in the normal distribution

        # Works, and can backpropagate through this cause in calculus:
        # f(x) = c * g(x)
        # f'(x) = c * g'(x)
        # Aka, constant multiplication stays, and the gradient for f'(x)
        # in relation to g'(x) is always c so no matter what c is, it's c,
        # thus we can ignore it (actually we can't, cause of what I'm saying later,
        # but the random epsilons all around should mostly cancel out), during backprop,
        # aka calculating gradients
        # Since the gradient of std is eps, aka the change in the loss in relation 
        # to std is eps, we might say we still account of randomness, and we kinda do, 
        # but the reason why we do this, (I've been pondering about this for a while) isn't 
        # actually to help make the reconstruction better, if we'd only care about the 
        # reconstruction, we'd just train a normal autoencoder, instead we do it to align our
        # learned distibutions' mean and std to a N(0, 1) normal distribution

        return mean + std * eps

    def decode(self, z):
        output = self.decoder_linear(z)
        output = output.view(-1, 128, 8, 16)
        output = self.decoder_deconv_layers(output)
        return output

    def forward(self, input_frame):
        mu, logvar = self.encode(input_frame)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(original, reconstruction, mu, logvar, beta=1.0):
        # binary cross entropy is quite simple
        # take the inputs and the outputs
        # put it into this formula, where y is the original
        # p is the prediction
        # - (y * log(p) + (1 - y) * log(1 - p))
        # closer to 0 the better,
        # we're just comparing inputs and outputs,
        # essentially a more advanced mean squared error
        reconstruction_loss = F.binary_cross_entropy(reconstruction, original, reduction='sum')

        # KL-loss is a bit trickier
        # We're trying to punish it, if it doesn't conform to
        # the normal distribution, aka we want the mean to be close to 0
        # and the variance (and thus the standard deviation) to be close to 1,
        # the more we differ from those, the higher the loss
        #kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # We use beta to control how much we want to conform to the normal
        # distribution, it's a tradeoff with accurate reconstruction
        return reconstruction_loss + kl_loss * beta
