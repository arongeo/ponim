import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from vision.encoder import Encoder
from vision.decoder import Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim_size):
        super().__init__()
        self.latent_dim_size = latent_dim_size

        # We split the VAE, since, for our purpose,
        # we'll only need the decoder once training
        # is done.
        self.encoder = Encoder(latent_dim_size)
        self.decoder = Decoder(latent_dim_size)

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

    def forward(self, input_frame):
        mu, logvar = self.encoder.encode(input_frame)
        z = self.reparameterize(mu, logvar)
        return self.decoder.decode(z), mu, logvar

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
        #
        # once we have the difference between the pixels, we
        # sum up all the errors in a frame, and then take the mean
        # of all frames' errors in a batch
        reconstruction_loss = F.binary_cross_entropy(reconstruction, original, reduction='none').sum(dim=(1, 2, 3)).mean()

        # KL-loss is a bit trickier
        # We're trying to punish it, if it doesn't conform to
        # the normal distribution, aka we want the mean to be close to 0
        # and the variance (and thus the standard deviation) to be close to 1,
        # the more we differ from those, the higher the loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # We use beta to control how much we want to conform to the normal
        # distribution, it's a tradeoff with accurate reconstruction
        return reconstruction_loss + kl_loss * beta
