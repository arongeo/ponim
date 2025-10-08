
import torch
from torch import nn
from torch.nn import functional as F
import cognition
import vision

class Ponim(nn.Module):
    def __init__(self, latent_size, cog_hidden_size, device):
        self.latent_size = latent_size
        self.device = device
        self.cog_hidden_size = cog_hidden_size

        self.vae = vision.VAE(self.latent_size)
        self.cog = cognition.Cognition(self.cog_hidden_size, self.latent_size, self.device)

    def __init__(self, vae: vision.vae.VAE, cog_hidden_size, device):
        self.latent_size = vae.latent_dim_size
        self.device = device
        self.cog_hidden_size = cog_hidden_size

        self.vae = vae
        self.cog = cognition.Cognition(self.cog_hidden_size, self.latent_size, self.device)

    @staticmethod
    def loss(pred_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(pred_frame, target_frame, reduction='none').sum(dim=(1, 2, 3)).mean()