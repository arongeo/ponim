
import torch
from torch import nn
from torch.nn import functional as F
import cognition
import vision

class Ponim(nn.Module):
    def __init__(self, latent_size, cog_hidden_size, device, vae=None):
        super().__init__()

        self.latent_size = latent_size
        self.device = device
        self.cog_hidden_size = cog_hidden_size

        self.vae = vision.VAE(self.latent_size) if vae is None else vae

        for p in self.vae.parameters():
            p.requires_grad = False

        self.cog = cognition.Cognition(self.cog_hidden_size, self.latent_size, self.device)

    @staticmethod
    def loss(pred_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
        bs, ss, h, w = pred_frame.shape
        return F.binary_cross_entropy(pred_frame, target_frame, reduction='sum').mean()
