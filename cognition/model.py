
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class Cognition(nn.Module):
    def __init__(self, latent_dim_size, hidden_size):
        super().__init__()
