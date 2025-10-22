
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

USER_INPUTS_SIZE = 2        # left paddle input, right paddle input
POSSIBLE_RESULTS_SIZE = 3   # 3 possible outcomes, game continues, left wins, right wins

class Cognition(nn.Module):
    def __init__(self, hidden_size, mixtures, latent_dim_size, device, num_layers=1):
        super().__init__()
        self.hid_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.mixtures = mixtures
        self.latent_dim_size = latent_dim_size

        self.gru = nn.GRU(USER_INPUTS_SIZE + 5 * latent_dim_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=(0.3 if 1 < num_layers else 0.0))

        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear_hid_lat = nn.Linear(hidden_size, latent_dim_size)

 
    def forward(self, inputs: torch.Tensor, prev_lat_frame: torch.Tensor):
        o, self.hc = self.gru(torch.cat([inputs, prev_lat_frame], dim=-1), self.hc)
        o = self.dropout(o)
        return self.linear_hid_lat(o)

    def reset(self, batch_size: int):
        self.hc = torch.zeros(self.num_layers, batch_size, self.hid_size).to(self.device)

    @staticmethod
    def loss(z: torch.Tensor, target_z: torch.Tensor):
        lat_loss = F.mse_loss(z, target_z)
        return lat_loss

