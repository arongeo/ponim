
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

USER_INPUTS_SIZE = 2        # left paddle input, right paddle input
POSSIBLE_RESULTS_SIZE = 3   # 3 possible outcomes, game continues, left wins, right wins

class Cognition(nn.Module):
    def __init__(self, hidden_size, latent_dim_size, device, num_layers=1):
        super().__init__()

        self.hid_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.latent_dim_size = latent_dim_size

        #self.gru = nn.GRU(USER_INPUTS_SIZE + 2 * self.latent_dim_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=(0.3 if 1 < num_layers else 0.0))

        self.linear = nn.Sequential(
            nn.Linear(USER_INPUTS_SIZE + 5 * self.latent_dim_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.latent_dim_size)
        )

 
    def forward(self, inputs: torch.Tensor, prev_frames: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([inputs, prev_frames], dim=-1))

    def reset(self, batch_size: int):
        #self.h = torch.randn(self.num_layers, batch_size, self.hid_size).to(self.device) * 0.1
        self.h = torch.zeros(self.num_layers, batch_size, self.hid_size).to(self.device)


