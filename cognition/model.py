
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

USER_INPUTS_SIZE = 2        # left paddle input, right paddle input
POSSIBLE_RESULTS_SIZE = 3   # 3 possible outcomes, game continues, left wins, right wins

class Cognition(nn.Module):
    def __init__(self, hidden_size, latent_dim_size, device, num_layers=1):
        super().__init__()
        self.hid_size = hidden_size
        self.device = device
        self.num_layers = num_layers

        self.lstm = nn.LSTM(USER_INPUTS_SIZE, hidden_size, batch_first=True, num_layers=1)

        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear_hid_lat = nn.Linear(hidden_size, latent_dim_size)
    
    def forward(self, inputs):
        o, self.hc = self.lstm(inputs, self.hc)
        o = self.layer_norm(o)
        return self.linear_hid_lat(o)

    def reset(self, batch_size):
        self.hc = (torch.randn(self.num_layers, batch_size, self.hid_size).to(self.device) * 0.1, torch.zeros(self.num_layers, batch_size, self.hid_size).to(self.device))

    @staticmethod
    def loss(expected_lat_frame, predicted_lat_frame):
        lat_loss = F.mse_loss(predicted_lat_frame, expected_lat_frame)

        return lat_loss

