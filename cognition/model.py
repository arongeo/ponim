
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

USER_INPUTS_SIZE = 2        # left paddle input, right paddle input
POSSIBLE_RESULTS_SIZE = 3   # 3 possible outcomes, game continues, left wins, right wins

class Cognition(nn.Module):
    def __init__(self, hidden_size, latent_dim_size, device):
        super().__init__()
        self.hid_size = hidden_size
        self.device = device
        #self.linear_lat_hid = nn.Sequential(
        #        nn.Linear(latent_dim_size, hidden_size),
        #        nn.Tanh()
        #)

        self.lstm = nn.LSTM(USER_INPUTS_SIZE, hidden_size, batch_first=True)

        self.linear_hid_lat = nn.Linear(hidden_size, latent_dim_size)
        #nn.init.normal_(self.linear_hid_lat.weight, mean=0, std=0.5)
        self.linear_hid_out = nn.Sequential(
                nn.Linear(hidden_size, POSSIBLE_RESULTS_SIZE),
                nn.LogSoftmax(dim=1)
        )

    def forward(self, inputs):
        o, self.hc = self.lstm(inputs, self.hc)
        return self.linear_hid_lat(o), self.linear_hid_out(o)

    def reset(self, batch_size):
        self.hc = (torch.randn(1, batch_size, self.hid_size).to(self.device) * 1.0, torch.zeros(1, batch_size, self.hid_size).to(self.device))

    @staticmethod
    def loss(expected_lat_frame, predicted_lat_frame, expected_res, predicted_res, alpha=1.0):
        lat_loss = F.mse_loss(predicted_lat_frame, expected_lat_frame)
        res_loss = F.cross_entropy(predicted_res, expected_res)

        return lat_loss

