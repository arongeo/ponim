
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
        #self.linear_lat_hid = nn.Sequential(
        #        nn.Linear(latent_dim_size, hidden_size),
        #        nn.Tanh()
        #)

        self.lstm = nn.LSTM(USER_INPUTS_SIZE + latent_dim_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=(0.2 if 1 < num_layers else 0.0))
        #self.gru = nn.(USER_INPUTS_SIZE + latent_dim_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=(0.2 if 1 < num_layers else 0.0))

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.4)

        self.linear_hid_lat_mean = nn.Linear(hidden_size, latent_dim_size)
        self.linear_hid_lat_logvar = nn.Linear(hidden_size, latent_dim_size)
        #nn.init.normal_(self.linear_hid_lat.weight, mean=0, std=0.5)
        #self.linear_hid_out = nn.Sequential(
        #        nn.Linear(hidden_size, POSSIBLE_RESULTS_SIZE),
        #        nn.Dropout(0.3),
        #        nn.Softmax(dim=1)
        #)

    def forward(self, inputs, prev_lat_mean):
        o, self.hc = self.lstm(torch.cat([inputs, prev_lat_mean], dim=-1), self.hc)
        #o, self.hidden = self.gru(torch.cat([inputs, prev_lat_frames], dim=-1), self.hidden)
        o = self.layer_norm(o)
        o = self.dropout(o)
        return self.linear_hid_lat_mean(o), self.linear_hid_lat_logvar(o)

    def reset(self, batch_size):
        self.hc = (torch.randn(self.num_layers, batch_size, self.hid_size).to(self.device) * 0.5, torch.zeros(self.num_layers, batch_size, self.hid_size).to(self.device))
        #self.hidden = torch.randn(self.num_layers, batch_size, self.hid_size).to(self.device) * 0.1

    @staticmethod
    def loss(expected_lat_mean, predicted_lat_mean, predicted_logvar):#, expected_res, predicted_res, alpha=1.0):
        #lat_loss = F.mse_loss(predicted_lat_frame, expected_lat_frame)
        #res_loss = F.cross_entropy(predicted_res, expected_res)

        var = torch.exp(predicted_logvar) + 1e-6
        loss = F.gaussian_nll_loss(predicted_lat_mean, expected_lat_mean, var)

        return loss

