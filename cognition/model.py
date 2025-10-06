
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

        self.lstm = nn.LSTM(USER_INPUTS_SIZE, hidden_size, batch_first=True, num_layers=num_layers, dropout=(0.3 if 1 < num_layers else 0.0))

        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)

        #self.linear_hid_lat = nn.Linear(hidden_size, latent_dim_size)

        self.linear_hid_mix = nn.Linear(hidden_size, self.mixtures * (1 + 2 * self.latent_dim_size))
 
    def forward(self, inputs: torch.Tensor):
        o, self.hc = self.lstm(inputs, self.hc)
        o = self.dropout(o)
        #return self.linear_hid_lat(o)
        o = self.linear_hid_mix(o)

        mco, mu, stdev = torch.split(o, [
            self.mixtures,
            self.mixtures * self.latent_dim_size,
            self.mixtures * self.latent_dim_size
        ], dim=-1)

        mu = mu.view(mu.shape[0], mu.shape[1], self.mixtures, self.latent_dim_size)
        stdev = torch.exp(0.5 * mu.view(stdev.shape[0], stdev.shape[1], self.mixtures, self.latent_dim_size))
        mco = mco.unsqueeze(2)

        return mco, mu, stdev

    def reset(self, batch_size: int):
        self.hc = (torch.zeros(self.num_layers, batch_size, self.hid_size).to(self.device), torch.zeros(self.num_layers, batch_size, self.hid_size).to(self.device))

    @staticmethod
    def loss(expected_mu: torch.Tensor, mco: torch.Tensor, mu: torch.Tensor, stdev: torch.Tensor):
        expected_mu = expected_mu.unsqueeze(2)

        mix_log_prob = Normal(mu, stdev).log_prob(expected_mu)
        sum_mix_log_prob = mix_log_prob.sum(dim=-1)

        logp_mco = F.log_softmax(mco, dim=-1)

        return -1.0 * torch.logsumexp(sum_mix_log_prob + logp_mco, dim=-1).mean()

        #lat_loss = F.mse_loss(predicted_lat_frame, expected_lat_frame)

        #return lat_loss

