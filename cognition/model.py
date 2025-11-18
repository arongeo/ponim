
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from vision.vqvae import VQVAE

USER_INPUTS_SIZE = 2        # left paddle input, right paddle input
POSSIBLE_RESULTS_SIZE = 3   # 3 possible outcomes, game continues, left wins, right wins

class Cognition(nn.Module):
    def __init__(self, hidden_size: int, vqvae: VQVAE, device, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.latent_dim_size = vqvae.latent_dim_size
        self.codebook_size = vqvae.codebook_size

        self.quantizer = vqvae.quantizer

        self.input_embeddings = nn.Embedding(3, 4)

        self.linear_emb_hid = nn.Linear(self.quantizer.embedding_dim * self.latent_dim_size, self.hidden_size)

        self.gru = nn.GRU(
            2 * 4,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=(0.3 if 1 < num_layers else 0.0)
        )

        self.dropout = nn.Dropout(0.2)

        self.linear_hid_token = nn.Linear(hidden_size, self.codebook_size * self.latent_dim_size)
        
        
    def forward(self, inputs: torch.Tensor, next_frame_tokens: torch.Tensor, h: torch.Tensor, training=False, temperature=0.8):
        bs, ss, _ = inputs.shape

        input_embs = self.input_embeddings(inputs.int() + 1).view(bs, ss, -1)

        o, h = self.gru(self.dropout(input_embs), h)
        o = self.linear_hid_token(self.dropout(o))
        o = o.view(bs, ss, self.latent_dim_size, self.codebook_size)

        if training:
            return o, h
        else:
            #prob = torch.softmax(o / temperature, dim=-1)
            #return torch.multinomial(prob.view(-1, prob.shape[-1]), 1).view(o.shape[:-1])
            return torch.argmax(o, dim=-1), h

    @staticmethod
    def loss(generated_tokens, original_tokens, gru_o, emb_hid_o, beta=0.3):
        return F.cross_entropy(
            generated_tokens.view(-1, generated_tokens.shape[-1]),
            original_tokens.view(-1)
        ) + F.mse_loss(gru_o, emb_hid_o) * beta
