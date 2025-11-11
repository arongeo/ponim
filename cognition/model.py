
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

        self.quantizer = nn.Embedding.from_pretrained(vqvae.quantizer.weight)

        self.input_embeddings = nn.Embedding(3, self.hidden_size)

        self.linear_emb_hid = nn.Linear(self.quantizer.embedding_dim * self.latent_dim_size, self.hidden_size)

        self.gru = nn.GRU(
            3 * hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=(0.3 if 1 < num_layers else 0.0)
        )

        self.linear_hid_token = nn.Linear(hidden_size, self.codebook_size * self.latent_dim_size)
        
        
    def forward(self, inputs: torch.Tensor, prev_frame_tokens: torch.Tensor, training=False):
        bs, ss, _ = prev_frame_tokens.shape

        embeddings = self.quantizer(prev_frame_tokens)
        input_embs = self.input_embeddings(inputs.int() + 1).view(bs, ss, -1)
        emb_hid = self.linear_emb_hid(embeddings.flatten(2))

        o, self.h = self.gru(torch.cat([input_embs, emb_hid], dim=-1), self.h)
        o = self.linear_hid_token(o)
        o = o.view(bs, ss, self.latent_dim_size, self.codebook_size)

        if training:
            return o
        else:
            return torch.argmax(o, dim=-1)

    def reset(self, batch_size: int):
        self.h = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)

    @staticmethod
    def loss(generated_tokens, original_tokens):
        return F.cross_entropy(
            generated_tokens.view(-1, generated_tokens.shape[-1]),
            original_tokens.view(-1)
        )
