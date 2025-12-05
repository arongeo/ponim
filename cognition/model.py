
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from cognition.lse.model import LSE
from vision.vqvae import VQVAE

USER_INPUTS_SIZE = 2        # left paddle input, right paddle input
POSSIBLE_RESULTS_SIZE = 3   # 3 possible outcomes, game continues, left wins, right wins

class Cognition(nn.Module):
    def __init__(self, lse: LSE, device):
        super().__init__()

        self.hidden_size = int(lse.hidden_size * (4.0/3.0))
        self.device = device
        self.latent_dim_size = lse.vqvae.latent_dim_size
        self.codebook_size = lse.vqvae.codebook_size
        self.lse = lse

        for param in self.lse.parameters():
            param.requires_grad = False

        self.quantizer = self.lse.vqvae.quantizer

        self.input_embeddings = nn.Embedding(3, 4)

        self.gru = nn.GRUCell(
            2 * 4,
            self.hidden_size
        )

        self.dropout = nn.Dropout(0.2)

        self.linear_hid_lse = nn.Linear(self.hidden_size, self.lse.hidden_size)
        
        
    def forward(self, inputs: torch.Tensor, prev_tokens: torch.Tensor, h_in: torch.Tensor, training=False, temperature=0.8) -> tuple[torch.Tensor, torch.Tensor]:
        bs, _ = inputs.shape

        if training:
            with torch.no_grad():
                embs = self.quantizer(prev_tokens)
            
            hid = self.lse.linear_emb_hid(embs.flatten(1))
            h = torch.cat([hid, h_in[:, hid.shape[1]:]], dim=-1)
        else:
            h = h_in

        input_embs = self.input_embeddings(inputs.int() + 1).view(bs, -1)

        h_out = self.gru(input_embs, self.dropout(h))

        o = self.lse.linear_hid_emb(self.dropout(h_out[:, :self.lse.hidden_size]))
        o = o.view(bs, self.latent_dim_size, self.codebook_size)

        if training:
            return o, h_out
        else:
            #prob = torch.softmax(o / temperature, dim=-1)
            #return torch.multinomial(prob.view(-1, prob.shape[-1]), 1).view(o.shape[:-1])
            return torch.argmax(o, dim=-1), h_out

    '''
    @staticmethod
    def loss(generated_tokens, original_tokens, hs, phs, beta=0.3):
        return F.cross_entropy(
            generated_tokens.view(-1, generated_tokens.shape[-1]),
            original_tokens.view(-1)
        ) + beta * F.mse_loss(hs, phs)
    '''

    def loss(self, generated_tokens, original_tokens, generated_hid):
        return F.cross_entropy(
            generated_tokens.view(-1, generated_tokens.shape[-1]),
            original_tokens.view(-1)
        )
