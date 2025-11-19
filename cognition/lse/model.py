
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from vision.vqvae import VQVAE

class LSE(nn.Module):
    def __init__(self, hidden_size, vqvae: VQVAE):
        super().__init__()
        self.hidden_size = hidden_size

        self.vqvae = vqvae
        for p in self.vqvae.parameters():
            p.requires_grad = False

        self.linear_emb_hid = nn.Linear(self.vqvae.latent_dim_size * self.vqvae.quantizer.embedding_dim, hidden_size)
        self.linear_hid_emb = nn.Linear(hidden_size, self.vqvae.latent_dim_size * self.vqvae.codebook_size)

    def forward(self, emb):
        bs, _, _ = emb.shape
        o = self.linear_emb_hid(emb.flatten(1))
        o = self.linear_hid_emb(o)
        return o.view(bs, self.vqvae.latent_dim_size, self.vqvae.codebook_size)

    @staticmethod
    def loss(generated_logits, original_tokens):
        return F.cross_entropy(
            generated_logits.view(-1, generated_logits.shape[-1]),
            original_tokens.view(-1)
        )
