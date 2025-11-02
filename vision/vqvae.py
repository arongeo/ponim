import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from vision.encoder import Encoder
from vision.decoder import Decoder

class VQVAE(nn.Module):
    def __init__(self, codebook_size):
        super().__init__()
        self.codebook_size = codebook_size
        # For a 64x32 pixel frame, we'll get back
        # a 128 channel convolution
        self.latent_dim_size = 128 

        # We split the VAE, since, for our purpose,
        # we'll only need the decoder once training
        # is done.
        self.encoder = Encoder()

        self.quantizer = nn.Embedding(codebook_size, 128)
        self.quantizer.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.decoder = Decoder()

    def forward(self, input_frame):
        encoded_frame = self.encoder.encode(input_frame)

        # We don't need the spatial information, we just need the encoded data:
        # First put the channel data to the end with permute,
        # then group the batch size, height and width together into one dimension
        ze = encoded_frame.permute(0, 2, 3, 1).contiguous()
        ze_flat = ze.reshape(-1, 128)
        
        # L2 norm:
        #   ||z - e||2 = ∑((z - e)^2) = ∑(z^2) + ∑(e^2) - ∑(2ze)
        z_norm = torch.sum(ze_flat**2, dim=1, keepdim=True)
        e_norm = torch.sum(self.quantizer.weight**2, dim=1).unsqueeze(0)
        dp = 2 * torch.matmul(ze_flat, self.quantizer.weight.t())
        dist = z_norm + e_norm - dp
        
        # We get the indices/tokens of the closest codebook entries and their values
        tokens = torch.argmin(dist, dim=1)
        token_multiplier = torch.zeros(tokens.shape[0], self.codebook_size)
        token_multiplier.scatter_(1, tokens, 1)

        zq = torch.matmul(token_multiplier, self.quantizer.weight).view(ze.shape)

        # Straight-through estimator 
        # (we pass the encoder the same gradients as we have in the decoder)
        zq = ze + (zq - ze).detach() # TODO: possible change here, if it doesn't work

        reconstruction = self.decoder.decode(zq.permute(0, 3, 1, 2).contiguous())

        return reconstruction, zq, ze


    @staticmethod
    def loss(original, reconstruction, zq, ze, beta=1.0) -> torch.Tensor:
        reconstruction_loss = F.binary_cross_entropy(reconstruction, original, reduction='mean')
        codebook_loss = F.mse_loss(zq, ze.detach(), reduction='mean')
        commitment_loss = F.mse_loss(zq.detach(), ze, reduction='mean')

        return reconstruction_loss + codebook_loss + commitment_loss * beta
