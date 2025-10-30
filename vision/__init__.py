# vision contains all code in relation to the
# VAE, which is responsible for encoding gameplay
# frames into a latent space, as well as
# generating valid frames from latent vectors
# 
# quick file guide:
# - vae.py:     holds together the encoder and decoder
#               for the training
# - encoder.py: the encoder, which generates the 
#               latent area from a gameplay frame
#               (mostly unused after training)
# - decoder.py: the decoder, generates a gameplay frame
#               from a latent vector
# - trainer.py: contains the training and sampling functions

from vision.vqvae import VQVAE
from vision.trainer import train_test, sample
