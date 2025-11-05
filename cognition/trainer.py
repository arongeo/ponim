
from cognition.model import Cognition
from vision.vqvae import VQVAE
import torch
from torch.nn import functional as F

def train_test(cognition: Cognition, vqvae: VQVAE, training_batches, testing_batches, epochs):
    vqvae.eval()
    
    optim = torch.optim.Adam(cognition.parameters(), lr=1e-3)

    for epoch in range(epochs):
        training_loss = 0

        cognition.train()
        for batch in training_batches:
            bs, ss, h, w = batch["frames"].shape

            cognition.reset(bs) # TODO: don't know if this is actually needed in this form

            tokens = vqvae.encode(batch["frames"].view(bs * ss, 1, h, w)).int().view(bs, ss, -1)

            # TODO: Maybe try randn instead of zeros here to introduce randomness
            tcog = torch.cat([torch.zeros(bs, 1, cognition.latent_dim_size), tokens], dim=1)
            actions = torch.cat([torch.zeros(bs, 1, 2), batch["actions"]], dim=1)

            #cognition.forward(actions[:, :-1], tcog[:, :-1])

        testing_loss = 0

        cognition.eval()
        for batch in testing_batches:
            pass

        print(f"Epoch {epoch + 1} - training loss: {training_loss/len(training_batches)} - testing loss: {testing_loss/len(testing_batches)}")
