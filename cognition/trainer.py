
from cognition.model import Cognition
from vision.vqvae import VQVAE
import torch
from torch.nn import functional as F

def train_test(cognition: Cognition, vqvae: VQVAE, training_batches, testing_batches, epochs):
    vqvae.eval()
    
    optim = torch.optim.Adam(cognition.parameters(), lr=1e-3)

    for epoch in epochs:
        training_loss = 0

        cognition.train()
        for batch in training_batches:
            tokens = vqvae.encode(batch).int()

            pass

        testing_loss = 0

        cognition.eval()
        for batch in testing_batches:
            pass

        print(f"Epoch {epoch + 1} - training loss: {training_loss/len(training_batches)} - testing loss: {testing_loss/len(testing_batches)}")
