
from cognition.lse.model import LSE
from vision.vqvae import VQVAE
import torch
from torch.nn import functional as F

def train_test(lse: LSE, train_loader, test_loader, epochs: int):
    lse.train()
    lse.vqvae.eval()

    optim = torch.optim.Adam(lse.parameters(), lr=1e-3)

    for epoch in range(epochs):
        lse.train()
        training_loss = 0
        for batch in train_loader:
            optim.zero_grad()

            with torch.no_grad():
                tokens = lse.vqvae.encode(batch)
                embs = lse.vqvae.quantizer(tokens.unsqueeze(1))

            reconst = lse.forward(embs)

            loss = LSE.loss(reconst.squeeze(1), tokens)
            
            loss.backward()
            optim.step()

            training_loss += loss.item()

        lse.eval()
        testing_loss = 0
        for batch in test_loader:
            with torch.no_grad():
                tokens = lse.vqvae.encode(batch)
                embs = lse.vqvae.quantizer(tokens.unsqueeze(1))

            reconst = lse.forward(embs)

            loss = LSE.loss(reconst.squeeze(1), tokens)

            testing_loss += loss.item()
        print(f"Epoch {epoch + 1} - training loss: {training_loss/len(train_loader.dataset)} - testing loss: {testing_loss/len(test_loader.dataset)}")

