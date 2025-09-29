
from cognition.model import Cognition
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from vision import VAE

def train_test(model, encoder: VAE, training_batches, testing_batches, epochs):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        for epoch in range(epochs):
            model.train()

            train_loss = 0.0

            for batch in training_batches:
                model.reset(batch["actions"].size(0))
                
                optim.zero_grad()

                mu, logvar = encoder.encode(batch["frames"])
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(std) * std

                bs, _, lds = z.shape

                pred_lat = model.forward(batch["actions"])

                loss = Cognition.loss(z, pred_lat)

                loss.backward()
                optim.step()

                train_loss += loss

            test_loss = 0.0

            model.eval()
            for batch in testing_batches:
                model.reset(batch["actions"].size(0))

                bmu, logvar = encoder.encode(batch["frames"])
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(std) * std

                bs, _, lds = z.shape
                
                pred_lat = model.forward(batch["actions"], torch.cat([torch.zeros(bs, 1, lds), z[:, :-1]], dim=1))

                loss = Cognition.loss(z, pred_lat)

                test_loss += loss

            print(f"Epoch {epoch + 1} - training loss: {train_loss} - testing loss: {test_loss}")
    except KeyboardInterrupt:
        print("Interrupted training")
    finally:
        torch.save(model.state_dict(), "cog_snapshot.ptm")
        torch.save(optim.state_dict(), "optim_snapshot.ptm")

