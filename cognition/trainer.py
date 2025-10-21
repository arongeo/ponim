
from cognition.model import Cognition
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from vision import VAE

from vision import VAE
from vision.encoder import Encoder

def train_test(model: Cognition, vae_encoder: Encoder, training_batches, testing_batches, epochs):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        for epoch in range(epochs):
            model.train()

            train_loss = 0.0

            for batch in training_batches:
                model.reset(batch["actions"].size(0))

                bs, ss, h, w = batch["frames"].shape

                vae_encoder.eval()
                with torch.no_grad():
                    mu, logvar = vae_encoder.encode(batch["frames"].view(bs * ss, 1, h, w))
                    #std = torch.exp(0.5 * logvar)
                    #z = mu + std * torch.randn_like(std)
                    z = mu.view(bs, ss, -1)
                    z = torch.cat([torch.randn(bs, 5, z.shape[2]) * 0.01, z], dim=1)

                optim.zero_grad()

                actions = torch.cat([torch.zeros(bs, 5, batch["actions"].shape[2]), batch["actions"]], dim=1)
                prev_frames = torch.cat([z[:, :-5], z[:, 1:-4], z[:, 2:-3], z[:, 3:-2], z[:, 4:-1]], dim=1)

                mco, mu, stdev = model.forward(actions, prev_frames)

                loss = Cognition.loss(z[:, 5:], mco, mu, stdev)

                loss.backward()
                optim.step()

                train_loss += loss

            test_loss = 0.0

            model.eval()
            for batch in testing_batches:
                model.reset(batch["actions"].size(0))

                bs, ss, h, w = batch["frames"].shape

                vae_encoder.eval()
                with torch.no_grad():
                    mu, logvar = vae_encoder.encode(batch["frames"].view(bs * ss, 1, h, w))
                    #std = torch.exp(0.5 * logvar)
                    #z = mu + std * torch.randn_like(std)
                    z = mu.view(bs, ss, -1)
                    z = torch.cat([torch.randn(bs, 5, z.shape[2]) * 0.01, z], dim=1)

                optim.zero_grad()

                actions = torch.cat([torch.zeros(bs, 5, batch["actions"].shape[2]), batch["actions"]], dim=1)
                prev_frames = torch.cat([z[:, :-5], z[:, 1:-4], z[:, 2:-3], z[:, 3:-2], z[:, 4:-1]], dim=1)

                mco, mu, stdev = model.forward(actions, prev_frames)

                loss = Cognition.loss(z[:, 5:], mco, mu, stdev)

                test_loss += loss

            print(f"Epoch {epoch + 1} - training loss: {train_loss/len(training_batches)} - testing loss: {test_loss/len(testing_batches)}")
    except KeyboardInterrupt:
        print("Interrupted training")
    finally:
        torch.save(model.state_dict(), "cog_snapshot.ptm")
        torch.save(optim.state_dict(), "optim_snapshot.ptm")

