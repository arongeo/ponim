
from ponim.model import Ponim
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from vision import VAE

def train_test(ponim: Ponim, training_batches: torch.Tensor, testing_batches: torch.Tensor, epochs: int):
    optim = torch.optim.Adam(ponim.parameters(), lr=1e-3)

    try:
        for epoch in range(epochs):
            ponim.vae.eval()
            ponim.cog.train()

            train_loss = 0.0

            for batch in training_batches:
                ponim.cog.reset(batch["actions"].size(0))

                bs, ss, h, w = batch["frames"].shape

                target_frames = batch["frames"].view(bs * ss, 1, h, w)

                with torch.no_grad():
                    mu, logvar = ponim.vae.encoder.encode(target_frames)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    z = z.view(bs, ss, -1)

                pred_z = ponim.cog.forward(batch["actions"])
                pred_frame = ponim.vae.decoder.decode(pred_z)
                loss = Ponim.loss(pred_frame, target_frames)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss += loss

            test_loss = 0.0

            ponim.vae.eval()
            ponim.cog.eval()
            ponim.eval()
            for batch in testing_batches:
                ponim.cog.reset(batch["actions"].size(0))

                bs, ss, h, w = batch["frames"].shape

                target_frames = batch["frames"].view(bs * ss, 1, h, w)

                with torch.no_grad():
                    mu, logvar = ponim.vae.encoder.encode(target_frames)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    z = z.view(bs, ss, -1)

                pred_z = ponim.cog.forward(batch["actions"])
                pred_frame = ponim.vae.decoder.decode(pred_z)
                loss = Ponim.loss(pred_frame, target_frames)

                optim.zero_grad()

                test_loss += loss

            print(f"Epoch {epoch + 1} - training loss: {train_loss} - testing loss: {test_loss}")
    except KeyboardInterrupt:
        print("Interrupted training")
    finally:
        torch.save(ponim.state_dict(), "cog_snapshot.ptm")
        torch.save(optim.state_dict(), "optim_snapshot.ptm")