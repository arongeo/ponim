
from ponim.model import Ponim
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

def train_test(ponim: Ponim, training_batches, testing_batches, epochs: int):
    optim = torch.optim.Adam(ponim.parameters(), lr=1e-3)
    use_amp = False  # disable AMP to avoid dtype mismatches during backward
    recon_weight = 1.0
    latent_weight = 0.1

    try:
        for epoch in range(epochs):
            ponim.train()

            train_loss = 0.0
            train_steps = 0

            for batch in training_batches:
                ponim.cog.reset(batch["actions"].size(0))

                bs, ss, h, w = batch["frames"].shape

                with torch.no_grad():
                    mu, logvar = ponim.vae.encoder.encode(batch["frames"].view(bs * ss, 1, h, w))
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    z = z.view(bs, ss, -1)

                optim.zero_grad(set_to_none=True)

                # Forward cognition and decode in float32
                pred_z = ponim.cog.forward(batch["actions"][:, :-1], z[:, :-1]).view(bs * (ss - 1), -1)
                pred_frame = ponim.vae.decoder.decode(pred_z).view(bs, (ss - 1), h, w)

                # Normalize BCE by number of pixels and elements
                pred_frame_bce = F.binary_cross_entropy(
                    pred_frame.view(bs * (ss - 1), 1, h, w).float(),
                    batch["frames"][:, 1:].contiguous().view(bs * (ss - 1), 1, h, w).float(),
                    reduction='sum'
                ) / (bs * (ss - 1) * h * w)

                # Latent prediction loss (compare to next latent)
                pred_z_seq = pred_z.view(bs, (ss - 1), -1)
                latent_target = z[:, 1:]
                latent_l2 = F.mse_loss(pred_z_seq.float(), latent_target.float(), reduction='mean')

                loss = recon_weight * pred_frame_bce + latent_weight * latent_l2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(ponim.parameters(), max_norm=1.0)
                optim.step()

                train_loss += loss.detach().item()
                train_steps += 1

            test_loss = 0.0
            test_steps = 0

            ponim.eval()
            for batch in testing_batches:
                ponim.cog.reset(batch["actions"].size(0))

                bs, ss, h, w = batch["frames"].shape

                with torch.no_grad():
                    mu, logvar = ponim.vae.encoder.encode(batch["frames"].view(bs * ss, 1, h, w))
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    z = z.view(bs, ss, -1)

                    pred_z = ponim.cog.forward(batch["actions"][:, :-1], z[:, :-1]).view(bs * (ss - 1), -1)
                    pred_frame = ponim.vae.decoder.decode(pred_z).view(bs, (ss - 1), h, w)

                    pred_frame_bce = F.binary_cross_entropy(
                        pred_frame.view(bs * (ss - 1), 1, h, w).float(),
                        batch["frames"][:, 1:].contiguous().view(bs * (ss - 1), 1, h, w).float(),
                        reduction='sum'
                    ) / (bs * (ss - 1) * h * w)
                    pred_z_seq = pred_z.view(bs, (ss - 1), -1)
                    latent_target = z[:, 1:]
                    latent_l2 = F.mse_loss(pred_z_seq.float(), latent_target.float(), reduction='mean')
                    loss = recon_weight * pred_frame_bce + latent_weight * latent_l2

                test_loss += loss.item()
                test_steps += 1

            avg_train = train_loss / max(1, train_steps)
            avg_test = test_loss / max(1, test_steps)
            print(f"Epoch {epoch + 1} - training loss: {avg_train:.6f} - testing loss: {avg_test:.6f}")
    except KeyboardInterrupt:
        print("Interrupted training")
    finally:
        torch.save(ponim.state_dict(), "ponim_snapshot.ptm")
        torch.save(optim.state_dict(), "optim_snapshot.ptm")
