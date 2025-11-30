
from cognition.model import Cognition
from vision.vqvae import VQVAE
import torch
from torch.nn import functional as F

def train_test(cog: Cognition, vqvae: VQVAE, training_batches, testing_batches, epochs):
    vqvae.eval()
    
    optim = torch.optim.Adam(cog.parameters(), lr=1e-3)

    try:
        for epoch in range(epochs):
            training_loss = 0

            cog.train()
            for batch in training_batches:
                batch_loss = 0
                frames = batch["frames"].permute(1, 0, 2, 3)
                actions = batch["actions"].permute(1, 0, 2, 3)

                _, bs, h, w = frames.shape

                #frames = torch.cat([torch.zeros(1, bs, h, w), frames], dim=0)
                actions = torch.cat([torch.zeros(1, bs, 2), actions[:-1]], dim=0)

                ss = frames.shape[0]

                hid = torch.zeros(bs, cog.hidden_size)

                for s in range(ss):
                    tokens = vqvae.encode(frames[s].view(bs, 1, h, w)).long().view(bs, -1)

                    pred_tokens, hid = cog.forward(actions[s], hid)

                    loss = Cognition.loss(pred_tokens, tokens[s])
                
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    batch_loss += loss
                
                training_loss += (batch_loss / ss)

            testing_loss = 0
            cog.eval()
            for batch in testing_batches:
                batch_loss = 0
                frames = batch["frames"].permute(1, 0, 2, 3)
                actions = batch["actions"].permute(1, 0, 2, 3)

                _, bs, h, w = frames.shape

                #frames = torch.cat([torch.zeros(1, bs, h, w), frames], dim=0)
                actions = torch.cat([torch.zeros(1, bs, 2), actions[:-1]], dim=0)

                ss = frames.shape[0]

                hid = torch.zeros(bs, cog.hidden_size)

                for s in range(ss):
                    tokens = vqvae.encode(frames[s].view(bs, 1, h, w)).long().view(bs, -1)

                    pred_tokens, hid = cog.forward(actions[s], hid)

                    loss = Cognition.loss(pred_tokens, tokens[s])
                    batch_loss += loss
                
                testing_loss += (batch_loss / ss)

            print(f"Epoch {epoch + 1} - training loss: {training_loss/len(training_batches)} - testing loss: {testing_loss/len(testing_batches)}")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        torch.save(cog.state_dict(), "cog.ptm")
