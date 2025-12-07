
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

            tfr = max(0.5, 1.0 - epoch / (epochs * 2))

            cog.train()
            for batch in training_batches:
                #batch_loss = torch.Tensor(0).to(cog.device)
                batch_loss = 0

                fbs, fss, fw, fh = batch["frames"].shape
                frames = torch.cat([torch.zeros(fbs, 1, fw, fh), batch["frames"]], dim=1)
                tokens = vqvae.encode(frames.view(fbs * (fss + 1), 1, fh, fw)).long().view(fbs, (fss + 1), -1).permute(1, 0, 2).contiguous().detach()
                actions = batch["actions"].permute(1, 0, 2).contiguous()

                ss, bs, tks = tokens.shape

                hid = torch.zeros(bs, cog.hidden_size)
                prev_tokens = tokens[0]

                for s in range(ss - 1):
                    pred_token_logits, hid = cog.forward(actions[s], prev_tokens, hid, training=True)

                    if torch.rand(1).item() < tfr:
                        with torch.no_grad():
                            prev_tokens = torch.argmax(pred_token_logits, dim=-1)
                    else:
                        prev_tokens = tokens[s + 1]

                    hid = hid.detach()
                    
                    loss = cog.loss(pred_token_logits, tokens[s + 1], hid)

                    batch_loss += loss
                
                optim.zero_grad()
                batch_loss.backward()
                optim.step()
                
                training_loss += (batch_loss.item() / ss)

            testing_loss = 0
            cog.eval()
            for batch in testing_batches:
                #batch_loss = torch.Tensor(0).to(cog.device)
                batch_loss = 0

                fbs, fss, fw, fh = batch["frames"].shape
                frames = torch.cat([torch.zeros(fbs, 1, fw, fh), batch["frames"]], dim=1)
                tokens = vqvae.encode(frames.view(fbs * (fss + 1), 1, fh, fw)).long().view(fbs, (fss + 1), -1).permute(1, 0, 2).contiguous().detach()
                actions = batch["actions"].permute(1, 0, 2)

                ss, bs, tks = tokens.shape

                hid = torch.zeros(bs, cog.hidden_size)
                prev_tokens = tokens[0]

                for s in range(ss - 1):
                    pred_token_logits, hid = cog.forward(actions[s], prev_tokens, hid, training=True)

                    if torch.rand(1).item() < tfr:
                        with torch.no_grad():
                            prev_tokens = torch.argmax(pred_token_logits, dim=-1)
                    else:
                        prev_tokens = tokens[s + 1]

                    loss = cog.loss(pred_token_logits, tokens[s + 1], hid)

                    batch_loss += loss
                
                testing_loss += (batch_loss.item() / ss)

            print(f"Epoch {epoch + 1} - training loss: {training_loss/len(training_batches)} - testing loss: {testing_loss/len(testing_batches)}")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        torch.save(cog.state_dict(), "cog.ptm")
