
from cognition.model import Cognition
from vision.vqvae import VQVAE
import torch
from torch.nn import functional as F

def train_test(cognition: Cognition, vqvae: VQVAE, training_batches, testing_batches, epochs):
    vqvae.eval()
    
    optim = torch.optim.Adam(cognition.parameters(), lr=1e-3)

    try:
        for epoch in range(epochs):
            training_loss = 0

            cognition.train()
            for batch in training_batches:
                bs, ss, h, w = batch["frames"].shape
                hid = torch.randn(cognition.num_layers, bs, cognition.hidden_size)

                tokens = vqvae.encode(batch["frames"].view(bs * ss, 1, h, w)).long().view(bs, ss, -1)

                tcog = torch.cat([vqvae.encode(torch.zeros(bs, 1, h, w)).long().view(bs, 1, -1), tokens], dim=1)
                actions = torch.cat([torch.zeros(bs, 1, 2), batch["actions"]], dim=1)

                pred_tokens = cognition.forward(actions[:, :-1], tokens, hid, training=True)

                loss = Cognition.loss(pred_tokens, tokens, emb_hid_o, gru_o)
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                training_loss += loss

            testing_loss = 0

            cognition.eval()
            for batch in testing_batches:
                bs, ss, h, w = batch["frames"].shape

                hid = torch.randn(cognition.num_layers, bs, cognition.hidden_size)

                tokens = vqvae.encode(batch["frames"].view(bs * ss, 1, h, w)).long().view(bs, ss, -1)

                tcog = torch.cat([vqvae.encode(torch.zeros(bs, 1, h, w)).long().view(bs, 1, -1), tokens], dim=1)
                actions = torch.cat([torch.zeros(bs, 1, 2), batch["actions"]], dim=1)

<<<<<<< HEAD
                pred_tokens = cognition.forward(actions[:, :-1], tcog[:, :-1], hid, training=True)
=======
                #pred_tokens = cognition.forward(actions[:, :-1], tcog[:, :-1], training=True)
                pred_tokens, emb_hid_o, gru_o = cognition.forward(actions[:, :-1], tcog[:, :-1], training=True)
>>>>>>> 73ff260462d65868a009f114e251d090b13c01c4

                loss = Cognition.loss(pred_tokens, tokens, emb_hid_o, gru_o)
                #loss = Cognition.loss(pred_tokens, tokens)
                testing_loss += loss

            print(f"Epoch {epoch + 1} - training loss: {training_loss/len(training_batches)} - testing loss: {testing_loss/len(testing_batches)}")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        torch.save(cognition.state_dict(), "cog.ptm")
