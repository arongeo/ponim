
from cognition.model import Cognition
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

def train_test(model, training_batches, testing_batches, epochs):
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    try:
        for epoch in range(epochs):
            model.train()

            train_loss = 0.0

            for batch in training_batches:
                model.reset(batch["actions"].size(0))
                bs, ss, lds = batch["latframes"].shape

                latframes = torch.cat([torch.zeros(bs, 1, lds), batch["latframes"][:, :-1]], dim=1)

                for start in range(0, ss, 50):
                    optim.zero_grad()
                    
                    end = min(start + 50, ss)

                    #pred_lat, pred_res = model.forward(batch["actions"], torch.cat([torch.randn(bs, 1, lds), batch["latframes"][:, :-1]], dim=1))
                    pred_mu, pred_lv = model.forward(batch["actions"][:, start:end, :], latframes[:, start:end, :])
                    model.hc = (torch.zeros(model.num_layers, bs, model.hid_size).to(model.device), torch.zeros(model.num_layers, bs, model.hid_size).to(model.device))

                    loss = Cognition.loss(batch["latframes"][:, start:end, :], pred_mu, pred_lv)#, batch["results"], pred_res, alpha=0)

                    loss.backward(retain_graph=True)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optim.step()

                    train_loss += loss

            test_loss = 0.0

            model.eval()
            for batch in testing_batches:
                model.reset(batch["actions"].size(0))

                pred_mu, pred_lv = model.forward(batch["actions"], torch.cat([torch.zeros(bs, 1, lds), batch["latframes"][:, :-1]], dim=1))

                loss = Cognition.loss(batch["latframes"], pred_mu, pred_lv)#, batch["results"], pred_res, alpha=0)

                test_loss += loss

            print(f"Epoch {epoch + 1} - training loss: {train_loss} - testing loss: {test_loss}")
    except KeyboardInterrupt:
        print("Interrupted training, saving models")
    finally:
        torch.save(model.state_dict(), "cog_snapshot.ptm")
        torch.save(optim.state_dict(), "optim_snapshot.ptm")

