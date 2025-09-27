
from cognition.model import Cognition
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

def train_test(model, training_batches, testing_batches, epochs):
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for batch in training_batches:
            model.reset(batch["actions"].size(0))
            bs, ss, lds = batch["latframes"].shape

            optim.zero_grad()

            #pred_lat, pred_res = model.forward(batch["actions"], torch.cat([torch.randn(bs, 1, lds), batch["latframes"][:, :-1]], dim=1))
            pred_lat = model.forward(batch["actions"], torch.cat([torch.randn(bs, 1, lds), batch["latframes"][:, :-1]], dim=1))

            loss = Cognition.loss(batch["latframes"], pred_lat)#, batch["results"], pred_res, alpha=0)

            loss.backward()
            optim.step()

            train_loss += loss


        test_loss = 0.0

        model.eval()
        for batch in testing_batches:
            model.reset(batch["actions"].size(0))

            pred_lat = model.forward(batch["actions"], torch.cat([torch.randn(bs, 1, lds), batch["latframes"][:, :-1]], dim=1))

            loss = Cognition.loss(batch["latframes"], pred_lat)#, batch["results"], pred_res, alpha=0)

            test_loss += loss

        print(f"Epoch {epoch + 1} - training loss: {train_loss} - testing loss: {test_loss}")

