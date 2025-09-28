
from cognition.model import Cognition
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

def train_test(model, training_batches, testing_batches, epochs):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for batch in training_batches:
            model.reset(batch["actions"].size(0))

            optim.zero_grad()

            pred_lat = model.forward(batch["actions"])

            loss = Cognition.loss(batch["latframes"], pred_lat)

            loss.backward()
            optim.step()

            train_loss += loss


        test_loss = 0.0

        model.eval()
        for batch in testing_batches:
            model.reset(batch["actions"].size(0))

            pred_lat = model.forward(batch["actions"])

            loss = Cognition.loss(batch["latframes"], pred_lat)

            test_loss += loss

        print(f"Epoch {epoch + 1} - training loss: {train_loss} - testing loss: {test_loss}")

