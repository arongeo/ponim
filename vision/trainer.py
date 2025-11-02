import vision
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

def train_test(model, train_loader, test_loader, epochs, beta=0.25):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            reconstruction, loss = model.train_forward(batch, beta=beta)
            loss += F.binary_cross_entropy(reconstruction, batch, reduction='mean')

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                reconstruction, loss = model.train_forward(batch, beta=beta)
                loss += F.binary_cross_entropy(reconstruction, batch, reduction='mean')

                test_loss += loss.item()

        print(f"Epoch {epoch + 1} - training loss: {train_loss/len(train_loader.dataset)} - testing loss: {test_loss/len(test_loader.dataset)}")

def sample(model, frame, filename):
    model.eval()
    
    reconstruction, _ = model.train_forward(frame)

    save_image(frame.clone().detach().cpu(), filename + "_original.png")
    save_image(reconstruction.clone().detach().cpu(), filename + "_reconstructed.png")
