import vision
import torch
from torchvision.utils import save_image

def train_test(model, train_loader, test_loader, epochs, beta=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            reconstruction, zq, ze = model.forward(batch)
            loss = vision.VQVAE.loss(batch, reconstruction, zq, ze, beta=beta)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                reconstruction, zq, ze = model.forward(batch)
                loss = vision.VQVAE.loss(batch, reconstruction, zq, ze, beta=beta)

                test_loss += loss.item()

        print(f"Epoch {epoch + 1} - training loss: {train_loss/len(train_loader.dataset)} - testing loss: {test_loss/len(test_loader.dataset)}")

def sample(model, frame, filename):
    model.eval()
    
    reconstruction, zq, ze = model.forward(frame)

    save_image(frame.clone().detach().cpu(), filename + "_original.png")
    save_image(reconstruction.clone().detach().cpu(), filename + "_reconstructed.png")
