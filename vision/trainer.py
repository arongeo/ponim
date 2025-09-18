import vision
import torch
from torchvision.utils import save_image

def train_test(model, train_loader, test_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()

        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            reconstruction, mu, logvar = model.forward(batch)
            loss = vision.VAE.loss(batch, reconstruction, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                reconstruction, mu, logvar = model.forward(batch)
                loss = vision.VAE.loss(batch, reconstruction, mu, logvar)

                test_loss += loss.item()

        print(f"Epoch {epoch + 1} - training loss: {train_loss/len(train_loader.dataset)} - testing loss: {test_loss/len(test_loader.dataset)}")

def sample(model, frame, filename):
    model.eval()
    
    m, lv = model.encoder.encode(frame)
    z = model.reparameterize(m, lv)

    save_image(frame.clone().detach().cpu(), filename + "_original.png")
    save_image(model.decoder.decode(z).clone().detach().cpu(), filename + "_reconstructed.png")
