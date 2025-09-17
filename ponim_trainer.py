import ponim
import torch
import random
from torchvision.utils import save_image

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.accelerator.current_accelerator()
torch.set_default_device(device)

print("running on", device)

sequences = torch.load("pongdata.pt")

allframes = []

for sequence in sequences:
    for frame in sequence["frames"]:
        allframes.append(frame.unsqueeze(0)) # Add a channel cause conv layers expect it

allframes = torch.stack(allframes).to(device)

# Calculate actual counts
total_frames = len(allframes)
used_frames = int(0.15 * total_frames)
train_size = int(0.85 * used_frames)
test_size = used_frames - train_size

frames_used, _ = torch.utils.data.random_split(
    allframes, 
    [used_frames, total_frames - used_frames], 
    generator=torch.Generator(device=device).manual_seed(2025)
)

train_set, test_set = torch.utils.data.random_split(
    frames_used, 
    [train_size, test_size], 
    generator=torch.Generator(device=device).manual_seed(2025)
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))

print(len(frames_used))
print(len(train_set))
print(len(test_set))

vae = ponim.VAE(128).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def traintest(epochs):
    for epoch in range(epochs):
        vae.train()

        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            reconstruction, mu, logvar = vae.forward(batch)
            loss = ponim.VAE.loss(batch, reconstruction, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                reconstruction, mu, logvar = vae.forward(batch)
                loss = ponim.VAE.loss(batch, reconstruction, mu, logvar)

                test_loss += loss.item()

        print(f"Epoch {epoch + 1} - training loss: {train_loss/len(train_loader.dataset)} - testing loss: {test_loss/len(test_loader.dataset)}")

def sample(frame, filename):
    vae.eval()
    
    m, lv = vae.encode(frame)
    z = vae.reparameterize(m, lv)

    save_image(frame.clone().detach().cpu(), filename + "_original.png")
    save_image(vae.decode(z).clone().detach().cpu(), filename + "_reconstructed.png")

traintest(50)

sample(frames_used[random.randint(0, len(frames_used) - 1)].clone().detach().unsqueeze(0), "a")
sample(frames_used[random.randint(0, len(frames_used) - 1)].clone().detach().unsqueeze(0), "b")
sample(frames_used[random.randint(0, len(frames_used) - 1)].clone().detach().unsqueeze(0), "c")
sample(frames_used[random.randint(0, len(frames_used) - 1)].clone().detach().unsqueeze(0), "d")
sample(frames_used[random.randint(0, len(frames_used) - 1)].clone().detach().unsqueeze(0), "e")
