import vision 
import torch
import random
from torchvision.utils import save_image
import os

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

vae_model = vision.VAE(32).to(device)

if os.path.exists("vae.ptm"):
    vae_model.load_state_dict(torch.load("vae.ptm", weights_only=True))
else:
    train_set, test_set = torch.utils.data.random_split(
        frames_used, 
        [train_size, test_size], 
        generator=torch.Generator(device=device).manual_seed(2025)
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))

    print("training on:", len(train_set), "frames - testing on:", len(test_set), "frames")

    vision.train_test(vae_model, train_loader, test_loader, 50)
    torch.save(vae_model.state_dict(), "vae.ptm")

print("sampling from", len(frames_used), "frames")

for n in range(10):
    vision.sample(vae_model, frames_used[random.randint(0, len(frames_used) - 1)].clone().detach().unsqueeze(0), str(n))
