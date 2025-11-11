from cognition.model import Cognition
import vision
import cognition
import torch
import random
from torchvision.utils import save_image
import os

from vision.trainer import train_test
#import ponim

#device = torch.device("cpu")
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
train_size = int(0.85 * total_frames)
test_size = total_frames - train_size

vqvae = vision.VQVAE(16).to(device)

if os.path.exists("vae.ptm"):
    vqvae.load_state_dict(torch.load("vae.ptm", weights_only=True, map_location=device))
else:
    train_set, test_set = torch.utils.data.random_split(
        allframes, 
        [train_size, test_size], 
        generator=torch.Generator(device=device).manual_seed(2025)
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, generator=torch.Generator(device=device))

    print("training on:", len(train_set), "frames - testing on:", len(test_set), "frames")

    vision.train_test(vqvae, train_loader, test_loader, 10, beta=0.5)
    torch.save(vqvae.state_dict(), "vae.ptm")

random.shuffle(allframes)
num_frames = 10
rand_i = random.randint(0, len(allframes) - 1 - num_frames)
vision.sample(vqvae, torch.Tensor(allframes[rand_i:rand_i+num_frames]).clone().detach())

del allframes

grouped_seqs = {}
for sequence in sequences:
    if len(sequence["frames"]) not in grouped_seqs:
        grouped_seqs[len(sequence["frames"])] = []
    grouped_seqs[len(sequence["frames"])].append(sequence)

batches = []
curr_batch = []
prev_length = list(grouped_seqs.keys())[0]
for length, seqs in grouped_seqs.items():
    for sequence in seqs:
        if len(curr_batch) == 128 or length != prev_length:
            batches.append({
                "frames": torch.stack([seq["frames"].to(device) for seq in curr_batch]),
                "actions": torch.stack([seq["actions"].to(device) for seq in curr_batch]),
                "results": torch.stack([seq["results"].to(device) for seq in curr_batch]),
            })
            curr_batch = []
        curr_batch.append(sequence)
    prev_length = length

if len(curr_batch) != 0:
    batches.append({
        "frames": torch.stack([seq["frames"].to(device) for seq in curr_batch]),
        "actions": torch.stack([seq["actions"].to(device) for seq in curr_batch]),
        "results": torch.stack([seq["results"].to(device) for seq in curr_batch]),
    })

random.shuffle(batches)

training_testing_split = int(0.8 * len(batches))

training_batches = batches[:training_testing_split]
testing_batches = batches[training_testing_split:]

cog = Cognition(256, vqvae, device)

cognition.train_test(cog, vqvae, training_batches, testing_batches, 100)
