import vision
import cognition
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
train_size = int(0.85 * total_frames)
test_size = total_frames - train_size

vae_model = vision.VAE(32).to(device)

if os.path.exists("vae.ptm"):
    vae_model.load_state_dict(torch.load("vae.ptm", weights_only=True))
else:
    train_set, test_set = torch.utils.data.random_split(
        allframes, 
        [train_size, test_size], 
        generator=torch.Generator(device=device).manual_seed(2025)
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, generator=torch.Generator(device=device))

    print("training on:", len(train_set), "frames - testing on:", len(test_set), "frames")

    vision.train_test(vae_model, train_loader, test_loader, 50)
    torch.save(vae_model.state_dict(), "vae.ptm")

for n in range(10):
    vision.sample(vae_model, allframes[random.randint(0, len(allframes) - 1)].clone().detach().unsqueeze(0), str(n))

del allframes

if "latframes" not in sequences[0]:
    with torch.no_grad():
        for sid, sequence in enumerate(sequences):
            sequences[sid]["latframes"] = []
            for frame in sequence["frames"]:
                sequences[sid]["latframes"].append(vae_model.encoder.encode(frame.unsqueeze(0).unsqueeze(0).to(device))[0].squeeze())
            sequences[sid]["latframes"] = torch.stack(sequences[sid]["latframes"])
    torch.save(sequences, "pongdata.pt")

grouped_seqs = {}
for sequence in sequences:
    if len(sequence["latframes"]) not in grouped_seqs:
        grouped_seqs[len(sequence["latframes"])] = []
    grouped_seqs[len(sequence["latframes"])].append(sequence)

batches = []
curr_batch = []
prev_length = list(grouped_seqs.keys())[0]
for length, seqs in grouped_seqs.items():
    for sequence in seqs:
        if len(curr_batch) == 32 or length != prev_length:
            batches.append({
                "latframes": torch.stack([seq["latframes"].to(device) for seq in curr_batch]),
                "actions": torch.stack([torch.cat([torch.zeros(2).unsqueeze(0).to(device), seq["actions"][:-1].to(device)]) for seq in curr_batch]),
                "results": torch.stack([seq["results"].to(device) for seq in curr_batch]),
            })
            curr_batch = []
        curr_batch.append(sequence)
    prev_length = length

if len(curr_batch) != 0:
    batches.append({
        "latframes": torch.stack([seq["latframes"].to(device) for seq in curr_batch]),
        "actions": torch.stack([torch.cat([torch.zeros(2).unsqueeze(0).to(device), seq["actions"][:-1].to(device)]) for seq in curr_batch]),
        "results": torch.stack([seq["results"].to(device) for seq in curr_batch]),
    })

random.shuffle(batches)

training_testing_split = int(0.8 * len(batches))

training_batches = batches[:training_testing_split]
testing_batches = batches[training_testing_split:]

cog = cognition.Cognition(256, 32, device)

cognition.train_test(cog, training_batches, testing_batches, 100)

torch.save(cog.state_dict(), "cog.ptm")
