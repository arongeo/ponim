import pygame
import torch
import os

from ponim import Ponim

device = torch.device("cpu")

model = Ponim(64, 256, device).to(device)
model.eval()

loaded = False
if os.path.exists("ponim.ptm"):
    model.load_state_dict(torch.load("ponim.ptm", weights_only=True, map_location=device))
    loaded = True
elif os.path.exists("ponim_snapshot.ptm"):
    model.load_state_dict(torch.load("ponim_snapshot.ptm", weights_only=True, map_location=device))
    loaded = True
if not loaded:
    print("No Ponim model found (ponim.ptm/ponim_snapshot.ptm), quitting")
    quit()

# Use internal modules
vae = model.vae
cog = model.cog

pygame.init()
screen = pygame.display.set_mode((640, 320))
clock = pygame.time.Clock()
wsquare = pygame.Surface((10, 10))
wsquare.fill((255, 255, 255))
bsquare = pygame.Surface((10, 10))
bsquare.fill((0, 0, 0))

rmovement = 0.0

running = True

prev_latent = torch.zeros(1, 1, 64).to(device)
cog.reset(1)

# Seed the latent state from a real frame if available
dataset_loaded = False
sequences = None
if os.path.exists("pongdata.pt"):
    try:
        sequences = torch.load("pongdata.pt", map_location=device)
        if len(sequences) > 0 and len(sequences[0]["frames"]) > 0:
            seed = sequences[0]["frames"][0].to(device)  # [32, 64]
            seed = seed.unsqueeze(0).unsqueeze(0)         # [1, 1, 32, 64]
            with torch.no_grad():
                mu, _ = vae.encoder.encode(seed)
            prev_latent = mu.unsqueeze(1)                 # [1, 1, 64]
            dataset_loaded = True
    except Exception:
        # Fallback to zeros if dataset can't be loaded
        prev_latent = torch.zeros(1, 1, 64, device=device)

# Runtime knobs
gain = 3.0            # scales latent magnitude before decoding
ema = 0.2             # blend factor with previous latent (stability)
teacher_forcing = False
seq_idx = 0
frame_idx = 0
if dataset_loaded:
    import random
    seq_idx = random.randrange(0, len(sequences))
    frame_idx = 0

while running:
    rmovement = 0.0

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # continuous key press handling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        rmovement = -1.0
    elif keys[pygame.K_DOWN]:
        rmovement = 1.0
    else:
        rmovement = 0.0

    # Toggle teacher-forcing with 'T', reseed from dataset with 'R'
    if keys[pygame.K_t] and dataset_loaded:
        teacher_forcing = not teacher_forcing
        pygame.time.wait(150)
    if keys[pygame.K_r] and dataset_loaded:
        try:
            seed = sequences[seq_idx]["frames"][frame_idx].to(device)
            seed = seed.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                mu, _ = vae.encoder.encode(seed)
            prev_latent = mu.unsqueeze(1)
        except Exception:
            pass

    # Adjust gain live with left/right arrows (hold Shift for bigger steps)
    step = 0.1 + 0.4 * keys[pygame.K_LSHIFT]
    if keys[pygame.K_LEFT]:
        gain = max(0.1, gain - step)
    if keys[pygame.K_RIGHT]:
        gain = min(10.0, gain + step)
    
    with torch.no_grad():
        if teacher_forcing and dataset_loaded:
            # Step through real frames by encoding latents (verifies VAE/display)
            seed = sequences[seq_idx]["frames"][frame_idx].to(device)
            seed = seed.unsqueeze(0).unsqueeze(0)
            mu, _ = vae.encoder.encode(seed)
            lat = mu.unsqueeze(1)
            prev_latent = lat.clone().detach()
            framebuf = vae.decoder.decode(lat * gain).squeeze().squeeze()
            frame_idx = (frame_idx + 1) % len(sequences[seq_idx]["frames"])
        else:
            rmt = torch.tensor([[[0.0, rmovement]]], dtype=torch.float32, device=device)
            lat = cog.forward(rmt, prev_latent)
            # Blend with previous latent for smoother dynamics
            lat = (1.0 - ema) * lat + ema * prev_latent
            prev_latent = lat.clone().detach()
            # Scale latent magnitude to escape mid-gray basin
            framebuf = vae.decoder.decode(lat * gain).squeeze().squeeze()

    # print a tiny status line
    mode = "TF" if teacher_forcing else "FREE"
    print(f"[{mode}] gain={gain:.2f} mean={float(framebuf.mean()):.4f} min={float(framebuf.min()):.4f} max={float(framebuf.max()):.4f}")

    # clear screen
    screen.fill((0, 0, 0))

    # threshold to 0/1 for display
    frame_np = (framebuf.cpu().numpy() > 0.5)
    for i in range(32):
        for j in range(64):
            if 0.5 < frame_np[i][j]:
                rect = pygame.Rect(j * 10, i * 10, 10, 10)
                screen.blit(wsquare, rect)
            else:
                rect = pygame.Rect(j * 10, i * 10, 10, 10)
                screen.blit(bsquare, rect)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(30)
