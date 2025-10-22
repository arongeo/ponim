import pygame
import torch
import os

from cognition import Cognition
from vision import VAE

device = torch.device("cpu")

vae = VAE(64).to(device)
cog = Cognition(512, 5, 64, device)
cog.reset(1)

if os.path.exists("vae.ptm"):
    vae.load_state_dict(torch.load("vae.ptm", weights_only=True, map_location=device))
else:
    print("No Vision model found, quitting")
    quit()

if os.path.exists("cog_snapshot.ptm"):
    cog.load_state_dict(torch.load("cog_snapshot.ptm", weights_only=True, map_location=device))
else:
    print("No Cognition model found, quitting")
    quit()

pygame.init()
screen = pygame.display.set_mode((640, 320))
clock = pygame.time.Clock()
wsquare = pygame.Surface((10, 10))
wsquare.fill((255, 255, 255))
bsquare = pygame.Surface((10, 10))
bsquare.fill((0, 0, 0))

rmovement = 0.0

running = True

prev_lat_frames = torch.randn(1, 5, 64).to(device) * 0.01

while running:
    rmovement = 0

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        rmovement = -1.0
    elif keys[pygame.K_DOWN]:
        rmovement = 1.0
    
    rmt = torch.Tensor([[[0.0, rmovement]]]).to(device)
    
    print(rmt)

    mco, lat, stdev = cog.forward(rmt, prev_lat_frames.view(1, 1, -1))
    most_likely = torch.argmax(mco, dim=-1).squeeze().squeeze().squeeze()
    prev_lat_frames = torch.cat([lat[:, :, most_likely].clone().detach(), prev_lat_frames[:, :-1]], dim=1)
    framebuf = vae.decoder.decode(lat[:, :, most_likely]).squeeze().squeeze()

    for i in range(32):
        for j in range(64):
            if framebuf[i][j] == 1:
                rect = pygame.Rect(j * 10, i * 10, 10, 10)
                screen.blit(wsquare, rect)
            else:
                rect = pygame.Rect(j * 10, i * 10, 10, 10)
                screen.blit(bsquare, rect)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(30)
