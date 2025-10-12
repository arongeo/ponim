import pygame
import torch
import os

from ponim import Ponim

device = torch.device("cpu")

model = Ponim(64, 256, device)

if os.path.exists("ponim_snapshot.ptm"):
    model.load_state_dict(torch.load("ponim_snapshot.ptm", weights_only=True, map_location=device))
else:
    print("No Vision model found, quitting")
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

prev_lat_frames = torch.randn(1, 5, 64) * 0.01

while running:
    rmovement = 0

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                rmovement = -1.0
            if event.key == pygame.K_DOWN:
                rmovement = 1.0
    
    rmt = torch.Tensor([[[0.0, rmovement]]]).to(device)
    lat = model.cog.forward(rmt, prev_lat_frames.view(1, 1, -1))
    prev_lat_frames = torch.cat([prev_lat_frames[:, 1:], lat.clone().detach()], dim=1)
    framebuf = torch.round(model.vae.decoder.decode(lat).squeeze().squeeze())

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
