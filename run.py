import pygame
import torch
import os

from cognition import Cognition
from vision import VAE

device = torch.accelerator.current_accelerator()

cog = Cognition(256, 32, device).to(device)
cog.reset(1)
vae = VAE(32).to(device)

if os.path.exists("vae.ptm"):
    vae.load_state_dict(torch.load("vae.ptm", weights_only=True))
else:
    print("No Vision model found, quitting")
    quit()

if os.path.exists("cog.ptm"):
    cog.load_state_dict(torch.load("cog.ptm", weights_only=True))
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
    
    rmt = torch.Tensor([[[0.0, rmovemen]]]).to(device)
    lat, res = cog.forward(rmt)
    framebuf = vae.decoder.decode(lat).squeeze().squeeze()

    print(res)

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
