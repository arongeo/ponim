import pygame
import torch
import os

import cognition
import vision

device = torch.device("cpu")

vqvae = vision.VQVAE(16).to(device)
lse = cognition.lse.model.LSE(256, vqvae)
cog = cognition.Cognition(lse, device)

if os.path.exists("vae.ptm"):
    vqvae.load_state_dict(torch.load("vae.ptm", weights_only=True, map_location=device))
else:
    print("No Vision model found, quitting")
    quit()

if os.path.exists("cog.ptm"):
    cog.load_state_dict(torch.load("cog.ptm", weights_only=True, map_location=device))
else:
    print("No Vision model found, quitting")
    quit()

cog.eval()
vqvae.eval()

pygame.init()
screen = pygame.display.set_mode((640, 320))
clock = pygame.time.Clock()
wsquare = pygame.Surface((10, 10))
wsquare.fill((255, 255, 255))
bsquare = pygame.Surface((10, 10))
bsquare.fill((0, 0, 0))

rmovement = 0.0

running = True

'''
sequences = torch.load("pongdata.pt")
ctx_frames = sequences[0]["frames"][0:5]
print(ctx_frames.shape)
mu, lv = model.vae.encoder.encode(ctx_frames.unsqueeze(1))
print(mu.shape)
prev_lat_frames = mu.unsqueeze(0)
'''
prev_lat_frame = vqvae.encode(torch.zeros(1, 1, 32, 64)).long()

hid = torch.zeros(1, cog.hidden_size)

while running:
    rmovement = 0
    lmovement = 0

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
    elif keys[pygame.K_w]:
        lmovement = -1.0
    elif keys[pygame.K_r]:
        lmovement = 1.0

    rmt = torch.Tensor([[lmovement, rmovement]]).to(device)
    print(rmt)
    lat, hid = cog.forward(rmt, prev_lat_frame, hid)
    prev_lat_frame = lat.clone().detach()
    framebuf = torch.round(vqvae.decode(lat).squeeze().squeeze())

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

    clock.tick(20)
