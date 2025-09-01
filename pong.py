import pygame
import torch
import random
import math

framebuf = torch.zeros((32, 64))
running = True

class Padel:
    def __init__(self, py, px, size):
        self.py = py
        self.px = px
        self.size = size
        self.vy = 0

    def update(self, direction):
        self.draw(0)
        self.py += direction
        self.vy = direction
        if self.py < 0:
            self.py = 0
        if (32 - self.size) < self.py:
            self.py = (32 - self.size)
        self.draw(1)

    def draw(self, color):
        for i in range(self.py, self.py + self.size):
            framebuf[i][self.px] = color

leftpadel = Padel(random.randint(0, 27), 2, 5)
rightpadel = Padel(random.randint(0, 27), 61, 5)

leftpadel.draw(1)
rightpadel.draw(1)

class Ball:
    def __init__(self):
        self.py = random.randint(16, 17)
        self.px = random.randint(32, 33)
        angle = random.randint(-60, 60) * (math.pi / 180)
        speed = 1
        self.vx = random.choice([-1, 1]) * speed * math.cos(angle)
        self.vy = speed * math.sin(angle)

        while (0.8 < self.vx) or (0.8 < self.vy) or self.vx == 0 or self.vy == 0:
            self.vx = random.choice([-1, 1]) * speed * math.cos(angle)
            self.vy = speed * math.sin(angle)

        self.ey = 0
        self.ex = 0

    def inline_padel(self, padel):
        return padel.py <= self.py and self.py <= (padel.py + padel.size - 1)

    def update(self):
        global running

        if self.inline_padel(leftpadel) and (leftpadel.px + 1) == round(self.px):
            self.vx = -self.vx
            if leftpadel.vy != 0:
                self.vy = max(-0.8, min(0.8, (leftpadel.vy * 0.3 + self.vy)))
                self.vx = max(-0.8, min(0.8, math.sqrt(1 - self.vy**2)))

        if self.inline_padel(rightpadel) and (rightpadel.px - 1) == round(self.px):
            self.vx = -self.vx
            if rightpadel.vy != 0:
                self.vy = max(-0.8, min(0.8, (rightpadel.vy * 0.3 + self.vy)))
                self.vx = max(-0.8, min(0.8, math.sqrt(1 - self.vy**2)))

        self.ey += abs(self.vy)
        self.ex += abs(self.vx)

        if (self.ey > 1.0):
            self.py += 1 if 0 < self.vy else -1
            self.ey -= 1.0

        if (self.ex > 1.0):
            self.px += 1 if 0 < self.vx else -1
            self.ex -= 1.0
        
        if self.py >= 31: 
            self.py = 31
            self.vy = -self.vy
        if self.py <= 0:
            self.py = 0
            self.vy = -self.vy
            
        if self.px >= 63:
            self.px = 63
            self.vx = -self.vx
            running = False
        if self.px <= 0:
            self.px = 0
            self.vx = -self.vx
            running = False

ball = Ball()
ERROR_CHANCE = 0.5

def get_ai_direction(padel):
    if (random.random() < 0.3):
        return 0
    if (random.random() < 0.1):
        return random.choice([-1, 0, 1])

    t = ball.py + random.uniform(-ERROR_CHANCE, ERROR_CHANCE)
    if ball.inline_padel(padel):
        return 0
    elif ball.py < padel.py:
        return -1
    else:
        return 1

def update():
    ldir = get_ai_direction(leftpadel)
    rdir = get_ai_direction(rightpadel)
    leftpadel.update(ldir)
    rightpadel.update(rdir)
    print(ldir)
    print(rdir)

    if (ball.inline_padel(leftpadel) or ball.inline_padel(rightpadel)) and (ball.px == leftpadel.px) or (ball.px == rightpadel.px):
        framebuf[ball.py][ball.px] = 1
    else:
        framebuf[ball.py][ball.px] = 0
    ball.update()
    framebuf[ball.py][ball.px] = 1

pygame.init()
screen = pygame.display.set_mode((640, 320))
clock = pygame.time.Clock()
wsquare = pygame.Surface((10, 10))
wsquare.fill((255, 255, 255))
bsquare = pygame.Surface((10, 10))
bsquare.fill((0, 0, 0))

def start():
    global framebuf, ball, leftpadel, rightpadel, running
    framebuf = torch.zeros((32, 64))
    ball = Ball()
    leftpadel = Padel(random.randint(0, 27), 2, 5)
    rightpadel = Padel(random.randint(0, 27), 61, 5)
    leftpadel.draw(1)
    rightpadel.draw(1)
    running = True

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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

        update()

        clock.tick(30)  # limits FPS to 60

if __name__ == '__main__':
    start()

pygame.quit()

