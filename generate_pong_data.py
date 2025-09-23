import torch
import random
import math
import time

framebuf = torch.zeros((32, 64))
running = True

class Padel:
    def __init__(self, py, px, size):
        self.py = py
        self.px = px
        self.size = size
        self.vy = 0
        self.vdecay = 0.7

    def update(self, direction):
        self.draw(0)
        self.py += direction
        if direction == 0:
            self.vy = self.vy * self.vdecay
            if abs(self.vy) < 0.2:
                self.vy = 0
        else:
            self.vy = direction
            self.vdecay = 0.7

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
            angle = random.randint(-60, 60) * (math.pi / 180)
            self.vx = random.choice([-1, 1]) * speed * math.cos(angle)
            self.vy = speed * math.sin(angle)

        self.ey = 0
        self.ex = 0

    def inline_padel(self, padel, extend=0):
        return padel.py - extend <= self.py and self.py <= (padel.py + padel.size - 1) + extend

    def update(self):
        global running

        old_py = self.py
        old_px = self.px

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
            framebuf[self.py][self.px] = 0
            framebuf[old_py][old_px] = 0
            return 1
        if self.px <= 0:
            self.px = 0
            self.vx = -self.vx
            framebuf[self.py][self.px] = 0
            framebuf[old_py][old_px] = 0
            return 2

        if self.inline_padel(leftpadel, extend=1) and self.px <= (leftpadel.px + 1) and self.vx < 0 and leftpadel.px <= self.px:
            self.vx = abs(self.vx)
            self.px = leftpadel.px + 1
            if leftpadel.vy != 0:
                self.vy = max(-0.8, min(0.8, (leftpadel.vy * 0.3 + self.vy)))
                speed = math.sqrt(self.vx**2 + self.vy**2)
                if speed > 0:
                    self.vx = (self.vx / speed) * 1.0
                    self.vy = (self.vy / speed) * 1.0

        if self.inline_padel(rightpadel, extend=1) and (rightpadel.px - 1) <= self.px and 0 < self.vx and self.px <= rightpadel.px:
            self.vx = -abs(self.vx)
            self.px = rightpadel.px - 1
            if rightpadel.vy != 0:
                self.vy = max(-0.8, min(0.8, (rightpadel.vy * 0.3 + self.vy)))
                speed = math.sqrt(self.vx**2 + self.vy**2)
                if speed > 0:
                    self.vx = (self.vx / speed) * 1.0
                    self.vy = (self.vy / speed) * 1.0

        return 0

ball = Ball()
ERROR_CHANCE = 0.3

def get_ai_direction(padel):
    if (random.random() < 0.3):
        return 0
    if (random.random() < 0.5):
        return random.choice([-1, 0, 1])

    t = ball.py + random.uniform(-ERROR_CHANCE, ERROR_CHANCE)
    if ball.inline_padel(padel):
        return 0
    elif ball.py < padel.py:
        return -1
    else:
        return 1

def update(ldir, rdir):
    leftpadel.update(ldir)
    rightpadel.update(rdir)

    if (ball.inline_padel(leftpadel) and (ball.px == leftpadel.px)) or (ball.inline_padel(rightpadel) and (ball.px == rightpadel.px)):
        framebuf[ball.py][ball.px] = 1
    else:
        framebuf[ball.py][ball.px] = 0

    r = ball.update()
    if r == 0:
        framebuf[ball.py][ball.px] = 1
    else:
        return r

    return 0

class Recorder:
    def __init__(self):
        self.current_sequence = []
        self.sequences = []
        self.sequence_max_frames = 50
        self.current_sequence_frame_count = 0
        self.all_frames = 0
        self.sequences_with_results = 0
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def record_frame(self, frame, inputleft, inputright, result):
        global running

        res = torch.zeros(3)
        res[result] = 1.0

        inputs = torch.Tensor([inputleft, inputright])

        self.current_sequence.append({
            "frame": frame.clone().detach(),
            "inputs": inputs,
            "result": res 
        })

        self.current_sequence_frame_count += 1

        if result != 0 or self.sequence_max_frames < len(self.current_sequence):
            if len(self.current_sequence) < 5:
                self.current_sequence = []
                return

            if result != 0:
                self.sequences_with_results += 1

            self.all_frames += self.current_sequence_frame_count
            self.current_sequence_frame_count = 0

            self.sequences.append({
                "frames": torch.stack([s["frame"] for s in self.current_sequence]),
                "actions": torch.stack([s["inputs"] for s in self.current_sequence]),
                "results": torch.stack([s["result"] for s in self.current_sequence])
            })

            self.current_sequence = []
            running = False

    def save_dataset(self, filename="pongdata.pt"):
        torch.save(self.sequences, filename)       

recorder = Recorder()

def start():
    global framebuf, ball, leftpadel, rightpadel, running, recorder
    framebuf = torch.zeros((32, 64))
    ball = Ball()
    leftpadel = Padel(14, 2, 5)
    rightpadel = Padel(14, 61, 5)
    leftpadel.draw(1)
    rightpadel.draw(1)
    framebuf[ball.py][ball.px] = 1
    running = True
    r = 0

    ldir = 0
    rdir = 0

    while running:
        recorder.record_frame(framebuf, ldir, rdir, r)

        ldir = get_ai_direction(leftpadel)
        rdir = get_ai_direction(rightpadel)

        r = update(ldir, rdir)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser("python3 generate_pong_data.py")
    parser.add_argument("frames", help="The amount of frames the script should generate.", nargs='?', type=int, const=150000, default=150000)
    args = parser.parse_args()
    print(f"Generating {args.frames} frames of Pong data")
    while recorder.all_frames < args.frames:
        start()
        if len(recorder.sequences) % 20 == 0:
            print("new game; sequences:", str(len(recorder.sequences)) + "; all frames:", str(recorder.all_frames) + "; seq with res:", recorder.sequences_with_results)
    recorder.save_dataset()

