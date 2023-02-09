import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from ..environment import BaseEnvironment


class SimpleCNN(nn.Module):
    def __init__(self, n, L):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32 x L/2 x L/2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32 x L/4 x L/4
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32 x L/8 x L/8
            nn.Flatten(),
            nn.Linear(32 * L // 8 * L // 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head_p = nn.Linear(256, L * L * 4)
        self.head_v = nn.Linear(256, 1)

    def forward(self, x, hidden=None):
        h = self.network(x)
        h_p = self.head_p(h)
        h_v = self.head_v(h)
        return {"policy": h_p, "value": torch.sigmoid(h_v)}


class Environment(BaseEnvironment):
    # L = 10000
    # L = 100
    L = 32
    # L = 50
    Q_MAX = L * L
    DIRECT = "ULDR"
    DIRECT_TO_INT = {"U": 0, "L": 1, "D": 2, "R": 3}

    def __init__(self, args=None):
        super().__init__()
        self.reset(args)

    def reset(self, args=None):
        # self.N = round(50 * (4 ** np.random.rand()))
        # self.N = 50
        self.N = 16
        # self.N = 10
        # self.N = 5
        all_points = [(i, j) for i in range(self.L) for j in range(self.L)]
        self.XY = random.sample(all_points, self.N)
        q = (
            [0]
            + sorted(1 + np.random.choice(self.Q_MAX - 1, self.N - 1, replace=False))
            + [self.Q_MAX]
        )
        self.R = np.diff(q)

        self.record = []
        # 1x1サイズから開始
        self.rects = np.array(
            [
                [self.XY[i][0], self.XY[i][1], self.XY[i][0] + 1, self.XY[i][1] + 1]
                for i in range(self.N)
            ],
            dtype=np.int32,
        )
        self.scores = np.array(
            [
                1 - (1 - min(self.R[i], 1) / max(self.R[i], 1)) ** 2
                for i in range(self.N)
            ]
        )
        self.board = np.zeros((self.L, self.L), dtype=bool)
        for a, b, c, d in self.rects:
            self.board[a:c, b:d] = True

        self.board2id = -np.ones((self.L, self.L), dtype=int)
        for i, (x, y) in enumerate(self.XY):
            self.board2id[x, y] = i

    def action2str(self, a, _=None):
        return f"{a//4}{self.DIRECT[a%4]}"

    def str2action(self, s, _=None):
        return int(s[:-1]) * 4 + self.DIRECT_TO_INT[s[-1]]

    def record_string(self):
        return " ".join([self.action2str(a) for a in self.record])

    def __str__(self):
        return str(self.rects)

    @staticmethod
    def __calc_score(a, b, c, d, r):
        s = (c - a) * (d - b)
        return 1 - (1 - min(r, s) / max(r, s)) ** 2

    def calc_score(self, i):
        a, b, c, d = self.rects[i]
        return Environment.__calc_score(a, b, c, d, self.R[i])

    def play(self, action, _=None):
        # state transition function
        pos, v = action // 4, action % 4
        i = self.board2id[pos // self.L, pos % self.L]
        assert i >= 0
        a, b, c, d = self.rects[i]
        if v == 0:
            self.board[(a - 1) : a, b:d] = True
            self.rects[i, v] -= 1
        elif v == 1:
            self.board[a:c, (b - 1) : b] = True
            self.rects[i, v] -= 1
        elif v == 2:
            self.board[c : (c + 1), b:d] = True
            self.rects[i, v] += 1
        elif v == 3:
            self.board[a:c, d : (d + 1)] = True
            self.rects[i, v] += 1
        else:
            raise

        self.scores[i] = self.calc_score(i)
        self.record.append(action)

    def diff_info(self, _):
        if len(self.record) == 0:
            return ""
        return self.action2str(self.record[-1])

    def update(self, info, reset):
        if reset:
            self.reset()
        else:
            action = self.str2action(info)
            self.play(action)

    def turn(self):
        return self.players()[0]

    def terminal(self):
        # check whether the state is terminal
        return len(self.legal_actions()) == 0

    # def reward(self):
    #     return {0: np.sum(self.scores) / self.N}

    def outcome(self):
        # terminal outcome
        # return {0: 1e9 * sum(self.scores) / self.N}
        return {0: np.sum(self.scores) / self.N}

    def legal_actions(self, _=None):
        # legal action list
        return [
            (i * self.L + j) * 4 + v
            for i in range(self.L)
            for j in range(self.L)
            for v in range(4)
            if self.action_check(self.board2id[i, j], v)
        ]

    def action_check(self, i, v):
        if i < 0:
            return False
        a, b, c, d = self.rects[i]
        if v == 0:
            return (
                a - 1 >= 0
                and not self.board[(a - 1) : a, b:d].any()
                and Environment.__calc_score(a - 1, b, c, d, self.R[i]) > self.scores[i]
            )
        elif v == 1:
            return (
                b - 1 >= 0
                and not self.board[a:c, (b - 1) : b].any()
                and Environment.__calc_score(a, b - 1, c, d, self.R[i]) > self.scores[i]
            )
        elif v == 2:
            return (
                c + 1 <= self.L
                and not self.board[c : (c + 1), b:d].any()
                and Environment.__calc_score(a, b, c + 1, d, self.R[i]) > self.scores[i]
            )
        elif v == 3:
            return (
                d + 1 <= self.L
                and not self.board[a:c, d : (d + 1)].any()
                and Environment.__calc_score(a, b, c, d + 1, self.R[i]) > self.scores[i]
            )
        else:
            raise

    def players(self):
        return [0]

    def net(self):
        return SimpleCNN(self.N, self.L)

    def observation(self, player=None):
        # input feature for neural nets
        board_score = np.zeros((self.L, self.L), dtype=np.float32)
        for (x, y), s in zip(self.XY, self.scores):
            board_score[x, y] = s
        board_position = np.zeros((self.L, self.L), dtype=np.float32)
        for (x, y), r in zip(self.XY, self.R):
            board_position[x, y] = r
        return np.stack([self.board, board_score, board_position])

    def print_input(self):
        print(self.N)
        for i in range(self.N):
            print(self.XY[i][0], self.XY[i][1], self.R[i])

    def print_output(self):
        for a, b, c, d in self.rects:
            print(a, b, c, d)

    def draw(self, filename):
        plt.clf()
        ax = plt.axes()

        for i, (a, b, c, d) in enumerate(self.rects):
            r = patches.Rectangle(
                xy=(a, b), width=c - a, height=d - b, fc="c", ec="k", fill=True
            )
            ax.add_patch(r)
            plt.text(a, b, f"{self.scores[i]:.4f}")
        for i in range(self.N):
            c = patches.Circle(xy=self.XY[i], radius=1, fc="r")
            ax.add_patch(c)

        plt.axis("scaled")
        ax.set_aspect("equal")
        plt.title(f"{self.outcome()[0]}")

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)


if __name__ == "__main__":
    e = Environment()
    scores = []
    for i in range(100):
        e.reset()
        while not e.terminal():
            # print(e)
            actions = e.legal_actions()
            # print([e.action2str(a) for a in actions])
            e.play(random.choice(actions))
        print("input")
        e.print_input()
        print("output")
        e.print_output()
        print(e.outcome()[0])
        e.draw(f"random/ahc{i}.png")
        scores.append(e.outcome()[0])
        print("mean:", np.mean(scores))
