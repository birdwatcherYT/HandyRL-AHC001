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


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(
            filters0,
            filters1,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Head(nn.Module):
    def __init__(self, input_size, out_filters, outputs):
        super().__init__()

        self.board_size = input_size[1] * input_size[2]
        self.out_filters = out_filters

        self.conv = Conv(input_size[0], out_filters, 1, bn=False)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(self.board_size * out_filters, outputs, bias=False)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = self.fc(h.view(-1, self.board_size * self.out_filters))
        return h


class SimpleConv2dModel(nn.Module):
    def __init__(self, L):
        super().__init__()
        layers, filters = 3, 32

        self.conv = nn.Conv2d(3, filters, 3, stride=1, padding=1)
        self.blocks = nn.ModuleList(
            [Conv(filters, filters, 3, bn=True) for _ in range(layers)]
        )
        self.head_p = Head((filters, 3, 3), 2, L**4)  # サイズが大きすぎるためactionを考え直す必要あり
        self.head_v = Head((filters, 3, 3), 1, 1)

    def forward(self, x, hidden=None):
        h = F.relu(self.conv(x))
        for block in self.blocks:
            h = F.relu(block(h))
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return {"policy": h_p, "value": torch.tanh(h_v)}


class Environment(BaseEnvironment):
    # L = 10000
    L = 100
    Q_MAX = L * L

    def __init__(self, args=None):
        super().__init__()
        self.reset(args)

    def reset(self, args=None):
        # self.N = round(50 * (4 ** np.random.rand()))
        # self.N = 50
        self.N = 10
        all_points = [(i, j) for i in range(self.L) for j in range(self.L)]
        self.XY = random.sample(all_points, self.N)
        self.XY = np.array(self.XY, dtype=int)
        q = (
            [0]
            + sorted(1 + np.random.choice(self.Q_MAX - 1, self.N - 1, replace=False))
            + [self.Q_MAX]
        )
        self.R = np.diff(q)

        self.record = []
        self.rects = []
        self.scores = []
        self.board = np.zeros((self.L, self.L), dtype=bool)

    def action2str(self, act, _=None):
        L1 = self.L + 1
        L2 = L1 * L1
        ab = act % L2
        cd = act // L2
        a = ab % L1
        b = ab // L1
        c = cd % L1
        d = cd // L1
        return f"{a},{b},{c},{d}"

    def str2action(self, s, _=None):
        a, b, c, d = map(int, s.split(","))
        L1 = self.L + 1
        L2 = L1 * L1
        return (a + b * L1) + (c + d * L1) * L2

    def record_string(self):
        return " ".join([self.action2str(a) for a in self.record])

    def __str__(self):
        return str(self.rects)

    def calc_score(self, i):
        a, b, c, d = self.rects[i]
        s = (c - a) * (d - b)
        return 1 - (1 - min(self.R[i], s) / max(self.R[i], s)) ** 2

    def play(self, act, _=None):
        # state transition function
        L1 = self.L + 1
        L2 = L1 * L1
        ab = act % L2
        cd = act // L2
        a = ab % L1
        b = ab // L1
        c = cd % L1
        d = cd // L1
        #
        i = len(self.rects)
        self.board[a:c, b:d] = True
        self.rects.append((a, b, c, d))
        self.scores.append(self.calc_score(i))
        self.record.append(act)

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
        # return len(self.legal_actions()) == 0
        return len(self.rects) == self.N

    def outcome(self):
        # terminal outcome
        # return {0: 1e9 * sum(self.scores) / self.N}
        return {0: np.sum(self.scores) / self.N}

    def legal_actions(self, _=None):
        # legal action list
        i = len(self.rects)
        x, y = self.XY[i]
        L1 = self.L + 1
        L2 = L1 * L1
        return [
            (a + b * L1) + (c + d * L1) * L2
            for a in range(x + 1)
            for b in range(y + 1)
            for c in range(x + 1, L1)
            for d in range(y + 1, L1)
            if self.action_check(i, a, b, c, d)
        ]

    def action_check(self, i, a, b, c, d):
        x, y = self.XY[i]
        return (a <= x <= c and b <= y <= d) and (not self.board[a:c, b:d].any())

    def players(self):
        return [0]

    def net(self):
        return SimpleConv2dModel(self.L)

    def observation(self, player=None):
        # input feature for neural nets
        board_R = np.zeros((self.L, self.L), dtype=np.float32)
        for (x, y), r in zip(self.XY, self.R):
            board_R[x, y] = r
        # 次に決めたい長方形
        next_point = np.zeros((self.L, self.L), dtype=np.float32)
        i = len(self.rects)
        x, y = self.XY[i]
        next_point[x, y] = 1
        return np.stack([self.board, board_R / self.Q_MAX, next_point])

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
