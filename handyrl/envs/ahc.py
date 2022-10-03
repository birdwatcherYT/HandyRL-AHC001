# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Tic-Tac-Toe

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import BaseEnvironment


class SimpleFCModel(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4 * n, n)
        self.linear2 = nn.Linear(n, 64)

        self.head_p = nn.Linear(64, n * 4)
        self.head_v = nn.Linear(64, 1)

    def forward(self, x, hidden=None):
        h = self.flatten(x)
        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        h_p = self.head_p(h)
        h_v = self.head_v(h)

        return {"policy": h_p, "value": torch.tanh(h_v)}


class Environment(BaseEnvironment):
    L = 10000
    Q_MAX = 100000000

    def __init__(self, args=None):
        super().__init__()
        self.reset(args)

    def reset(self, args=None):
        # self.N = round(50 * (4 ** np.random.rand()))
        self.N = 50
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
        self.rects = [
            [self.XY[i][0], self.XY[i][1], self.XY[i][0] + 1, self.XY[i][1] + 1]
            for i in range(self.N)
        ]
        self.scores = [
            1 - (1 - min(self.R[i], 1) / max(self.R[i], 1)) ** 2 for i in range(self.N)
        ]
        self.board = np.zeros((self.L, self.L), dtype=bool)
        for a, b, c, d in self.rects:
            self.board[a:c, b:d] = True

    def action2str(self, a, _=None):
        return f"{a[0]}{a[1]}"

    def str2action(self, s, _=None):
        return (int(s[:-1]), s[-1])

    def record_string(self):
        return " ".join([self.action2str(a) for a in self.record])

    def __str__(self):
        return str(self.rects)

    def calc_score(self, i, a, b, c, d):
        s = (c - a) * (d - b)
        return 1 - (1 - min(self.R[i], s) / max(self.R[i], s)) ** 2

    def play(self, action, _=None):
        # state transition function
        i, v = action // 4, action % 4
        a, b, c, d = self.rects[i]
        if v == 0:
            self.board[(a - 1) : a, b:d] = True
            self.rects[i][v] -= 1
        elif v == 1:
            self.board[a:c, (b - 1) : b] = True
            self.rects[i][v] -= 1
        elif v == 2:
            self.board[c : (c + 1), b:d] = True
            self.rects[i][v] += 1
        elif v == 3:
            self.board[a:c, d : (d + 1)] = True
            self.rects[i][v] += 1
        else:
            raise

        self.scores[i] = self.calc_score(i, a, b, c, d)
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

    def outcome(self):
        # terminal outcome
        return {0: 1e9 * sum(self.scores) / self.N}

    def legal_actions(self, _=None):
        # legal action list
        return [
            i * 4 + v
            for i in range(self.N)
            for v in range(4)
            if self.action_check(i, v)
        ]

    def action_check(self, i, v):
        a, b, c, d = self.rects[i]
        if v == 0:
            return a - 1 >= 0 and not self.board[(a - 1) : a, b:d].any()
        elif v == 1:
            return b - 1 >= 0 and not self.board[a:c, (b - 1) : b].any()
        elif v == 2:
            return c + 1 <= self.L and not self.board[c : (c + 1), b:d].any()
        elif v == 3:
            return d + 1 <= self.L and not self.board[a:c, d : (d + 1)].any()
        else:
            raise

    def players(self):
        return [0]

    def net(self):
        return SimpleFCModel(self.N)

    def observation(self, player=None):
        # input feature for neural nets
        a = np.array(self.rects).astype(np.float32)
        return a


if __name__ == "__main__":
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = e.legal_actions()
            print([e.action2str(a) for a in actions])
            e.play(random.choice(actions))
        print(e)
        print(e.outcome())
