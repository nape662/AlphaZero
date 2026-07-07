"""The AlphaZero network: state in, (move probabilities, value) out.

A small CNN. Convolutions matter here: a "three in a row plus an empty
cell" is the same local pattern anywhere on the board, and a conv filter
learns it once and applies it at every offset — an MLP would have to
re-memorize it per position (tried; it didn't generalize). Three 3x3
layers give a 7-cell receptive field, enough to see any 4-line whole.

Heads as in the paper:
  - policy: 1x1 conv -> linear -> 7 logits -> softmax
  - value:  1x1 conv -> linear -> scalar -> tanh, in [-1, 1] for the
            player to move
"""

import torch
import torch.nn as nn

from connect4 import ROWS, COLS, NUM_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(state):
    """Bitboards -> two 6x7 planes: current player's pieces, opponent's."""
    current, opponent = state
    planes = torch.zeros(2, ROWS, COLS)
    for col in range(COLS):
        for row in range(ROWS):
            bit = 1 << (7 * col + row)
            if current & bit:
                planes[0, row, col] = 1.0
            elif opponent & bit:
                planes[1, row, col] = 1.0
    return planes


class PolicyValueNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(ROWS * COLS, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        h = self.trunk(x)
        probs = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h))
        if single:
            probs, value = probs.squeeze(0), value.squeeze(0)
        return probs, value
