"""The AlphaZero network: state in, (move probabilities, value) out.

Bare minimum: an MLP with two hidden layers in fp32. The shared trunk feeds
two heads:
  - policy: 7 logits -> softmax = probability of dropping in each column
  - value:  1 scalar -> tanh    = expected outcome in [-1, 1] for the player
            to move (unused until MCTS, but the training loss couples both
            heads, so the architecture carries it from the start)
"""

import torch
import torch.nn as nn

from connect4 import ROWS, COLS, NUM_ACTIONS


def encode(state):
    """Bitboards -> 42 floats: +1 current player's piece, -1 opponent's, 0 empty.

    The net can't do bit operations, so each playable cell becomes its own
    input feature. Ternary in one vector (rather than two binary planes)
    keeps similar positions numerically close, and gradients flow fine.
    Order: column by column, bottom to top — any fixed order works, it just
    has to be consistent.
    """
    current, opponent = state
    vals = []
    for col in range(COLS):
        for row in range(ROWS):
            bit = 1 << (7 * col + row)
            vals.append(1.0 if current & bit else -1.0 if opponent & bit else 0.0)
    return torch.tensor(vals)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(ROWS * COLS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        probs = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h))
        return probs, value
