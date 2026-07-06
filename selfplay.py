"""Self-play training: MCTS-driven games -> (state, pi, z) -> gradient steps.

Each game records the position, the search distribution pi, and (once the
game ends) the result z from that position's player-to-move perspective.
Loss per AlphaZero: (z - v)^2 - pi . log p, with L2 regularization via
the optimizer's weight_decay.
"""

import numpy as np
import torch

from connect4 import NUM_ACTIONS, INITIAL, play, terminal_value
from mcts import mcts
from net import PolicyValueNet, encode

TEMP_MOVES = 10  # sample from pi this many plies, then switch to argmax


def play_game(net, sims=800):
    """One self-play game. Returns (states, pis, zs), one entry per ply."""
    state = INITIAL
    states, pis = [], []
    while terminal_value(state) is None:
        pi = mcts(state, net, sims)
        states.append(state)
        pis.append(pi)
        if len(states) <= TEMP_MOVES:
            col = int(np.random.choice(NUM_ACTIONS, p=pi))
        else:
            col = int(np.argmax(pi))
        state = play(state, col)
    tv = terminal_value(state)
    # tv is from the final position's perspective; flip once per ply back
    zs = [tv * (-1) ** (len(states) - i) for i in range(len(states))]
    return states, pis, zs


def train(n_games=1000, sims=800, lr=1e-3):
    net = PolicyValueNet()
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    for game in range(n_games):
        states, pis, zs = play_game(net, sims)
        x = torch.stack([encode(s) for s in states])
        pi = torch.from_numpy(np.stack(pis)).float()
        z = torch.tensor(zs, dtype=torch.float32)
        p, v = net(x)
        loss = ((z - v.squeeze(-1)) ** 2).mean() \
            - (pi * torch.log(p.clamp(min=1e-8))).sum(dim=1).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"game {game}: {len(states)} moves, z[0]={zs[0]:+d}, loss {loss.item():.3f}")
    return net


if __name__ == "__main__":
    train(n_games=5, sims=100)  # smoke-test numbers; crank for a real run
