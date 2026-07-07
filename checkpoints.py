"""Play with saved checkpoints.

  python checkpoints.py A.pt B.pt    checkpoint match, 50 games, mcts50
  python checkpoints.py A.pt random  same, vs a random mover
  python checkpoints.py A.pt         you vs the checkpoint in the terminal
"""

import sys

import numpy as np
import torch

from connect4 import INITIAL, legal_moves, play, render, terminal_value
from mcts import mcts
from net import DEVICE, PolicyValueNet
from selfplay import EVAL_SIMS, pit


def load(path):
    net = PolicyValueNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


def human_vs(net, sims=200):
    human_first = input("go first? [y/n] ").strip().lower() != "n"
    state, ply = INITIAL, 0
    while terminal_value(state) is None:
        mark = "X" if ply % 2 == 0 else "O"
        print("\n" + render(state, mark))
        if (ply % 2 == 0) == human_first:
            col = int(input(f"you ({mark}), column? "))
            if col not in legal_moves(state):
                print("illegal move")
                continue
        else:
            col = int(np.argmax(mcts(state, net, sims, noise=False)))
            print(f"net ({mark}) plays {col}")
        state = play(state, col)
        ply += 1
    print("\n" + render(state, "X" if ply % 2 == 0 else "O"))
    if terminal_value(state) == 0:
        print("draw")
    else:  # the player to move lost; they are the human iff human moves next
        print("you lose" if (ply % 2 == 0) == human_first else "you win")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        human_vs(load(args[0]))
    else:
        a = load(args[0])
        b = None if args[1] == "random" else load(args[1])
        w, d, l = pit(a, b, n_pairs=25, sims=EVAL_SIMS)
        print(f"{args[0]} vs {args[1]}: {w}-{d}-{l} (W-D-L)")
