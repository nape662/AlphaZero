"""Self-play loop: the (untrained) net plays a game against itself.

This is the IO-level milestone: state -> net -> sampled column -> play ->
repeat until terminal. No search, no training — just proof the plumbing works.
"""

import torch

from connect4 import INITIAL, legal_moves, play, terminal_value, render
from net import PolicyValueNet, encode


def play_game(net):
    """Play one full game, return (final state, list of columns played)."""
    state = INITIAL
    log = []
    while terminal_value(state) is None:
        with torch.no_grad():  # inference only, no gradients needed
            probs, _value = net(encode(state))
        # Zero out full columns and renormalize, so sampling can't pick an
        # illegal move (resampling instead could spin forever on a confident
        # net whose favorite column is full).
        mask = torch.zeros_like(probs)
        mask[legal_moves(state)] = 1.0
        probs = probs * mask
        col = torch.multinomial(probs / probs.sum(), 1).item()
        log.append(col)
        state = play(state, col)
    return state, log


if __name__ == "__main__":
    net = PolicyValueNet()
    state, log = play_game(net)

    # X moved first, so after an even number of moves it's X's turn again.
    to_move = "X" if len(log) % 2 == 0 else "O"
    print(render(state, to_move))
    print("moves:", log)
    if terminal_value(state) == 0:
        print("result: draw")
    else:
        # terminal_value == -1: the player who just moved (not to_move) won.
        print("result:", "O" if to_move == "X" else "X", "wins after", len(log), "moves")
