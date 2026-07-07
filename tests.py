"""Run with: python tests.py (or pytest tests.py)."""

import numpy as np
import torch

from connect4 import FULL, INITIAL, has_won, legal_moves, play, terminal_value
from mcts import mcts
from net import DEVICE, PolicyValueNet


def bit(col, row):
    return 1 << (7 * col + row)


def state_after(cols):
    s = INITIAL
    for c in cols:
        s = play(s, c)
    return s


def test_has_won_directions():
    assert has_won(bit(3, 0) | bit(3, 1) | bit(3, 2) | bit(3, 3))  # |
    assert has_won(bit(1, 2) | bit(2, 2) | bit(3, 2) | bit(4, 2))  # -
    assert has_won(bit(0, 0) | bit(1, 1) | bit(2, 2) | bit(3, 3))  # /
    assert has_won(bit(0, 3) | bit(1, 2) | bit(2, 1) | bit(3, 0))  # \
    assert not has_won(bit(1, 2) | bit(2, 2) | bit(3, 2))  # only 3


def test_no_wrap_across_columns():
    # top of col 0 + bottom of col 1: bit-adjacent, geometrically nonsense
    assert not has_won(bit(0, 3) | bit(0, 4) | bit(0, 5) | bit(1, 0))
    assert not has_won(bit(0, 5) | bit(1, 5) | bit(2, 5) | bit(3, 0))


def test_games():
    # vertical win: player to move (O) sees a loss
    assert terminal_value(state_after([0, 1, 0, 1, 0, 1, 0])) == -1
    # horizontal win
    assert terminal_value(state_after([0, 6, 1, 6, 2, 6, 3])) == -1
    # ongoing game
    assert terminal_value(state_after([0, 1, 2])) is None


def test_full_column_illegal():
    s = state_after([5, 5, 5, 5, 5, 5])
    assert legal_moves(s) == [0, 1, 2, 3, 4, 6]


def test_draw():
    # full board, alternating-row pattern, no 4-in-a-row anywhere
    grid = ["XXOOXXO", "OOXXOOX"] * 3
    x = o = 0
    for row, line in enumerate(grid):
        for col, ch in enumerate(line):
            if ch == "X":
                x |= bit(col, row)
            else:
                o |= bit(col, row)
    assert (x | o) == FULL
    assert not has_won(x) and not has_won(o)
    assert terminal_value((x, o)) == 0
    assert legal_moves((x, o)) == []


def _net():
    torch.manual_seed(0)
    np.random.seed(0)
    return PolicyValueNet().to(DEVICE)


def test_mcts_pi_is_distribution():
    pi = mcts(INITIAL, _net(), sims=100)
    assert abs(pi.sum() - 1) < 1e-9
    assert (pi >= 0).all()


def test_mcts_masks_full_column():
    pi = mcts(state_after([5, 5, 5, 5, 5, 5]), _net(), sims=100)
    assert pi[5] == 0


def test_mcts_finds_win_in_1():
    # X on bottom of cols 1,2,3 — col 4 wins on the spot; an untrained
    # net knows nothing, so this is terminal backup + signs at work
    pi = mcts(state_after([1, 0, 2, 0, 3]), _net(), sims=200)
    assert np.argmax(pi) == 4, pi


def test_mcts_blocks_win_in_1():
    # X has 3 stacked in col 0, O to move: anything but 0 loses next ply
    pi = mcts(state_after([0, 6, 0, 6, 0]), _net(), sims=400)
    assert np.argmax(pi) == 0, pi


def test_selfplay_z_perspective():
    from selfplay import play_games

    states, pis, zs = play_games(_net(), 1, sims=25, parallel=1)[0]
    assert len(states) == len(pis) == len(zs)
    if zs[-1] != 0:
        # the last player to move can only have won ("you can't lose
        # via your own move"), and perspectives alternate every ply
        assert zs[-1] == 1
        assert all(zs[i] == -zs[i + 1] for i in range(len(zs) - 1))
    else:
        assert all(z == 0 for z in zs)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
