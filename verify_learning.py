"""Definitive verification suite for a finished training run.

Usage: python verify_learning.py <run_name>
Tests, in order of what they prove:
  V1 raw net vs random        - real strength without search
  V2 checkpoint ladder        - strength compounds over training
  V3 on-distribution tactics  - net masters its own game positions
  V4 off-distribution tactics - generalization to random-play positions
Verdicts printed at the end.
"""
import sys
sys.path.insert(0, "/home/nape662/Coding/AlphaZero")

import pathlib

import numpy as np
import torch

from connect4 import legal_moves, play, terminal_value
from checkpoints import load
from selfplay import make_probes, probe_accuracy, play_games, pit

run = sys.argv[1] if len(sys.argv) > 1 else "v2"
ckpts = sorted(pathlib.Path(f"/home/nape662/Coding/AlphaZero/checkpoints/{run}").glob("*.pt"))
early, mid, final = ckpts[0], ckpts[len(ckpts) // 2], ckpts[-1]
print(f"run {run}: early={early.name} mid={mid.name} final={final.name}\n")
net_e, net_m, net_f = load(early), load(mid), load(final)

# V1: raw net (no search) vs random, 40 games
w, d, l = pit(net_f, None, n_pairs=20, sims=0)
print(f"V1 raw-final vs random: {w}-{d}-{l}  ({w / (w + d + l):.0%} wins)")
v1 = w / (w + d + l) >= 0.85

# V2: checkpoint ladder with search (mcts50, noise off), 20 games each
fm = pit(net_f, net_m, n_pairs=10, sims=50)
me = pit(net_m, net_e, n_pairs=10, sims=50)
fe = pit(net_f, net_e, n_pairs=10, sims=50)
print(f"V2 ladder (mcts50): final vs mid {fm}, mid vs early {me}, final vs early {fe}")
v2 = fm[0] > fm[2] and me[0] > me[2] and fe[0] >= 2 * fe[2]

# V3: win-in-1 accuracy on positions from the final net's own games
games = play_games(net_f, 64, sims=200)
own = []
for ss, _, _ in games:
    for s in ss:
        wins = [a for a in legal_moves(s) if terminal_value(play(s, a)) == -1]
        if wins:
            own.append((s, wins))
on_acc = probe_accuracy(net_f, own) if own else float("nan")
print(f"V3 on-distribution win-in-1: {on_acc:.0%} ({len(own)} positions)")
v3 = on_acc >= 0.7

# V4: the fixed random-play probes, early vs final (trajectory)
win_p, block_p = make_probes()
for name, n in (("early", net_e), ("final", net_f)):
    print(f"V4 {name}: win-in-1 {probe_accuracy(n, win_p):.0%}, "
          f"block-in-1 {probe_accuracy(n, block_p):.0%}")
v4 = probe_accuracy(net_f, win_p) >= 0.55

print()
for name, ok in (("V1 raw beats random", v1), ("V2 ladder climbs", v2),
                 ("V3 on-dist tactics", v3), ("V4 generalizes", v4)):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
