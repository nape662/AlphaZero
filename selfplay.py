"""Self-play training: MCTS-driven games -> (state, pi, z) -> gradient steps.

Games run `parallel` at a time in a single thread: each tick, every game
advances its search by one simulation up to the net call (descend), all the
resulting leaf positions go through the net as ONE batch, then each game
completes its simulation (set_prior + backprop). A game that finishes its
sims for the current move plays it; a game that ends is harvested for
training data and a fresh one takes its slot.

Each finished game yields per ply: the position, the search distribution pi,
and the result z from that position's player-to-move perspective.
Loss per AlphaZero: (z - v)^2 - pi . log p, with L2 regularization via the
optimizer's weight_decay. One gradient step per game played, each on a
random minibatch of the current wave's positions.

Every eval_every games the current net plays the snapshot from the previous
eval round: paired games from random 2-ply openings (each opening played
twice with colors swapped — Connect 4's first-player advantage is huge),
both as raw policy argmax and with a small MCTS, noise off.
"""

import copy
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import torch

from connect4 import NUM_ACTIONS, INITIAL, legal_moves, play, terminal_value
from mcts import Node, add_noise, backprop, descend, mcts, set_prior
from net import DEVICE, PolicyValueNet, encode

TEMP_MOVES = 10   # sample from pi this many plies, then switch to argmax
EVAL_PAIRS = 5    # openings per eval match; x2 colors = 10 games
EVAL_SIMS = 50
MINIBATCH = 128
BUFFER = 100000   # replay buffer: latest positions (after mirroring); large so
                  # early tactically-rich games keep being rehearsed as the
                  # net's own games grow cleaner (else it forgets the basics)
STEPS_PER_GAME = 16  # gradient steps per self-play game; generation is the
                     # wall-clock cost, so heavy optimization is nearly free


class Game:
    def __init__(self):
        self.root = Node(INITIAL)
        self.sims_done = 0
        self.states, self.pis = [], []


def play_games(net, n_games, sims, parallel=64):
    """Self-play n_games, up to `parallel` in flight, one net batch per tick.

    Returns a list of (states, pis, zs) triples, one per game.
    """
    finished, to_spawn, pool = [], n_games, []
    while pool or to_spawn:
        while to_spawn and len(pool) < parallel:
            pool.append(Game())
            to_spawn -= 1

        # 1) each game advances one simulation up to its net call
        requests = []
        for g in pool:
            if g.root.P is None:  # fresh root: must get priors before any sim
                requests.append((g, [], g.root))
                continue
            request = descend(g.root)
            g.sims_done += 1
            if request is not None:
                requests.append((g, *request))

        # 2) one batched net call for every leaf that asked
        if requests:
            x = torch.stack([encode(leaf.state) for _, _, leaf in requests])
            with torch.no_grad():
                probs, values = net(x.to(DEVICE))
            probs = probs.cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()
            for (g, path, leaf), p, v in zip(requests, probs, values):
                set_prior(leaf, p)
                if not path:  # it was a root: inject exploration noise
                    add_noise(leaf)
                backprop(path, float(v))

        # 3) games whose search is done play their move
        for g in pool[:]:
            if g.sims_done < sims:
                continue
            pi = g.root.N / g.root.N.sum()
            g.states.append(g.root.state)
            g.pis.append(pi)
            if len(g.states) <= TEMP_MOVES:
                col = int(np.random.choice(NUM_ACTIONS, p=pi))
            else:
                col = int(np.argmax(pi))
            state = play(g.root.state, col)
            tv = terminal_value(state)
            if tv is None:
                g.root = Node(state)
                g.sims_done = 0
            else:
                # tv is from the final position's perspective; flip per ply back
                zs = [tv * (-1) ** (len(g.states) - i) for i in range(len(g.states))]
                finished.append((g.states, g.pis, zs))
                pool.remove(g)
    return finished


def play_game(net, sims=800):
    """One self-play game. Returns (states, pis, zs), one entry per ply."""
    return play_games(net, 1, sims, parallel=1)[0]


_worker_net = None  # one lazily-built net per worker process


def _worker_play(args):
    """Runs in a spawned worker: rebuild the net once, load the latest
    weights, and self-play a chunk of the wave."""
    global _worker_net
    state_dict, n_games, sims, parallel = args
    if _worker_net is None:
        _worker_net = PolicyValueNet().to(DEVICE)
    _worker_net.load_state_dict(state_dict)
    return play_games(_worker_net, n_games, sims, parallel)


def pit(net_a, net_b, n_pairs=EVAL_PAIRS, sims=0):
    """Match net_a vs net_b, returns (a wins, draws, b wins).

    sims=0 plays the raw policy argmax; otherwise MCTS without noise.
    net_b=None plays uniformly random moves instead.
    """
    score = [0, 0, 0]
    for _ in range(n_pairs):
        opening = INITIAL
        for _ in range(2):
            opening = play(opening, int(np.random.choice(legal_moves(opening))))
        for a_to_move in (True, False):
            state = opening
            while terminal_value(state) is None:
                net = net_a if a_to_move else net_b
                if net is None:
                    col = int(np.random.choice(legal_moves(state)))
                elif sims:
                    col = int(np.argmax(mcts(state, net, sims, noise=False)))
                else:
                    with torch.no_grad():
                        probs, _ = net(encode(state).to(DEVICE))
                    legal = legal_moves(state)
                    col = legal[int(np.argmax(probs.cpu().numpy()[legal]))]
                state = play(state, col)
                a_to_move = not a_to_move
            if terminal_value(state) == 0:
                score[1] += 1
            else:  # player to move lost
                score[2 if a_to_move else 0] += 1
    return score


def make_probes(n=100, seed=0):
    """Tactical positions from random play, ground truth from the rules alone.

    win-in-1: some legal move ends the game in our favor (target: those moves).
    block-in-1: no win of ours, opponent threatens exactly one winning column
    (target: that column). Two or more threats = lost anyway, skipped.
    """
    rng = np.random.default_rng(seed)
    win_probes, block_probes = [], []
    while len(win_probes) < n or len(block_probes) < n:
        state = INITIAL
        while terminal_value(state) is None:
            legal = legal_moves(state)
            wins = [a for a in legal if terminal_value(play(state, a)) == -1]
            threats = [a for a in legal
                       if terminal_value(play((state[1], state[0]), a)) == -1]
            if wins and len(win_probes) < n:
                win_probes.append((state, wins))
            elif not wins and len(threats) == 1 and len(block_probes) < n:
                block_probes.append((state, threats))
            state = play(state, int(rng.choice(legal)))
    return win_probes, block_probes


def probe_accuracy(net, probes):
    """Fraction of probes where the RAW policy argmax is a target move."""
    x = torch.stack([encode(s) for s, _ in probes])
    with torch.no_grad():
        p, _ = net(x.to(DEVICE))
    p = p.cpu().numpy()
    hits = sum(legal_moves(s)[int(np.argmax(row[legal_moves(s)]))] in good
               for (s, good), row in zip(probes, p))
    return hits / len(probes)


def train(n_games=1000, sims=800, lr=1e-3, parallel=64, eval_every=128, run=None,
          workers=1):
    run = run or time.strftime("%m%d-%H%M%S")
    ckpt_dir = pathlib.Path("checkpoints") / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE} | run: {run}")
    net = PolicyValueNet().to(DEVICE)
    snapshot = copy.deepcopy(net)
    win_probes, block_probes = make_probes()
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
    t0, done = time.time(), 0
    buf_x = buf_pi = buf_z = None
    while done < n_games:
        if pool:
            wave = min(workers * parallel, n_games - done)
            per = [wave // workers + (i < wave % workers) for i in range(workers)]
            sd = {k: v.cpu() for k, v in net.state_dict().items()}
            chunks = pool.map(_worker_play,
                              [(sd, n, sims, parallel) for n in per if n])
            games = [g for chunk in chunks for g in chunk]
        else:
            wave = min(parallel, n_games - done)
            games = play_games(net, wave, sims, parallel)

        x = torch.stack([encode(s) for ss, _, _ in games for s in ss]).to(DEVICE)
        pi = torch.tensor(np.concatenate([ps for _, ps, _ in games]),
                          dtype=torch.float32, device=DEVICE)
        z = torch.tensor([z for _, _, zs in games for z in zs],
                         dtype=torch.float32, device=DEVICE)
        # Connect 4 is left-right symmetric: mirror the boards (flip the
        # column axis) and the pi targets for free extra data.
        x = torch.cat([x, torch.flip(x, dims=[-1])])
        pi = torch.cat([pi, torch.flip(pi, dims=[-1])])
        z = torch.cat([z, z])
        buf_x = x if buf_x is None else torch.cat([buf_x, x])[-BUFFER:]
        buf_pi = pi if buf_pi is None else torch.cat([buf_pi, pi])[-BUFFER:]
        buf_z = z if buf_z is None else torch.cat([buf_z, z])[-BUFFER:]

        for _ in range(STEPS_PER_GAME * wave):
            idx = torch.randint(len(buf_x), (min(MINIBATCH, len(buf_x)),), device=DEVICE)
            p, v = net(buf_x[idx])
            loss = ((buf_z[idx] - v.squeeze(-1)) ** 2).mean() \
                - (buf_pi[idx] * torch.log(p.clamp(min=1e-8))).sum(dim=1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        prev_done, done = done, done + wave
        torch.save(net.state_dict(), ckpt_dir / "latest.pt")
        dt = time.time() - t0
        print(f"[{done}/{n_games}] {dt:.0f}s ({dt / done:.2f}s/game), "
              f"loss {loss.item():.3f}", flush=True)
        if done // eval_every > prev_done // eval_every:
            for label, s in (("raw", 0), (f"mcts{EVAL_SIMS}", EVAL_SIMS)):
                w, d, l = pit(net, snapshot, sims=s)
                rw, rd, rl = pit(net, None, sims=s)
                print(f"  {label}: vs snapshot {w}-{d}-{l}, vs random {rw}-{rd}-{rl} (W-D-L)")
            print(f"  raw net probes: win-in-1 {probe_accuracy(net, win_probes):.0%}, "
                  f"block-in-1 {probe_accuracy(net, block_probes):.0%}", flush=True)
            snapshot = copy.deepcopy(net)
            torch.save(net.state_dict(), ckpt_dir / f"g{done:05d}.pt")
    if pool:
        pool.close()
    torch.save(net.state_dict(), ckpt_dir / f"g{done:05d}.pt")
    return net


if __name__ == "__main__":
    train(n_games=1000, sims=100, parallel=64, eval_every=128,
          run=sys.argv[1] if len(sys.argv) > 1 else None)
