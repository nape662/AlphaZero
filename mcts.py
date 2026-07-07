"""MCTS with PUCT selection, as in AlphaZero.

Edge statistics (P, N, W) live on the parent; children are just pointers.
All values are from the perspective of the node's player-to-move, so
backprop negates at every step up.

One simulation is split at the net call so callers can batch evaluations
across many independent trees: descend() walks to a fresh leaf and returns
it as an evaluation request; set_prior() + backprop() complete the
simulation once the caller has the net's answer.
"""

import numpy as np
import torch

from connect4 import NUM_ACTIONS, legal_moves, play, terminal_value
from net import DEVICE, encode

C_PUCT = 1.5
DIRICHLET_ALPHA = 1.0
DIRICHLET_EPS = 0.25


class Node:
    def __init__(self, state):
        self.state = state
        self.terminal = terminal_value(state)
        self.legal = legal_moves(state)
        self.P = None
        self.N = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.children = [None] * NUM_ACTIONS


def select(node):
    Q = np.divide(node.W, node.N, out=np.zeros(NUM_ACTIONS), where=node.N > 0)
    # +1 so the very first pick at a fresh node follows the priors
    # instead of an all-zero tie.
    U = C_PUCT * node.P * np.sqrt(node.N.sum() + 1) / (1 + node.N)
    scores = np.full(NUM_ACTIONS, -np.inf)
    scores[node.legal] = (Q + U)[node.legal]
    return int(np.argmax(scores))


def descend(root):
    """One simulation, up to the point it needs the net.

    Walks the tree by Q+U. If the sim ends on a terminal node, backs up the
    exact value right here and returns None: the simulation is complete.
    Otherwise returns (path, leaf) — an evaluation request; the caller runs
    the net and completes the sim with set_prior(leaf, ...) + backprop(path, v).
    """
    node, path = root, []
    while True:
        a = select(node)
        path.append((node, a))
        child = node.children[a]
        if child is None:
            child = Node(play(node.state, a))
            node.children[a] = child
        if child.terminal is not None:
            backprop(path, child.terminal)
            return None
        if child.P is None:
            return path, child
        node = child


def backprop(path, v):
    """v is from the leaf's perspective; parents alternate, so flip each step."""
    for node, a in reversed(path):
        v = -v
        node.W[a] += v
        node.N[a] += 1


def set_prior(node, probs):
    node.P = np.zeros(NUM_ACTIONS)
    node.P[node.legal] = probs[node.legal]
    node.P /= node.P.sum()


def add_noise(root):
    eta = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.legal))
    root.P[root.legal] = (1 - DIRICHLET_EPS) * root.P[root.legal] + DIRICHLET_EPS * eta


def evaluate(node, net):
    """Single-position net call: fill node.P, return the value."""
    with torch.no_grad():
        probs, value = net(encode(node.state)[None].to(DEVICE))
    set_prior(node, probs[0].cpu().numpy())
    return value.item()


def mcts(root_state, net, sims=800, noise=True):
    """Single-game search: run the split halves back-to-back, no batching."""
    root = Node(root_state)
    evaluate(root, net)
    if noise:
        add_noise(root)
    for _ in range(sims):
        request = descend(root)
        if request is not None:
            path, leaf = request
            backprop(path, evaluate(leaf, net))
    return root.N / root.N.sum()
