"""MCTS with PUCT selection, as in AlphaZero.

Edge statistics (P, N, W) live on the parent; children are just pointers.
All values are from the perspective of the node's player-to-move, so
backprop negates at every step up.
"""

import numpy as np
import torch

from connect4 import NUM_ACTIONS, legal_moves, play, terminal_value
from net import encode

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


def evaluate(node, net):
    """Run the net: fill node.P (masked to legal moves), return value."""
    with torch.no_grad():
        probs, value = net(encode(node.state))
    P = np.zeros(NUM_ACTIONS)
    P[node.legal] = probs.numpy()[node.legal]
    node.P = P / P.sum()
    return value.item()


def select(node):
    Q = np.divide(node.W, node.N, out=np.zeros(NUM_ACTIONS), where=node.N > 0)
    # +1 so the very first pick at a fresh node follows the priors
    # instead of an all-zero tie.
    U = C_PUCT * node.P * np.sqrt(node.N.sum() + 1) / (1 + node.N)
    scores = np.full(NUM_ACTIONS, -np.inf)
    scores[node.legal] = (Q + U)[node.legal]
    return int(np.argmax(scores))


def mcts(root_state, net, sims=800):
    root = Node(root_state)
    evaluate(root, net)
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.legal))
    root.P[root.legal] = (1 - DIRICHLET_EPS) * root.P[root.legal] + DIRICHLET_EPS * noise

    for _ in range(sims):
        node, path = root, []
        while True:
            a = select(node)
            path.append((node, a))
            child = node.children[a]
            if child is None:
                child = Node(play(node.state, a))
                node.children[a] = child
                v = child.terminal if child.terminal is not None else evaluate(child, net)
                break
            if child.terminal is not None:
                v = child.terminal
                break
            node = child
        for node, a in reversed(path):
            v = -v
            node.W[a] += v
            node.N[a] += 1

    return root.N / root.N.sum()
