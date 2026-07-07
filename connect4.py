r"""Barebones Connect 4 on bitboards.

Board layout — each column takes 7 bits (6 playable rows + 1 sentinel):

    bit index = 7*col + row

    row 6 |  6 13 20 27 34 41 48   <- sentinel row, never occupied
    row 5 |  5 12 19 26 33 40 47
    row 4 |  4 11 18 25 32 39 46
    row 3 |  3 10 17 24 31 38 45
    row 2 |  2  9 16 23 30 37 44
    row 1 |  1  8 15 22 29 36 43
    row 0 |  0  7 14 21 28 35 42   <- bottom row, pieces "fall" here
          +---------------------
    col      0  1  2  3  4  5  6

Why the sentinel row: with exactly 6 bits per column, the top cell of
column 0 (bit 5) and the bottom cell of column 1 (bit 6) would be adjacent
integers, so the shift tricks below would see fake "lines" running across
the column boundary. One permanently-empty bit per column breaks that chain.

With this layout, stepping to a neighboring cell is a fixed shift:
    up          -> +1
    right       -> +7
    up-right  / -> +8
    down-right \ -> +6

State is a plain tuple (current, opponent): the bitboard of the player to
move and the bitboard of the other player. `play` returns the new state with
the two swapped, so "current" always means "player to move" — this is the
perspective AlphaZero's network wants, and immutable tuples mean MCTS can
keep states around without copying or undoing moves.
"""

ROWS, COLS = 6, 7
NUM_ACTIONS = COLS  # one action per column, needed by AlphaZero's policy head

# Bitboard with every playable cell set; when (current | opponent) == FULL
# the board is full and the game is a draw.
FULL = sum(1 << (7 * col + row) for col in range(COLS) for row in range(ROWS))

INITIAL = (0, 0)


def has_won(bb):
    """True if bitboard bb contains 4 in a row (any direction).

    For each direction's shift s: `bb & (bb >> s)` marks cells that have a
    same-player neighbor s steps away (i.e. all 2-in-a-rows). Doing the same
    again with 2*s on that result finds two overlapping 2-in-a-rows spaced
    2 apart — which is exactly a 4-in-a-row.
    """
    for shift in (1, 7, 6, 8):  # | — \ /
        m = bb & (bb >> shift)
        if m & (m >> 2 * shift):
            return True
    return False


def legal_moves(state):
    """Columns that can still take a piece.

    A column is playable iff its top playable cell (row 5) is empty.
    """
    mask = state[0] | state[1]
    return [col for col in range(COLS) if not (mask >> (7 * col + 5)) & 1]


def play(state, col):
    """Drop the current player's piece in `col`, return the new state.

    Finding the landing cell without tracking heights: adding a 1 at the
    column's bottom bit to the occupancy mask carries up through the filled
    cells and stops at the first empty one — that carry bit is the move.
    """
    current, opponent = state
    mask = current | opponent
    move = (mask + (1 << 7 * col)) & (0b111111 << 7 * col)
    # Swap the boards: the opponent becomes the player to move.
    return (opponent, current | move)


def terminal_value(state):
    """Game result from the current player's perspective, or None if ongoing.

    -1: the opponent (who just moved) won.  0: draw.  The current player can
    never already have won — you can't win on the opponent's move — so +1
    never occurs here.
    """
    current, opponent = state
    if has_won(opponent):
        return -1
    if (current | opponent) == FULL:
        return 0
    return None


def winning_moves(state):
    """Legal moves that win on the spot for the player to move.

    A winning move shows as terminal_value == -1 because `play` flips the
    perspective: the new "current" player is the one who just got beaten.
    """
    return [a for a in legal_moves(state) if terminal_value(play(state, a)) == -1]


def render(state, to_move="X"):
    """ASCII board, top row first. `to_move` is the mark of the player to move."""
    current, opponent = state
    marks = {to_move: current, ("O" if to_move == "X" else "X"): opponent}
    lines = []
    for row in reversed(range(ROWS)):
        cells = []
        for col in range(COLS):
            bit = 1 << (7 * col + row)
            cells.append("X" if marks["X"] & bit else "O" if marks["O"] & bit else ".")
        lines.append(" ".join(cells))
    lines.append(" ".join(str(c) for c in range(COLS)))
    return "\n".join(lines)


if __name__ == "__main__":
    # Two-human CLI game, mostly for eyeballing that the rules work.
    state, mark = INITIAL, "X"
    while terminal_value(state) is None:
        print(render(state, mark) + "\n")
        col = int(input(f"{mark} to move, column? "))
        if col not in legal_moves(state):
            print("illegal move")
            continue
        state = play(state, col)
        mark = "O" if mark == "X" else "X"
    print(render(state, mark))
    value = terminal_value(state)
    winner = "draw" if value == 0 else ("O" if mark == "X" else "X")
    print("result:", winner)
