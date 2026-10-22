"""Tic-tac-toe board.

Cells are numbered 0-8, row by row::

     0 | 1 | 2
    ---+---+---
     3 | 4 | 5
    ---+---+---
     6 | 7 | 8

A cell contains ``X`` (1), ``O`` (-1) or ``EMPTY`` (0). ``X`` always plays first.
"""

from typing import Sequence

X = 1
O = -1
EMPTY = 0

WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)

_SYMBOLS = {X: "X", O: "O", EMPTY: " "}


def winner_of(cells: Sequence[int]) -> int:
    """Return the winning player (``X`` or ``O``), or ``EMPTY`` if nobody has won."""
    for a, b, c in WINNING_LINES:
        if cells[a] != EMPTY and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return EMPTY


class Board:
    """Mutable 3x3 board."""

    def __init__(self) -> None:
        self.cells = [EMPTY] * 9

    def reset(self) -> None:
        """Empty the board."""
        self.cells = [EMPTY] * 9

    def get_state(self) -> list[int]:
        """Return a copy of the cells."""
        return self.cells.copy()

    def get_valid_moves(self) -> list[int]:
        """Return the indices of the empty cells."""
        return [i for i, cell in enumerate(self.cells) if cell == EMPTY]

    def is_valid_position(self, position: int) -> bool:
        """Whether ``position`` is on the board and empty."""
        return 0 <= position < 9 and self.cells[position] == EMPTY

    def make_move(self, position: int, player: int) -> bool:
        """Place ``player`` on ``position``. Return False if the move is illegal."""
        if player not in (X, O):
            raise ValueError("player must be 1 (X) or -1 (O)")
        if not self.is_valid_position(position):
            return False
        self.cells[position] = player
        return True

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def check_winner(self) -> int:
        """Return ``X`` or ``O`` if that player has won, ``EMPTY`` otherwise."""
        return winner_of(self.cells)

    def is_terminal(self) -> bool:
        """The game is over when someone has won or the board is full."""
        return self.is_full() or self.check_winner() != EMPTY

    def __str__(self) -> str:
        rows = []
        for row in range(3):
            a, b, c = self.cells[row * 3: row * 3 + 3]
            rows.append(f" {_SYMBOLS[a]} | {_SYMBOLS[b]} | {_SYMBOLS[c]} ")
        return "\n---+---+---\n".join(rows)

    def display(self) -> None:
        """Print the board in the terminal."""
        print(self)
