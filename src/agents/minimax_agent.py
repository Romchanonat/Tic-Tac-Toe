"""Perfect-play reference agent (negamax with memoisation).

Tic-tac-toe is a solved game: a perfect player never loses. This agent is used as
a strong benchmark and as a training opponent.
"""

import random
from functools import lru_cache
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.environment.board import EMPTY, winner_of
from src.utils import random_argmax


@lru_cache(maxsize=None)
def _negamax(cells: tuple[int, ...], player: int) -> int:
    """Game value for ``player`` to move: 1 = win, 0 = draw, -1 = loss (perfect play)."""
    if winner_of(cells) != EMPTY:
        return -1  # the previous player just won
    if EMPTY not in cells:
        return 0

    best = -1
    for move, cell in enumerate(cells):
        if cell == EMPTY:
            child = cells[:move] + (player,) + cells[move + 1:]
            best = max(best, -_negamax(child, -player))
    return best


class MinimaxAgent(BaseAgent):
    """Plays optimally, choosing at random among equally good moves.

    ``error_rate`` is the probability of playing a random move instead, which makes
    a weaker (beatable) opponent.
    """

    def __init__(self, player: int = 1, error_rate: float = 0.0) -> None:
        super().__init__(player)
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")
        self.error_rate = error_rate

    def select_action(self, state: Sequence[int], valid_moves: Sequence[int]) -> int:
        self._check_moves(state, valid_moves)

        if self.error_rate and random.random() < self.error_rate:
            return random.choice(valid_moves)

        cells = tuple(state)
        values = []
        for move in valid_moves:
            child = cells[:move] + (self.player,) + cells[move + 1:]
            values.append(-_negamax(child, -self.player))
        return random_argmax(valid_moves, values)
