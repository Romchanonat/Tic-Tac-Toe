"""Common interface shared by every agent."""

import random
from abc import ABC, abstractmethod
from typing import Sequence


class BaseAgent(ABC):
    """An agent plays ``player`` (1 for X, -1 for O) and picks moves from a board state.

    Learning agents also override :meth:`update`; the default implementation does
    nothing so that fixed-strategy agents can be used as opponents during training.
    """

    def __init__(self, player: int = 1) -> None:
        self.player = player

    @property
    def player(self) -> int:
        return self._player

    @player.setter
    def player(self, value: int) -> None:
        if value not in (1, -1):
            raise ValueError("player must be either 1 or -1")
        self._player = value

    @abstractmethod
    def select_action(self, state: Sequence[int], valid_moves: Sequence[int]) -> int:
        """Choose one move among ``valid_moves`` given the board ``state``."""

    def update(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int],
        next_valid_moves: Sequence[int],
        done: bool,
    ) -> None:
        """Learn from one transition. No-op for agents that do not learn."""

    def decay_epsilon(self) -> None:
        """Reduce the exploration rate. No-op for agents that do not explore."""

    def reset(self) -> None:
        """Clear any per-game internal state."""

    @staticmethod
    def _check_moves(state: Sequence[int], valid_moves: Sequence[int]) -> None:
        if not valid_moves:
            raise ValueError(f"no valid move for state {list(state)}")
