import random
from typing import Sequence

from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Plays a uniformly random legal move."""

    def select_action(self, state: Sequence[int], valid_moves: Sequence[int]) -> int:
        self._check_moves(state, valid_moves)
        return random.choice(valid_moves)
