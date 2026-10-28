"""Tabular Q-learning agent."""

import json
import random
from pathlib import Path
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.utils import random_argmax


class QLearningAgent(BaseAgent):
    """Learns a table Q(state, action) with the standard Q-learning update.

    States are expressed from the agent's point of view (own marks = 1, opponent
    marks = -1), so the same table is valid whether the agent plays X or O.
    """

    def __init__(
        self,
        player: int = 1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        super().__init__(player)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # (normalised state, action) -> value. Missing entries are worth 0.
        self.q_table: dict[tuple[tuple[int, ...], int], float] = {}

    def _normalize_state(self, state: Sequence[int]) -> tuple[int, ...]:
        return tuple(cell * self.player for cell in state)

    def get_q_value(self, state: Sequence[int], action: int) -> float:
        return self.q_table.get((self._normalize_state(state), action), 0.0)

    def select_action(self, state: Sequence[int], valid_moves: Sequence[int]) -> int:
        """Epsilon-greedy choice among the legal moves."""
        self._check_moves(state, valid_moves)

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        values = [self.get_q_value(state, move) for move in valid_moves]
        return random_argmax(valid_moves, values)

    def update(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int],
        next_valid_moves: Sequence[int],
        done: bool,
    ) -> None:
        """Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a)).

        ``next_state`` is the board the agent sees at its next turn (after the
        opponent's reply), so the target is ``reward + gamma * max_a' Q(s', a')``,
        or just ``reward`` when the game is over.
        """
        key = (self._normalize_state(state), action)
        current_q = self.q_table.get(key, 0.0)

        if done or not next_valid_moves:
            target = reward
        else:
            best_next = max(self.get_q_value(next_state, a) for a in next_valid_moves)
            target = reward + self.gamma * best_next

        self.q_table[key] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Write the Q-table to a JSON file (zero entries are implicit and skipped)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entries = [
            [list(state), action, round(value, 6)]
            for (state, action), value in self.q_table.items()
            if value != 0.0
        ]
        with path.open("w", encoding="utf-8") as file:
            json.dump({"q_table": entries}, file, separators=(",", ":"))

    def load(self, path: str | Path) -> None:
        """Load a Q-table written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Q-table not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.q_table = {
            (tuple(state), action): value for state, action, value in data["q_table"]
        }
