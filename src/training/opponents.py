"""Opponents an agent can be trained against."""

import random
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.agents.minimax_agent import MinimaxAgent

OpponentFactory = Callable[[int], BaseAgent]


def random_opponent(player: int) -> BaseAgent:
    return RandomAgent(player)


def minimax_opponent(player: int) -> BaseAgent:
    return MinimaxAgent(player)


def mixed_opponent(player: int) -> BaseAgent:
    """Each game is played against either a random or a perfect opponent."""
    return random.choice((RandomAgent, MinimaxAgent))(player)


OPPONENTS: dict[str, OpponentFactory] = {
    "random": random_opponent,
    "minimax": minimax_opponent,
    "mixed": mixed_opponent
}
