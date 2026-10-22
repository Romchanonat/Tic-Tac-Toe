"""Opponents an agent can be trained against."""

import random
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent

OpponentFactory = Callable[[int], BaseAgent]


def random_opponent(player: int) -> BaseAgent:
    return RandomAgent(player)


OPPONENTS: dict[str, OpponentFactory] = {
    "random": random_opponent,
}
