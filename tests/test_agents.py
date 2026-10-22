import pytest

from src.agents import RandomAgent
from src.environment import Board
from src.evaluation import Evaluator


def test_player_must_be_plus_or_minus_one():
    with pytest.raises(ValueError):
        RandomAgent(player=0)
    agent = RandomAgent(player=1)
    with pytest.raises(ValueError):
        agent.player = 2


def test_agents_refuse_to_play_without_valid_moves():
    for agent in (RandomAgent(1),):
        with pytest.raises(ValueError):
            agent.select_action([1] * 9, [])