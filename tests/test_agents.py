import pytest

from src.agents import RandomAgent, MinimaxAgent
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



# --------------------------------------------------------------------- minimax
def test_minimax_takes_the_winning_move():
    #  X X .
    #  O O .
    #  . . .
    state = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    agent = MinimaxAgent(player=1)
    assert agent.select_action(state, [2, 5, 6, 7, 8]) == 2


def test_minimax_blocks_the_opponent():
    #  O O .
    #  . X .
    #  . . X   -> X must block at 2 (and cannot win immediately)
    state = [-1, -1, 0, 0, 1, 0, 0, 0, 1]
    agent = MinimaxAgent(player=1)
    assert agent.select_action(state, [2, 3, 5, 6, 7]) == 2


def test_minimax_never_loses():
    evaluator = Evaluator()
    result = evaluator.evaluate(MinimaxAgent(), RandomAgent(), games=200)
    assert result["wins_b"] == 0
    result = evaluator.evaluate(MinimaxAgent(), MinimaxAgent(), games=50)
    assert result["draws"] == 50  # perfect play always ends in a draw


def test_minimax_error_rate_is_validated():
    with pytest.raises(ValueError):
        MinimaxAgent(error_rate=1.5)