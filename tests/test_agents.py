import pytest

from src.agents import RandomAgent, MinimaxAgent, QLearningAgent
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


# ----------------------------------------------------------------- Q-learning
def test_q_update_terminal_reward():
    agent = QLearningAgent(player=1, alpha=0.5, gamma=0.9, epsilon=0.0)
    state = [0] * 9
    agent.update(state, 4, 1, [1, 0, 0, 0, 1, 0, 0, 0, 0], [], True)
    assert agent.get_q_value(state, 4) == pytest.approx(0.5)  # 0 + 0.5 * (1 - 0)


def test_q_update_bootstraps_from_next_state():
    agent = QLearningAgent(player=1, alpha=1.0, gamma=0.9, epsilon=0.0)
    next_state = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    agent.q_table[(tuple(next_state), 2)] = 1.0  # best known value in s'

    agent.update([0] * 9, 0, 0, next_state, [2, 3], False)

    assert agent.get_q_value([0] * 9, 0) == pytest.approx(0.9)  # 0 + 0.9 * 1.0


def test_q_values_are_shared_between_x_and_o():
    """The same position seen as X or as O must map to the same table entry."""
    as_x = QLearningAgent(player=1)
    as_o = QLearningAgent(player=-1)
    board_for_x = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    board_for_o = [-1, 1, 0, 0, 0, 0, 0, 0, 0]  # same position, marks swapped

    as_x.update(board_for_x, 2, 1, board_for_x, [], True)
    as_o.q_table = dict(as_x.q_table)
    assert as_o.get_q_value(board_for_o, 2) == as_x.get_q_value(board_for_x, 2) > 0


def test_greedy_agent_picks_best_known_move():
    agent = QLearningAgent(player=1, epsilon=0.0)
    state = [0] * 9
    agent.q_table[(tuple(state), 7)] = 0.8
    assert agent.select_action(state, list(range(9))) == 7


def test_reading_does_not_grow_the_table():
    agent = QLearningAgent(player=1, epsilon=0.0)
    agent.select_action([0] * 9, list(range(9)))
    assert agent.q_table == {}


def test_epsilon_decays_down_to_its_floor():
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.2)
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.2)


def test_q_table_save_and_load_roundtrip(tmp_path):
    agent = QLearningAgent(player=1)
    agent.q_table[((0,) * 9, 4)] = 0.25
    agent.q_table[((1, -1) + (0,) * 7, 2)] = -0.5

    path = tmp_path / "nested" / "q.json"
    agent.save(path)

    loaded = QLearningAgent(player=1)
    loaded.load(path)
    assert loaded.q_table == agent.q_table


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        QLearningAgent().load(tmp_path / "missing.json")


def test_random_agent_only_plays_valid_moves():
    board = Board()
    board.make_move(0, 1)
    agent = RandomAgent(-1)
    for _ in range(50):
        assert agent.select_action(board.get_state(), board.get_valid_moves()) != 0
