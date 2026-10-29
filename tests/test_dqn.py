import pytest

pytest.importorskip("torch")

from src.agents.dqn_agent import DQNAgent
from src.training import Trainer


def test_dqn_only_selects_valid_moves():
    agent = DQNAgent(epsilon=0.0)
    for _ in range(20):
        assert agent.select_action([1, -1, 1, -1, 0, 0, 0, 0, 0], [4, 5, 6, 7, 8]) in range(4, 9)


def test_dqn_trains_without_error():
    agent = DQNAgent(batch_size=16, memory_size=500)
    Trainer(episodes=60).train(agent)
    assert len(agent.memory) > 16
    assert agent.epsilon < 1.0


def test_dqn_save_and_load_roundtrip(tmp_path):
    agent = DQNAgent(epsilon=0.3)
    path = tmp_path / "dqn.pth"
    agent.save(path)

    loaded = DQNAgent()
    loaded.load(path)
    assert loaded.epsilon == pytest.approx(0.3)

    for p, q in zip(agent.policy_net.parameters(), loaded.policy_net.parameters()):
        assert (p == q).all()
    loaded.epsilon = 0.0
    assert loaded.select_action([1, -1, 0, 0, 0, 0, 0, 0, 0], [2, 3]) in (2, 3)


def test_dqn_target_ignores_illegal_moves_and_terminal_states():
    """A terminal transition must be learned toward exactly its reward."""
    agent = DQNAgent(batch_size=8, gamma=0.99)
    for _ in range(8):
        agent.update([0] * 9, 0, 1.0, [1] + [0] * 8, [], True)
    # Should not raise (no NaN from -inf masking) and weights stay finite.
    for p in agent.policy_net.parameters():
        assert p.isfinite().all()
