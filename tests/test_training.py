import pytest

from src.evaluation import EvaluationMetrics, Evaluator
from src.agents import RandomAgent
from src.training import OPPONENTS, Trainer



def test_trainer_validates_episodes():
    with pytest.raises(ValueError):
        Trainer(episodes=0)


# ------------------------------------------------------------------ evaluation
def test_metrics_summary():
    metrics = EvaluationMetrics()
    for result in ("A", "A", "B", "draw"):
        metrics.record_result(result)
    summary = metrics.summary()
    assert summary["games"] == 4
    assert summary["win_rate_a"] == 0.5
    assert summary["draw_rate"] == 0.25
    with pytest.raises(ValueError):
        metrics.record_result("C")


def test_empty_metrics_do_not_divide_by_zero():
    assert EvaluationMetrics().summary()["win_rate_a"] == 0.0


def test_evaluator_alternates_the_starting_player():
    # Same agent type on both sides: results must be roughly symmetric.
    result = Evaluator().evaluate(RandomAgent(), RandomAgent(), games=400)
    assert result["games"] == 400
    assert abs(result["wins_a"] - result["wins_b"]) < 80


def test_pipeline_runs_with_random_agents():
    stats = Trainer(episodes=50).train(RandomAgent())
    assert stats["wins"] + stats["draws"] + stats["losses"] == 50
    result = Evaluator().evaluate(RandomAgent(), RandomAgent(), games=100)
    assert result["games"] == 100