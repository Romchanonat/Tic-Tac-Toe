"""Single place describing every available agent and how to build it."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.agents.base_agent import BaseAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.agents.QLearning_agent import QLearningAgent
from src.config import MODELS_DIR


@dataclass(frozen=True)
class AgentSpec:
    label: str
    factory: Callable[[], Callable[..., BaseAgent]]
    model_file: Optional[str] = None  # None -> nothing to train or load
    params: dict = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        return self.model_file is not None

def _dqn_class():
    from src.agents.dqn_agent import DQNAgent
    return DQNAgent




AGENTS: dict[str, AgentSpec] = {
    "random": AgentSpec("Random", lambda: RandomAgent),
    "minimax": AgentSpec("Minimax", lambda: MinimaxAgent),
    "q_learning": AgentSpec(
        "Q-learning",
        lambda: QLearningAgent,
        model_file="q_learning.json",
        params={"alpha": 0.1, "gamma": 0.9, "epsilon": 1.0}
    ),
    "dqn": AgentSpec(
        "DQN", _dqn_class, model_file="dqn.pth",
        params={"learning_rate": 1e-3, "gamma": 0.99, "epsilon": 1.0},
    )
}


TRAINABLE_AGENTS = [key for key, spec in AGENTS.items() if spec.trainable]


def model_path(key: str, models_dir: Path = MODELS_DIR) -> Optional[Path]:
    spec = AGENTS[key]
    return Path(models_dir) / spec.model_file if spec.model_file else None


def build_agent(
    key: str,
    player: int = 1,
    pretrained: bool = False,
    models_dir: Path = MODELS_DIR,
) -> BaseAgent:
    """Create an agent.

    With ``pretrained=True`` the saved model is loaded and exploration is switched
    off, which is what you want for playing and evaluation. Otherwise a fresh agent
    with the default training hyper-parameters is returned.
    """
    if key not in AGENTS:
        raise ValueError(f"unknown agent {key!r}; choose from {sorted(AGENTS)}")

    spec = AGENTS[key]
    agent = spec.factory()(player=player, **spec.params)

    if pretrained and spec.trainable:
        path = model_path(key, models_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model for {spec.label} at {path}. "
                f"Train it first: python -m tictactoe_rl train --agents {key}"
            )
        agent.load(path)
        agent.epsilon = 0.0

    return agent