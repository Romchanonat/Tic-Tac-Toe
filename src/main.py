from pathlib import Path

from agents.random import RandomAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent

from training.trainer import Trainer
from evaluation.run_evaluation import run_evaluation


MODELS = {
    RandomAgent: {
        "path": None,
        "train": False,
        "params": {},
    },
    QLearningAgent: {
        "path": "models/q_learning_tictactoe.json",
        "train": True,
        "params": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 1.0,
        },
    },
    DQNAgent: {
        "path": "models/dqn_tictactoe.pth",
        "train": True,
        "params": {},
    },
}


def train_models():

    trainer = Trainer(
        episodes=10_000,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    )

    for model_class, config in MODELS.items():

        if not config["train"]:
            continue

        model_path = config["path"]

        if Path(model_path).exists():
            continue

        agent = model_class(player=1, **config["params"])

        print(f"\nEntraînement de {model_class.__name__}...")

        results = trainer.train(agent)

        agent.save(model_path)

        print(f"Modèle sauvegardé dans {model_path}")
        print(f"Taux de victoire : {results['win_rate']:.2%}")


def main():
    train_models()
    run_evaluation(MODELS)


if __name__ == "__main__":
    main()