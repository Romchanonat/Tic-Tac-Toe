"""Minimal end-to-end run of the pipeline using only random agents."""

import random

from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer


def main() -> None:
    random.seed(42)
    agent = RandomAgent()

    print("Training (a random agent learns nothing, but the whole loop runs):")
    stats = Trainer(episodes=1000, log_every=500).train(agent)
    print(f"  {stats['wins']} wins, {stats['draws']} draws, {stats['losses']} losses\n")

    print("Evaluation, Random vs Random (1000 games, players alternate who starts):")
    result = Evaluator().evaluate(agent, RandomAgent(), games=1000)
    print(f"  A wins: {result['wins_a']}, B wins: {result['wins_b']}, draws: {result['draws']}")


if __name__ == "__main__":
    main()