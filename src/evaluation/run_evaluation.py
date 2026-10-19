
from pathlib import Path
from copy import deepcopy

from agents.random import RandomAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from evaluation.evaluator import Evaluator

GAMES_PER_MATCHUP = 1000


def load_models(models):
    loaded_models = {}

    for model_class, config in models.items():
        if model_class is RandomAgent:
            loaded_models[model_class] = None
            continue

        model_path = config.get("path")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modèle introuvable pour {model_class.__name__} : "
                f"{model_path}"
            )

        agent = model_class(
            player=1,
            **config.get("params", {}),
        )
        agent.load(model_path)

        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        loaded_models[model_class] = agent

    return loaded_models


def run_evaluation(models, games=GAMES_PER_MATCHUP):
    evaluator = Evaluator()
    loaded_models = load_models(models)

    def make_factory(model_class):
        def factory(player):
            if model_class is RandomAgent:
                return RandomAgent(player=player)

            # Copie le modèle déjà chargé pour chaque nouvelle partie.
            agent = deepcopy(loaded_models[model_class])
            agent.player = player
            return agent

        return factory

    factories = {
        model_class: make_factory(model_class)
        for model_class in models
    }


    model_classes = list(factories.keys())

    for i, name_a in enumerate(model_classes):
        for name_b in model_classes[i + 1:]:
            factory_a = factories[name_a]
            factory_b = factories[name_b]

            results = evaluator.evaluate(
                agent_a_factory=factory_a,
                agent_b_factory=factory_b,
                games=games,
            )

            print(f"\n{name_a.__name__} vs {name_b.__name__}")
            print(f"Parties : {results['games']}")
            print(f"Victoires A : {results['wins_a']}")
            print(f"Victoires B : {results['wins_b']}")
            print(f"Matchs nuls : {results['draws']}")
            print(f"Taux de victoire A : {results['win_rate_a']:.1%}")
            print(f"Taux de victoire B : {results['win_rate_b']:.1%}")
            print(f"Taux de match nul : {results['draw_rate']:.1%}")


