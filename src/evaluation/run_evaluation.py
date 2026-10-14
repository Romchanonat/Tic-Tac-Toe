from agents.random import RandomAgent
from .evaluator import Evaluator

GAMES_PER_MATCHUP=1000

def main():

    evaluator = Evaluator()

    agents = {
        RandomAgent : lambda player: RandomAgent(player=player)
    }

    for name_a, factory_a in agents.items():
        for name_b, factory_b in agents.items():
            if name_a == name_b:
                continue

            results = evaluator.evaluate(
                agent_a_factory=factory_a,
                agent_b_factory=factory_b,
                games=GAMES_PER_MATCHUP,
            )

            print(f"\n{name_a} vs {name_b}")
            print(f"Parties : {results['games']}")
            print(f"Victoires A : {results['wins_a']}")
            print(f"Victoires B : {results['wins_b']}")
            print(f"Matchs nuls : {results['draws']}")
            print(f"Taux de victoire A : {results['win_rate_a']:.1%}")
            print(f"Taux de victoire B : {results['win_rate_b']:.1%}")
            print(f"Taux de match nul : {results['draw_rate']:.1%}")


if __name__ == "__main__":
    main()

    