"""Command-line interface: ``python -m tictactoe_rl {train,evaluate,play}``."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from src.config import DOCS_DIR, MODELS_DIR
from src.registry import AGENTS, TRAINABLE_AGENTS, build_agent, model_path
from src.training import OPPONENTS, Trainer
from src.utils import set_seed


def cmd_train(args: argparse.Namespace) -> None:
    for key in args.agents:
        set_seed(args.seed)
        agent = build_agent(key)
        label = AGENTS[key].label

        print(f"\nTraining {label} for {args.episodes} episodes against '{args.opponent}'...")
        trainer = Trainer(
            episodes=args.episodes,
            opponent_factory=OPPONENTS[args.opponent],
            log_every=max(1, args.episodes // 5),
        )
        stats = trainer.train(agent)

        path = model_path(key, args.models_dir)
        agent.save(path)
        print(
            f"Saved to {path} (training: {stats['win_rate']:.1%} wins, "
            f"{stats['draw_rate']:.1%} draws, {stats['loss_rate']:.1%} losses)"
        )

        if args.plot_dir:
            from src.training.plotting import plot_learning_curve

            plot_path = Path(args.plot_dir) / f"learning_curve_{key}.png"
            plot_learning_curve(
                stats["history"],
                f"{label} — training against '{args.opponent}'",
                plot_path,
            )
            print(f"Learning curve saved to {plot_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.evaluation.report import run_evaluation

    set_seed(args.seed)
    print(f"Round-robin, {args.games} games per matchup (agents alternate who starts)\n")
    run_evaluation(args.agents, games=args.games, models_dir=args.models_dir)


def cmd_play(args: argparse.Namespace) -> None:
    from src.ui.graphical_interface import main as launch_gui

    launch_gui(models_dir=args.models_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tictactoe_rl",
        description="Tic-tac-toe agents trained with reinforcement learning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                         help="where models are saved/loaded (default: ./models)")

    train = subparsers.add_parser("train", help="train agents and save them")
    add_common(train)
    train.add_argument("--agents", nargs="+", choices=TRAINABLE_AGENTS,
                       default=TRAINABLE_AGENTS, help="agents to train (default: all)")
    train.add_argument("--episodes", type=int, default=30_000)
    train.add_argument("--opponent", choices=sorted(OPPONENTS), default="mixed",
                       help="opponent used during training (default: mixed)")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--plot-dir", type=Path, nargs="?", const=DOCS_DIR, default=None,
                       help="save learning curves as PNG (needs matplotlib)")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="round-robin between saved agents")
    add_common(evaluate)
    evaluate.add_argument("--agents", nargs="+", choices=sorted(AGENTS),
                          default=["random", "minimax", "q_learning", ]) #,  "dqn"
    evaluate.add_argument("--games", type=int, default=1000, help="games per matchup")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.set_defaults(func=cmd_evaluate)

    play = subparsers.add_parser("play", help="play against an agent in a window")
    add_common(play)
    play.set_defaults(func=cmd_play)

    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return 1
    return 0