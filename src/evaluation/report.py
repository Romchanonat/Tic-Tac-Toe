"""Round-robin evaluation of the saved agents."""

from itertools import combinations
from pathlib import Path
from typing import Sequence

from src.config import MODELS_DIR
from src.evaluation.evaluator import Evaluator
from src.registry import AGENTS, build_agent


def run_evaluation(
    agent_keys: Sequence[str],
    games: int = 1000,
    models_dir: Path = MODELS_DIR,
) -> list[dict]:
    """Play every pair of agents ``games`` times and print a Markdown table."""
    agents = {key: build_agent(key, pretrained=True, models_dir=models_dir) for key in agent_keys}
    evaluator = Evaluator()

    rows = []
    for key_a, key_b in combinations(agent_keys, 2):
        result = evaluator.evaluate(agents[key_a], agents[key_b], games=games)
        rows.append({"agent_a": key_a, "agent_b": key_b, **result})

    print(format_table(rows))
    return rows


def format_table(rows: Sequence[dict]) -> str:
    lines = [
        "| Agent A | Agent B | Games | A wins | B wins | Draws |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {AGENTS[row['agent_a']].label} | {AGENTS[row['agent_b']].label} "
            f"| {row['games']} "
            f"| {row['wins_a']} ({row['win_rate_a']:.1%}) "
            f"| {row['wins_b']} ({row['win_rate_b']:.1%}) "
            f"| {row['draws']} ({row['draw_rate']:.1%}) |"
        )
    return "\n".join(lines)
