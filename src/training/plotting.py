"""Learning-curve plot (requires matplotlib)."""

from pathlib import Path
from typing import Sequence


def rolling_rate(history: Sequence[int], outcome: int, window: int) -> list[float]:
    """Share of ``outcome`` over a sliding window of games."""
    rates = []
    for end in range(window, len(history) + 1):
        chunk = history[end - window:end]
        rates.append(chunk.count(outcome) / window)
    return rates


def plot_learning_curve(
    history: Sequence[int], title: str, path: str | Path, window: int = 500
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = min(window, len(history))
    x = range(window, len(history) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, rolling_rate(history, 1, window), label="wins")
    ax.plot(x, rolling_rate(history, 0, window), label="draws")
    ax.plot(x, rolling_rate(history, -1, window), label="losses")
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"share of the last {window} games")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
