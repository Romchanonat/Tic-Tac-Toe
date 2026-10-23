"""Small helpers shared across the package."""

import random
from typing import Sequence

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is only needed for the DQN agent
        return
    torch.manual_seed(seed)


def random_argmax(moves: Sequence[int], values: Sequence[float]) -> int:
    """Return the move with the highest value, breaking ties at random."""
    best = max(values)
    return random.choice([m for m, v in zip(moves, values) if v == best])