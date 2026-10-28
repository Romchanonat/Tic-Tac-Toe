from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Win/draw counters for a match between agent A and agent B."""

    games_played: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0

    def record_result(self, result: str) -> None:
        """``result`` is ``"A"``, ``"B"`` or ``"draw"``."""
        if result not in ("A", "B", "draw"):
            raise ValueError(f"invalid result: {result!r}")

        self.games_played += 1
        if result == "A":
            self.wins_a += 1
        elif result == "B":
            self.wins_b += 1
        else:
            self.draws += 1

    def summary(self) -> dict:
        total = self.games_played or 1  # avoid dividing by zero when empty
        return {
            "games": self.games_played,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "draws": self.draws,
            "win_rate_a": self.wins_a / total,
            "win_rate_b": self.wins_b / total,
            "draw_rate": self.draws / total,
        }
