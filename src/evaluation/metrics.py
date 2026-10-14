


class EvaluationMetrics:
    def __init__(self):
        self.games_played = 0
        self.wins_a = 0
        self.wins_b = 0
        self.draws = 0

    def record_result(self, result: str):
        if result not in ("A", "B", "draw"):
            raise ValueError(f"Résultat invalide : {result}")

        self.games_played += 1

        if result == "A":
            self.wins_a += 1
        elif result == "B":
            self.wins_b += 1
        else:
            self.draws += 1

    def summary(self) -> dict:
        total = self.games_played

        if total == 0:
            return {
                "games": 0,
                "wins_a": 0,
                "wins_b": 0,
                "draws": 0,
                "win_rate_a": 0.0,
                "win_rate_b": 0.0,
                "draw_rate": 0.0,
            }

        return {
            "games": total,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "draws": self.draws,
            "win_rate_a": self.wins_a / total,
            "win_rate_b": self.wins_b / total,
            "draw_rate": self.draws / total,
        }
