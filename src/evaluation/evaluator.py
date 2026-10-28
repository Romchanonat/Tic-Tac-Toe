
from src.agents.base_agent import BaseAgent
from src.environment.board import Board
from src.evaluation.metrics import EvaluationMetrics


class Evaluator:
    """Plays matches between two agents (no learning, no exploration)."""

    @staticmethod
    def play_game(first: BaseAgent, second: BaseAgent) -> int:
        """Play one game; ``first`` plays X. Return 1 (X wins), -1 (O wins) or 0 (draw)."""
        first.player, second.player = 1, -1
        first.reset()
        second.reset()

        board = Board()
        current = first

        while True:
            state = board.get_state()
            valid_moves = board.get_valid_moves()
            action = current.select_action(state, valid_moves)

            if action not in valid_moves:
                raise ValueError(
                    f"{type(current).__name__} chose an illegal move: {action}"
                )
            board.make_move(action, current.player)

            winner = board.check_winner()
            if winner != 0:
                return winner
            if board.is_full():
                return 0

            current = second if current is first else first

    def evaluate(self, agent_a: BaseAgent, agent_b: BaseAgent, games: int = 100) -> dict:
        """Play ``games`` games, alternating which agent starts."""
        if games <= 0:
            raise ValueError("games must be positive")

        metrics = EvaluationMetrics()

        for game_index in range(games):
            if game_index % 2 == 0:
                outcome = self.play_game(agent_a, agent_b)
                a_won, b_won = outcome == 1, outcome == -1
            else:
                outcome = self.play_game(agent_b, agent_a)
                a_won, b_won = outcome == -1, outcome == 1

            metrics.record_result("A" if a_won else "B" if b_won else "draw")

        return metrics.summary()
