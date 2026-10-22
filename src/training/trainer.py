"""Training loop: an agent learns by playing full games against an opponent."""

from typing import Optional

from src.agents.base_agent import BaseAgent
from src.environment.board import Board
from src.training.opponents import OpponentFactory, random_opponent

WIN, DRAW, LOSS = 1, 0, -1


class Trainer:
    """Plays ``episodes`` games and feeds every transition to the agent.

    The agent alternates between playing first (X) and second (O) so that it learns
    both roles. Rewards are +1 for a win, -1 for a loss and 0 otherwise.
    """

    def __init__(
        self,
        episodes: int = 30_000,
        opponent_factory: OpponentFactory = random_opponent,
        log_every: Optional[int] = None,
    ) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        self.episodes = episodes
        self.opponent_factory = opponent_factory
        self.log_every = log_every

    def train(self, agent: BaseAgent) -> dict:
        """Train ``agent`` in place and return statistics.

        ``history`` holds one outcome per episode (1 win, 0 draw, -1 loss) from the
        agent's point of view; note that it includes exploratory moves.
        """
        board = Board()
        history: list[int] = []

        for episode in range(self.episodes):
            agent.player = 1 if episode % 2 == 0 else -1
            opponent = self.opponent_factory(-agent.player)

            history.append(self._play_episode(agent, opponent, board))
            agent.decay_epsilon()

            if self.log_every and (episode + 1) % self.log_every == 0:
                recent = history[-self.log_every:]
                print(
                    f"  episode {episode + 1:>6}/{self.episodes} | "
                    f"epsilon {getattr(agent, 'epsilon', 0.0):.3f} | "
                    f"last {len(recent)}: "
                    f"{recent.count(WIN) / len(recent):.1%} wins, "
                    f"{recent.count(DRAW) / len(recent):.1%} draws"
                )

        wins, draws, losses = history.count(WIN), history.count(DRAW), history.count(LOSS)
        return {
            "episodes": self.episodes,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / self.episodes,
            "draw_rate": draws / self.episodes,
            "loss_rate": losses / self.episodes,
            "final_epsilon": getattr(agent, "epsilon", 0.0),
            "history": history,
        }

    @staticmethod
    def _play_episode(agent: BaseAgent, opponent: BaseAgent, board: Board) -> int:
        """Play one game and return its outcome from the agent's point of view."""
        board.reset()

        # X moves first: when the agent is O, the opponent opens the game.
        if opponent.player == 1:
            board.make_move(
                opponent.select_action(board.get_state(), board.get_valid_moves()),
                opponent.player,
            )

        while True:
            state = board.get_state()
            action = agent.select_action(state, board.get_valid_moves())
            board.make_move(action, agent.player)

            if board.check_winner() == agent.player:
                agent.update(state, action, 1, board.get_state(), [], True)
                return WIN

            if board.is_full():
                agent.update(state, action, 0, board.get_state(), [], True)
                return DRAW

            opponent_action = opponent.select_action(
                board.get_state(), board.get_valid_moves()
            )
            board.make_move(opponent_action, opponent.player)

            if board.check_winner() == opponent.player:
                agent.update(state, action, -1, board.get_state(), [], True)
                return LOSS

            if board.is_full():
                agent.update(state, action, 0, board.get_state(), [], True)
                return DRAW

            agent.update(
                state, action, 0, board.get_state(), board.get_valid_moves(), False
            )
