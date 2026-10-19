from environment.board import Board
from agents.random import RandomAgent
from agents.q_learning_agent import QLearningAgent


class Trainer:
    def __init__(
        self,
        episodes: int=10_000,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    ):
        self.episodes = episodes
        self.epsilon_decay = epsilon_decay
        self.min_epsilon=min_epsilon


    def train(self, agent: QLearningAgent) -> dict:

        board = Board()
        opponent = RandomAgent(player=-1)

        wins = 0
        losses = 0
        draws = 0

        for episode in range(self.episodes):
            board.reset()

            while True:
                state = board.get_state()
                valid_moves = board.get_valid_moves()
                action = agent.select_action(state, valid_moves)

                board.make_move(action, agent.player)
                winner=board.check_winner()

                if winner == agent.player:
                    agent.update(
                        state, action, 1,
                        board.get_state(), [], True
                    )
                    wins += 1
                    break

                if not board.get_valid_moves():
                    agent.update(
                        state, action, 0,
                        board.get_state(), [], True
                    )
                    draws += 1
                    break


                opponent_action = opponent.select_action(
                    board.get_state(),
                    board.get_valid_moves(),
                )
                board.make_move(opponent_action, opponent.player)
                winner = board.check_winner()


                if winner == opponent.player:
                    agent.update(
                        state, action, -1,
                        board.get_state(), [], True
                    )
                    losses += 1
                    break


                if not board.get_valid_moves():
                    agent.update(
                        state, action, 0,
                        board.get_state(), [], True
                    )
                    draws += 1
                    break

                next_state = board.get_state()
                next_valid_moves = board.get_valid_moves()

                agent.update(
                    state,
                    action,
                    0,
                    next_state,
                    next_valid_moves,
                    False,
                )

            agent.epsilon = max(
                self.min_epsilon,
                agent.epsilon * self.epsilon_decay,
            )

        return {
            "episodes": self.episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / self.episodes,
            "draw_rate": draws / self.episodes,
            "final_epsilon": agent.epsilon,
        }