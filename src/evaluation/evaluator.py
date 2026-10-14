
from environment.board import Board
from evaluation.metrics import EvaluationMetrics


class Evaluator:
    def __init__(self, board_factory=Board):
        self.board_factory = board_factory

    def play_game(self, agent_a, agent_b):

        board = self.board_factory()
        agents = {
            1: agent_a,
            -1: agent_b,
        }

        for agent in agents.values():
            agent.reset()

        current_player = 1

        while True:
            agent = agents[current_player]

            state = board.get_state()
            valid_moves = board.get_valid_moves()

            if not valid_moves:
                return "draw"

            action = agent.select_action(state, valid_moves)

            if action not in valid_moves:
                raise ValueError(
                    f"{agent.__class__.__name__} a choisi "
                    f"un coup invalide : {action}"
                )

            board.make_move(action, current_player)

            winner = board.check_winner()

            if winner == current_player:
                return "A" if current_player == 1 else "B"

            if not board.get_valid_moves():
                return "draw"

            current_player = -current_player

    def evaluate(self, agent_a_factory, agent_b_factory, games=100):

        if games <= 0:
            raise ValueError("games doit être strictement positif")

        metrics = EvaluationMetrics()

        for game_index in range(games):
            if game_index % 2 == 0:
                agent_a = agent_a_factory(1)
                agent_b = agent_b_factory(-1)

                result = self.play_game(agent_a, agent_b)

            else:
                agent_b = agent_b_factory(1)
                agent_a = agent_a_factory(-1)

                result = self.play_game(agent_b, agent_a)

                if result == "A":
                    result = "B"
                elif result == "B":
                    result = "A"

            metrics.record_result(result)

        return metrics.summary()
