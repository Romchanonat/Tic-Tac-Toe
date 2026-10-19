import random
from collections import defaultdict
import json
from pathlib import Path

from agents.base_agent import BaseAgent

TABLE_PATH = "models/q_learning_tictactoe.json"


class QLearningAgent(BaseAgent):

    def __init__(
            self, 
            player: int,
            alpha: float = 0.1,
            gamma: float = 0.9,
            epsilon: float = 0.1,
            ):
        
        super().__init__(player)
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.q_table = defaultdict(float)


    def _get_Q_value(self, state, action):
        state_key = tuple(state)
        return self.q_table[(state_key, action)]


    def select_action(self, state, valid_moves):
        if not valid_moves:
            raise ValueError(f"no valid move with state: {state}")

        # choose wether we want to explore or maximise the reward with our current knowledge
        
        #Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        #Exploitation
        self.q_values = [
            self._get_Q_value(state, action)
            for action in valid_moves
        ]

        max_q = max(self.q_values)


        best_moves = [
            action
            for action, q_value in zip(valid_moves, self.q_values)
            if q_value == max_q
        ]

        return random.choice(best_moves)


    def update(self, state, action, reward, next_state, next_valid_moves, done):

        current_q = self._get_Q_value(state, action)

        if done or not next_valid_moves:
            target = reward
        else:
            max_next_q = max(
                self._get_Q_value(next_state, next_action)
                for next_action in next_valid_moves
            )

            target = reward + self.gamma * max_next_q

        self.q_table[(tuple(state), action)] = (
            current_q + self.alpha * (target - current_q)
        )


    def decay_epsilon(self, decay=0.995, min_espilon=0.01):
        self.epsilon = max(min_espilon, self.epsilon * decay)




    def save(self, path="models/q_learning_tictactoe.json"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "q_table": [
                {
                    "state": list(state),
                    "action": action,
                    "value": value,
                }
                for (state, action), value in self.q_table.items()
            ]
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        print(f"Q-table sauvegardée dans {path}")


    def load(self, path="models/q_learning_tictactoe.json"):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Q-table introuvable : {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.q_table = defaultdict(float, {
            (tuple(item["state"]), item["action"]): item["value"]
            for item in data["q_table"]
        })

        print(f"Q-table chargée depuis {path}")
