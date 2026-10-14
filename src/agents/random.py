from .base_agent import BaseAgent

import random

class RandomAgent(BaseAgent):

    def select_action(self, state, valid_moves):
        if not valid_moves:
            raise ValueError(f"no valid move with state: {state}")
        
        return random.choice(valid_moves)