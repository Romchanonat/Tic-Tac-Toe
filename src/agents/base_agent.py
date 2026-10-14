"""
this file defines a skeleton for every agent that will play the game.
They all must implement the fonction defined below and inherit this base class
"""


from abc import ABC, abstractmethod


class BaseAgent:

    def __init__(self, player : int):
        """
        used to define the symbol (1 or -1) that will be used by the agent
        """

        if player not in (1,-1):
            raise ValueError("player must be either 1 or -1")

        self.player = player


    @abstractmethod
    def select_action(self, state, valid_moves):
        """
        This function makes the agent chose a move among the valid_moves based on state
        """
        pass


    def reset():
        """
        This function is used to reset the agent's internal state if it has one.
        """
        pass