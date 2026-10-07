from .board import Board


class TicTacToe:
    def __init__(self):
        self.board = Board()
        self.current_player = 1


    def reset(self):
        """Reset the game to its initial state."""
        self.board.reset()
        self.current_player = 1


    def get_state(self):
        """Return the current state of the game."""
        return self.board.get_state()


    def get_valid_moves(self):
        """Return all valid moves."""
        return self.board.get_valid_moves()


    def make_move(self, position):
        """
        Play a move for the current player.

        Returns:
            1 if the move was successfully made,
            0 if the move was invalid.
        """
        if self.board.make_move(position, self.current_player):
            self.current_player *= -1
            return 1

        return 0


    def check_winner(self):
        """Return the winner, or 0 if there is no winner."""
        return self.board.check_winner()


    def is_terminal(self):
        """Return True if the game is over."""
        return self.board.is_terminal()


    def get_result(self):
        """
        Return the result of the game.

        Returns:
            1  -> player won
            -1 -> agent won
            0  -> draw
            None -> game is still ongoing
        """
        winner = self.check_winner()

        if winner != 0:
            return winner

        if self.board.is_full():
            return 0

        return None


    def display(self):
        """Display the current board."""
        self.board.display()