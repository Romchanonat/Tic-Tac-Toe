

class Board:

    def __init__(self):
        # We initialize the board with zeros and
        # fill a cell with 1 for the player or -1 for the agent
        self.cells = [0] * 9


    def reset(self):
        """Reset the board to its initial state."""
        self.cells = [0] * 9


    def get_state(self):
        """Return a copy of the current board state."""
        return self.cells.copy()


    def get_valid_moves(self):
        """Return the positions that are still available."""
        return [
            i for i, cell in enumerate(self.cells)
            if cell == 0
        ]


    def is_valid_position(self, position):
        """
        Check whether a position is valid and empty.
        """
        return 0 <= position < 9 and self.cells[position] == 0


    def make_move(self, position, player):
        """
        Allocate a cell to a player.

        player should be:
            1  -> real player
            -1 -> agent

        Returns:
            1 if the move was successfully made,
            0 otherwise.
        """
        if self.is_valid_position(position):
            self.cells[position] = player
            return 1

        return 0


    def is_full(self):
        """Return True if there are no empty cells."""
        return 0 not in self.cells


    def check_winner(self):
        """
        Check whether a player has won.

        Returns:
            1  -> real player won
            -1 -> agent won
            0  -> nobody has won
        """

        winning_combinations = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6)
        ]

        for a, b, c in winning_combinations:
            if (
                self.cells[a] != 0
                and self.cells[a] == self.cells[b]
                and self.cells[a] == self.cells[c]
            ):
                return self.cells[a]

        return 0


    def is_terminal(self):
        """
        Return True if the game is over.

        A game ends when:
            - a player has won
            - the board is full
        """
        return self.is_full() or self.check_winner() != 0


    def display(self):
        """Display the board in the terminal."""

        symbols = {
            1: "X",
            -1: "O",
            0: " "
        }

        for row in range(3):
            start = row * 3

            print(
                f" {symbols[self.cells[start]]} "
                f"| {symbols[self.cells[start + 1]]} "
                f"| {symbols[self.cells[start + 2]]} "
            )

            if row < 2:
                print("---+---+---")

