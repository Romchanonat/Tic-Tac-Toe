
import tkinter as tk

from environment.board import Board
from agents.random import RandomAgent


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe - Random Agent")
        self.root.resizable(False, False)

        self.board = Board()

        self.human_player = 1
        self.agent_player = -1

        self.agent = RandomAgent(player=self.agent_player)
        self.game_over = False

        self.buttons = []

        self.status = tk.Label(
            root,
            text="À vous de jouer !",
            font=("Arial", 16),
            pady=15
        )
        self.status.pack()

        board_frame = tk.Frame(root, padx=15, pady=5)
        board_frame.pack()

        for i in range(9):
            button = tk.Button(
                board_frame,
                text="",
                font=("Arial", 28, "bold"),
                width=4,
                height=2,
                command=lambda index=i: self.human_move(index)
            )

            button.grid(
                row=i // 3,
                column=i % 3,
                padx=4,
                pady=4
            )

            self.buttons.append(button)

        self.restart_button = tk.Button(
            root,
            text="Nouvelle partie",
            font=("Arial", 12),
            command=self.restart_game
        )
        self.restart_button.pack(pady=15)

    def human_move(self, action):
        if self.game_over:
            return

        valid_moves = self.board.get_valid_moves()

        if action not in valid_moves:
            return

        self.board.make_move(action, self.human_player)
        self.update_board()

        if self.check_game_over():
            return

        self.status.config(text="L'agent réfléchit...")

        self.root.after(400, self.agent_move)

    def agent_move(self):
        if self.game_over:
            return

        state = self.board.get_state()
        valid_moves = self.board.get_valid_moves()

        if not valid_moves:
            self.check_game_over()
            return

        action = self.agent.select_action(state, valid_moves)

        self.board.make_move(action, self.agent_player)
        self.update_board()

        if not self.check_game_over():
            self.status.config(text="À vous de jouer !")

    def update_board(self):
        state = self.board.get_state()

        symbols = {
            1: "X",
            -1: "O",
            0: ""
        }

        for i, value in enumerate(state):
            self.buttons[i].config(text=symbols[value])

    def check_game_over(self):
        winner = self.board.check_winner()

        if winner == self.human_player:
            self.status.config(text="Vous avez gagné !")
            self.game_over = True
            return True

        if winner == self.agent_player:
            self.status.config(text="L'agent a gagné !")
            self.game_over = True
            return True

        if not self.board.get_valid_moves():
            self.status.config(text="Match nul !")
            self.game_over = True
            return True

        return False

    def restart_game(self):
        self.board.reset()
        self.agent.reset()
        self.game_over = False

        self.update_board()
        self.status.config(text="À vous de jouer !")


def main():
    root = tk.Tk()
    TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
