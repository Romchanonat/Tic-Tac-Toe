"""Tkinter window to play against any of the agents."""

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.config import MODELS_DIR
from src.environment.board import Board
from src.registry import AGENTS, build_agent

SYMBOLS = {1: "X", -1: "O", 0: ""}
COLORS = {1: "#1f5fbf", -1: "#c0392b", 0: "black"}
AGENT_DELAY_MS = 400


class TicTacToeGUI:
    def __init__(self, root: tk.Tk, models_dir: Path = MODELS_DIR) -> None:
        self.root = root
        self.models_dir = Path(models_dir)
        self.root.title("Tic-Tac-Toe")
        self.root.resizable(False, False)

        self.board = Board()
        self.human_player = 1
        self.agent_player = -1
        self.game_over = False
        self.agent_thinking = False

        self.agent: Optional[BaseAgent] = None
        self._agent_cache: dict[str, BaseAgent] = {}
        self._pending_move: Optional[str] = None  # id of the scheduled agent move
        self._labels = {spec.label: key for key, spec in AGENTS.items()}
        self._current_label = "Random"
        self.buttons: list[tk.Button] = []

        self._build_widgets()
        self.change_agent()

    # ------------------------------------------------------------------ layout
    def _build_widgets(self) -> None:
        self.status = tk.Label(
            self.root, text="", font=("Arial", 14), pady=12, wraplength=380
        )
        self.status.pack()

        options = tk.Frame(self.root, padx=10, pady=5)
        options.pack()

        tk.Label(options, text="Opponent:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        self.agent_choice = tk.StringVar(value=self._current_label)
        selector = ttk.Combobox(
            options,
            textvariable=self.agent_choice,
            values=list(self._labels),
            state="readonly",
            width=12,
        )
        selector.pack(side=tk.LEFT, padx=5)
        selector.bind("<<ComboboxSelected>>", self.change_agent)

        self.human_first = tk.BooleanVar(value=True)
        tk.Checkbutton(
            options,
            text="I play first",
            variable=self.human_first,
            command=self.restart_game,
        ).pack(side=tk.LEFT, padx=10)

        board_frame = tk.Frame(self.root, padx=15, pady=5)
        board_frame.pack()
        for i in range(9):
            button = tk.Button(
                board_frame,
                font=("Arial", 28, "bold"),
                width=4,
                height=2,
                command=lambda index=i: self.human_move(index),
            )
            button.grid(row=i // 3, column=i % 3, padx=4, pady=4)
            self.buttons.append(button)

        tk.Button(
            self.root, text="New game", font=("Arial", 12), command=self.restart_game
        ).pack(pady=12)

    # ------------------------------------------------------------------- agent
    def change_agent(self, _event=None) -> None:
        """Switch opponent (loading its model if needed) and start a new game."""
        label = self.agent_choice.get()
        key = self._labels[label]

        try:
            if key not in self._agent_cache:
                self._agent_cache[key] = build_agent(
                    key, pretrained=True, models_dir=self.models_dir
                )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            self.agent_choice.set(self._current_label)  # keep the previous opponent
            self.status.config(text=str(error))
            return

        self.agent = self._agent_cache[key]
        self._current_label = label
        self.restart_game()

    # -------------------------------------------------------------- game flow
    def restart_game(self) -> None:
        if self._pending_move is not None:  # an agent move scheduled on the old game
            self.root.after_cancel(self._pending_move)
            self._pending_move = None

        self.board.reset()
        self.game_over = False
        self.agent_thinking = False

        self.human_player = 1 if self.human_first.get() else -1
        self.agent_player = -self.human_player
        self.agent.player = self.agent_player

        if self.agent_player == 1:  # X always starts
            self._schedule_agent_move()
        else:
            self.status.config(text=f"Your turn — you are {SYMBOLS[self.human_player]}")
            self.render()

    def human_move(self, action: int) -> None:
        if self.game_over or self.agent_thinking:
            return
        if not self.board.make_move(action, self.human_player):
            return

        if not self.check_game_over():
            self._schedule_agent_move()
        self.render()

    def _schedule_agent_move(self) -> None:
        self.agent_thinking = True
        self.status.config(text="The agent is thinking...")
        self.render()
        self._pending_move = self.root.after(AGENT_DELAY_MS, self.agent_move)

    def agent_move(self) -> None:
        self._pending_move = None
        self.agent_thinking = False

        if self.game_over:
            return

        action = self.agent.select_action(
            self.board.get_state(), self.board.get_valid_moves()
        )
        self.board.make_move(action, self.agent_player)

        if not self.check_game_over():
            self.status.config(text=f"Your turn — you are {SYMBOLS[self.human_player]}")
        self.render()

    def check_game_over(self) -> bool:
        winner = self.board.check_winner()

        if winner == self.human_player:
            message = "You won!"
        elif winner == self.agent_player:
            message = "The agent won!"
        elif self.board.is_full():
            message = "It's a draw!"
        else:
            return False

        self.game_over = True
        self.status.config(text=message)
        return True

    # ---------------------------------------------------------------- display
    def render(self) -> None:
        """Redraw the marks and enable only the cells the human can play."""
        playable = not self.game_over and not self.agent_thinking
        for button, value in zip(self.buttons, self.board.get_state()):
            button.config(
                text=SYMBOLS[value],
                fg=COLORS[value],
                disabledforeground=COLORS[value],  # keep marks readable once played
                state=tk.NORMAL if playable and value == 0 else tk.DISABLED,
            )


def main(models_dir: Path = MODELS_DIR) -> None:
    root = tk.Tk()
    TicTacToeGUI(root, models_dir)
    root.mainloop()
