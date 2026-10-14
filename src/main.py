
from environment.board import Board
from agents.random import RandomAgent


def play_game():
    board = Board()

    human_player = 1
    agent_player = -1

    agent = RandomAgent(player=agent_player)

    current_player = human_player

    print("Bienvenue au Tic-Tac-Toe !")
    print("Vous jouez avec X (1), l'agent avec O (-1).")
    print("Choisissez une case entre 0 et 8.")

    while not board.is_terminal():
        board.display()

        if current_player == human_player:
            valid_moves = board.get_valid_moves()
            print("Cases disponibles :", valid_moves)

            try:
                action = int(input("Votre coup : "))
            except ValueError:
                print("Veuillez entrer un nombre.")
                continue

            if action not in valid_moves:
                print("Coup invalide, réessayez.")
                continue

        else:
            state = board.get_state()
            valid_moves = board.get_valid_moves()

            action = agent.select_action(state, valid_moves)

            print(f"L'agent joue en case {action}.")

        board.make_move(action, current_player)

        # On change de joueur uniquement après un coup valide.
        current_player = -current_player

    board.display()

    winner = board.check_winner()

    if winner == human_player:
        print("Vous avez gagné !")
    elif winner == agent_player:
        print("L'agent a gagné !")
    else:
        print("Match nul !")


if __name__ == "__main__":
    play_game()
