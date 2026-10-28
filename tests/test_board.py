import pytest

from src.environment import EMPTY, O, X, Board, winner_of


def test_new_board_is_empty():
    board = Board()
    assert board.get_state() == [EMPTY] * 9
    assert board.get_valid_moves() == list(range(9))
    assert not board.is_terminal()


def test_make_move_and_illegal_moves():
    board = Board()
    assert board.make_move(4, X)
    assert not board.make_move(4, O)  # occupied
    assert not board.make_move(9, O)  # off the board
    assert not board.make_move(-1, O)
    assert 4 not in board.get_valid_moves()


def test_make_move_rejects_invalid_player():
    with pytest.raises(ValueError):
        Board().make_move(0, 2)


def test_get_state_returns_a_copy():
    board = Board()
    board.get_state()[0] = X
    assert board.cells[0] == EMPTY


@pytest.mark.parametrize(
    "line", [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
)
@pytest.mark.parametrize("player", [X, O])
def test_every_winning_line_is_detected(line, player):
    board = Board()
    for position in line:
        board.make_move(position, player)
    assert board.check_winner() == player
    assert board.is_terminal()


def test_draw_has_no_winner():
    board = Board()
    # X O X
    # X O O
    # O X X
    for position, player in zip(range(9), [X, O, X, X, O, O, O, X, X]):
        board.make_move(position, player)
    assert board.is_full()
    assert board.check_winner() == EMPTY
    assert board.is_terminal()


def test_winner_of_accepts_tuples():
    assert winner_of((X, X, X, 0, 0, 0, 0, 0, 0)) == X


def test_reset_and_str():
    board = Board()
    board.make_move(0, X)
    board.reset()
    assert board.get_state() == [EMPTY] * 9
    assert str(board).count("---+---+---") == 2
