# Tic-Tac-Toe — Reinforcement Learning

An agent that learns to play tic-tac-toe by trial and error, implemented from scratch with
**tabular Q-learning**. The project includes the game environment, a training loop, a
perfect-play **Minimax** reference opponent, a round-robin evaluation, and a small GUI to
play against the agents.

<p align="center">
  <img src="docs/screenshot.png" alt="Playing against the Q-learning agent" width="260">
</p>

## Results

The Q-learning agent was trained for 30,000 games against a mix of random and perfect
(Minimax) opponents, playing as X and as O. It is then evaluated with **no exploration**,
in a round-robin of 1,000 games per pair, where the two players alternate who starts
(`python -m src evaluate`, seed 42):

<!-- Replace with the output of: python -m src evaluate --agents random minimax q_learning -->

| Agent A | Agent B | Games | A wins | B wins | Draws |
|---|---|---:|---:|---:|---:|
| Random | Minimax | 1000 | 0 (0.0%) | 856 (85.6%) | 144 (14.4%) |
| Random | Q-learning | 1000 | 8 (0.8%) | 844 (84.4%) | 148 (14.8%) |
| Minimax | Q-learning | 1000 | 0 (0.0%) | 0 (0.0%) | 1000 (100.0%) |

What this shows:

- **Q-learning never loses to the perfect player** (1000/1000 draws). Tic-tac-toe is a
  solved game: the best possible outcome against perfect play is a draw.
- It wins about **84%** of its games against a random opponent. Minimax wins 86% against the
  same opponent: it is optimal against a *perfect* player, but it does not try to set traps,
  so it is not the best possible player against a weak one.
- **Training against a random opponent alone is not enough.** An agent trained for 10,000
  games against `random` only lost **41.8%** of its games to Minimax, versus **5.0%** with the
  `mixed` opponent at the same budget. This is why `mixed` is the default.

### Learning curve

Share of wins / draws / losses over a sliding window of training games (this includes
exploratory moves). Against the mixed opponent, half of the games are against a perfect
player that cannot be beaten, so the win rate plateaus around 40–45% while losses fall
close to 0.

![Q-learning learning curve](docs/learning_curve_q_learning.png)

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The repository ships with a trained model in `models/`, so you can play right away:

```bash
python -m src play                       # GUI (Tkinter)
python -m src evaluate --games 1000      # round-robin between all agents
python -m src train                      # retrain the agent (takes seconds)
```

Useful options:

```bash
python -m src train --episodes 50000 --opponent minimax
python -m src train --plot-dir docs      # also save the learning curve (matplotlib)
python -m src --help
```

Tkinter ships with Python on Windows and macOS; on Debian/Ubuntu install it with
`sudo apt install python3-tk`.

## How it works

**Environment.** The board is a list of 9 cells (`1` = X, `-1` = O, `0` = empty). X always
starts. A state is the full board; an action is the index of an empty cell.

**Rewards.** +1 for a win, −1 for a loss, 0 for a draw or a non-terminal move.

**Point of view.** Boards are multiplied by the agent's own sign before being used, so the
agent always sees its own marks as `1`. One Q-table therefore serves both as X and as O, and
the trainer alternates the agent's side every episode.

**Q-learning** keeps a table `Q(state, action)` updated after each of the agent's moves:

```
Q(s, a) ← Q(s, a) + α · (r + γ · max_a' Q(s', a') − Q(s, a))
```

where `s'` is the board the agent sees at its next turn (after the opponent's reply), and
the target is just `r` when the game is over. Exploration is ε-greedy with a per-episode
decay.

**Minimax** is a memoised negamax search: it explores every possible continuation and picks
a move with the best guaranteed outcome, choosing at random among equally good moves. The
whole game has about 5,500 reachable positions, so it runs instantly.

**Opponents** (`--opponent`): `random`, `minimax`, or `mixed` (each game against one of the
two, chosen at random).

## Project structure

```
src/
├── environment/board.py        # game rules
├── agents/
│   ├── base.py                 # BaseAgent interface
│   ├── random_agent.py
│   ├── minimax_agent.py        # perfect-play reference
│   └── q_learning.py
├── training/                   # Trainer, opponents, learning-curve plot
├── evaluation/                 # Evaluator, metrics, round-robin report
├── ui/graphical_interface.py   # Tkinter GUI
├── registry.py                 # how each agent is built / loaded
└── cli.py                      # python -m src {train,evaluate,play}
models/                         # trained agent (Q-table as JSON)
tests/                          # pytest suite
docs/                           # screenshot and learning curve
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the game rules, the Q-learning update (hand-computed values), the symmetry
between X and O, Minimax optimality (never loses, always draws against itself), model
save/load, and a short training run.

## Limitations and ideas

- Opponents during training are only random or Minimax. There is **no self-play**, so the
  agent has not been tested against styles of play that are neither random nor perfect.
- Evaluation uses 1,000 games per pair with a fixed seed; small differences are within noise.
- Tabular Q-learning works because the state space is tiny (a few thousand reachable
  positions). It would not scale to larger boards.
- A Deep Q-Network version of this project lives on the
  [`dqn` branch](https://github.com/Romchanonat/Tic-Tac-Toe/tree/dqn). It is deliberately kept
  separate from `main`: on a game this small the table is already exact, so the DQN is there
  for comparison rather than as an improvement.
- Possible next steps: self-play and larger boards (4×4, Connect Four), where a DQN becomes
  more useful than a table.
