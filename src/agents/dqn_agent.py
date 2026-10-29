"""Deep Q-Network agent (PyTorch)."""

import random
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.utils import random_argmax


class QNetwork(nn.Module):
    """Small MLP approximating Q(s, .) for the 9 possible moves."""

    def __init__(self, state_size: int = 9, action_size: int = 9, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent(BaseAgent):
    """DQN with experience replay and a periodically synchronised target network.

    As for the tabular agent, boards are expressed from the agent's point of view,
    and illegal moves are masked out both when acting and when computing targets.
    """

    def __init__(
        self,
        player: int = 1,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10_000,
        target_update: int = 100,
        hidden_size: int = 64,
    ) -> None:
        super().__init__(player)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(hidden_size=hidden_size).to(self.device)
        self.target_net = QNetwork(hidden_size=hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Each transition: (state, action, reward, next_state, next_valid_mask, done)
        self.memory: deque = deque(maxlen=memory_size)
        self.steps = 0

    def _normalize_state(self, state: Sequence[int]) -> np.ndarray:
        return np.asarray(state, dtype=np.float32) * self.player

    def select_action(self, state: Sequence[int], valid_moves: Sequence[int]) -> int:
        """Epsilon-greedy choice among the legal moves."""
        self._check_moves(state, valid_moves)

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        state_tensor = torch.from_numpy(self._normalize_state(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor.to(self.device))[0].cpu().numpy()

        return random_argmax(valid_moves, [float(q_values[m]) for m in valid_moves])

    def update(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int],
        next_valid_moves: Sequence[int],
        done: bool,
    ) -> None:
        """Store the transition and run one gradient step once enough are collected."""
        next_mask = np.zeros(9, dtype=np.float32)
        next_mask[list(next_valid_moves)] = 1.0

        self.memory.append(
            (
                self._normalize_state(state),
                action,
                reward,
                self._normalize_state(next_state),
                next_mask,
                float(done),
            )
        )

        if len(self.memory) >= self.batch_size:
            self._train_step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self) -> None:
        """One mini-batch update towards the Bellman target r + gamma * max Q_target(s', a')."""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)

        def to_tensor(values, dtype=torch.float32) -> torch.Tensor:
            return torch.tensor(np.array(values), dtype=dtype, device=self.device)

        states = to_tensor(states)
        actions = to_tensor(actions, torch.long).unsqueeze(1)
        rewards = to_tensor(rewards)
        next_states = to_tensor(next_states)
        next_masks = to_tensor(next_masks)
        dones = to_tensor(dones)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q.masked_fill(next_masks == 0, float("-inf"))
            max_next_q = next_q.max(dim=1).values
            # Terminal transitions (or positions without legal move) have no future value.
            no_future = (dones == 1) | (next_masks.sum(dim=1) == 0)
            max_next_q = torch.where(no_future, torch.zeros_like(max_next_q), max_next_q)
            targets = rewards + self.gamma * max_next_q

        loss = nn.functional.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Save the networks, optimiser state and exploration rate."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load a checkpoint written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DQN checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
