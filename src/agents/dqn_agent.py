import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """Réseau neuronal qui approxime Q(s, a)."""

    def __init__(self, state_size=9, action_size=9, latent_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, action_size),
        )

    def forward(self, state):
        return self.network(state)


import torch
from pathlib import Path


class DQNAgent(BaseAgent):
    def __init__(
        self,
        player: int,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10_000,
        target_update: int = 100,
    ):
        super().__init__(player)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=alpha,
        )

        # Chaque transition contient :
        # (état, action, récompense, état suivant,
        #  coups valides suivants, partie terminée)
        self.memory = deque(maxlen=memory_size)
        self.steps = 0

    def _normalize_state(self, state):
        """Exprime le plateau du point de vue de l'agent."""
        return np.asarray(state, dtype=np.float32) * self.player

    def select_action(self, state, valid_moves):
        """Sélection epsilon-greedy parmi les coups légaux."""
        if not valid_moves:
            raise ValueError("Aucun coup valide disponible")

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        state_tensor = torch.tensor(
            self._normalize_state(state),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]

        # On ignore les cases déjà occupées.
        valid_q_values = q_values[valid_moves]
        max_q = valid_q_values.max().item()

        best_moves = [
            move
            for move in valid_moves
            if q_values[move].item() == max_q
        ]

        return random.choice(best_moves)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_valid_moves,
        done,
    ):
        """Enregistre une transition et entraîne le réseau."""
        transition = (
            self._normalize_state(state),
            action,
            reward,
            self._normalize_state(next_state),
            list(next_valid_moves),
            done,
        )

        self.memory.append(transition)

        if len(self.memory) >= self.batch_size:
            self._train_step()

        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

    def _train_step(self):
        """Effectue une mise à jour par mini-batch."""
        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, next_moves, dones = zip(
            *batch
        )

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # Q(s, a) pour les actions réellement jouées.
        current_q = self.policy_net(states).gather(
            1, actions
        ).squeeze(1)

        # Cible de Bellman : r + gamma * max Q_target(s', a').
        # Les coups invalides sont exclus du maximum.
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = torch.zeros(
                self.batch_size,
                dtype=torch.float32,
                device=self.device,
            )

            for i, valid_moves in enumerate(next_moves):
                if not dones[i].item() and valid_moves:
                    max_next_q[i] = next_q_values[
                        i, valid_moves
                    ].max()

            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(
            current_q, targets
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=1.0
        )
        self.optimizer.step()

    def decay_epsilon(self):
        """Réduit progressivement l'exploration."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def reset(self):
        """Ne réinitialise pas le réseau entre les parties."""
        pass

    def save(self, path="dqn_model.pth"):
        """Sauvegarde le réseau et les paramètres d'entraînement."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path="dqn_model.pth"):
        """Recharge un modèle sauvegardé."""
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )
        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        self.epsilon = checkpoint["epsilon"]
