"""
Value-based pruning component (Eq. 13): a linear Q-function
Q(s,a; theta) = theta . phi(s,a) trained with TD-target Q-learning and a
slowly-updated target network, matching the paper's DQN-style training
(experience replay + target network) in a lightweight, dependency-free
form suitable for this simulation.
"""
from __future__ import annotations

import numpy as np

from .features import STATE_ACTION_DIM


class QNetwork:
    def __init__(self, dim: int = STATE_ACTION_DIM, lr: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, 0.02, size=dim)
        # Apply expert heuristic initialization to guide pruning
        self.theta[1] = -0.3  # Penalize high primary mastery
        self.theta[3] = 0.3   # Favor high prereq readiness
        self.theta[7] = 0.3   # Favor high room to grow
        self.theta[9] = 0.4   # Favor high gap readiness
        self.theta[11] = 0.4  # Favor high urgency
        self.target_theta = self.theta.copy()
        self.lr = lr

    def q(self, phi: np.ndarray, use_target: bool = False) -> float:
        w = self.target_theta if use_target else self.theta
        return float(w @ phi)

    def q_batch(self, phis: np.ndarray, use_target: bool = False) -> np.ndarray:
        w = self.target_theta if use_target else self.theta
        return phis @ w

    def update_batch(self, phis: np.ndarray, targets: np.ndarray):
        """One gradient step of mean-squared TD error over a minibatch."""
        preds = phis @ self.theta
        errors = preds - targets
        grad = (phis.T @ errors) / max(len(targets), 1)
        grad = np.clip(grad, -1.0, 1.0)
        self.theta -= self.lr * grad
        return float(np.mean(errors ** 2))

    def sync_target(self, tau: float = 1.0):
        """tau=1.0 -> hard update; tau<1 -> Polyak averaging."""
        self.target_theta = tau * self.theta + (1 - tau) * self.target_theta


class ReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.capacity = capacity
        self.data = []
        self.rng = np.random.default_rng(seed)

    def push(self, item):
        self.data.append(item)
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.data))
        idx = self.rng.choice(len(self.data), size=n, replace=False)
        return [self.data[i] for i in idx]

    def __len__(self):
        return len(self.data)
