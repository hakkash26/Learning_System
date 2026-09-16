from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .features import STATE_ACTION_DIM, STATE_DIM


class PPOPolicyNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPWeightsProxy:
    def __init__(self, valuenet: ValueNetwork):
        self.valuenet = valuenet

    def __rmatmul__(self, other: np.ndarray) -> np.ndarray:
        return self.valuenet.predict_batch(other)


class PPOPolicy:
    def __init__(self, dim: int = STATE_ACTION_DIM, lr: float = 3e-4,
                 clip_eps: float = 0.2, entropy_coef: float = 0.01, seed: int = 1):
        torch.manual_seed(seed)
        np.random.default_rng(seed)
        self.network = PPOPolicyNet(dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

    def action_probs(self, cand_phis: np.ndarray) -> np.ndarray:
        if len(cand_phis) == 0:
            return np.array([], dtype=np.float32)
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(cand_phis, dtype=torch.float32)
            logits = self.network(x).squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=0).numpy()
        return probs

    def sample_action(self, cand_phis: np.ndarray, cand_ids: list, rng: np.random.Generator):
        probs = self.action_probs(cand_phis)
        idx = rng.choice(len(cand_ids), p=probs)
        return cand_ids[idx], float(probs[idx]), idx

    def greedy_action(self, cand_phis: np.ndarray, cand_ids: list):
        probs = self.action_probs(cand_phis)
        idx = int(np.argmax(probs))
        return cand_ids[idx], float(probs[idx])

    def update(self, batch) -> float:
        """batch: list of dicts with keys
             cand_phis (K,D), chosen_idx, old_prob, advantage
           Performs several clipped-surrogate ascent steps (PPO epochs) using vectorized forward passes."""
        if not batch:
            return 0.0
        self.network.train()
        total_loss = 0.0
        
        lengths = [len(item["cand_phis"]) for item in batch]
        all_phis = np.concatenate([item["cand_phis"] for item in batch], axis=0)
        all_phis_t = torch.tensor(all_phis, dtype=torch.float32)
        
        a_idxs = [item["chosen_idx"] for item in batch]
        old_ps = torch.tensor([item["old_prob"] for item in batch], dtype=torch.float32)
        advantages = torch.tensor([item["advantage"] for item in batch], dtype=torch.float32)
        
        for _ in range(3):  # 3 PPO epochs per update for stability
            self.optimizer.zero_grad()
            
            # Forward pass once for all steps in the episode
            all_logits = self.network(all_phis_t).squeeze(-1)
            if all_logits.ndim == 0:
                all_logits = all_logits.unsqueeze(0)
                
            # Split back into individual step tensors
            logits_list = torch.split(all_logits, lengths)
            
            epoch_loss = torch.tensor(0.0)
            for t, (logits, a_idx, old_p, A) in enumerate(zip(logits_list, a_idxs, old_ps, advantages)):
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                probs = torch.softmax(logits, dim=0)
                new_p = probs[a_idx]

                ratio = new_p / max(old_p, 1e-8)
                unclipped = ratio * A
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * A
                obj = torch.minimum(unclipped, clipped)
                
                # Entropy bonus to encourage exploration and prevent premature collapse
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                epoch_loss = epoch_loss - obj - self.entropy_coef * entropy

            epoch_loss = epoch_loss / len(batch)
            epoch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss = float(-epoch_loss.item())
        return total_loss

    def save(self, filepath: str):
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath: str):
        self.network.load_state_dict(torch.load(filepath))


class ValueNetwork:
    def __init__(self, dim: int = STATE_DIM, lr: float = 3e-4, seed: int = 2):
        torch.manual_seed(seed)
        np.random.default_rng(seed)
        self.network = ValueNet(dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)

    def value(self, phi_s: np.ndarray) -> float:
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(phi_s, dtype=torch.float32)
            val = self.network(x).item()
        return val

    def predict_batch(self, state_phis: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(state_phis, dtype=torch.float32)
            vals = self.network(x).squeeze(-1).numpy()
        return vals

    @property
    def w(self) -> MLPWeightsProxy:
        return MLPWeightsProxy(self)

    def update_batch(self, phis: np.ndarray, returns: np.ndarray) -> float:
        self.network.train()
        self.optimizer.zero_grad()

        x = torch.tensor(phis, dtype=torch.float32)
        y = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

        preds = self.network(x)
        loss = torch.mean((preds - y) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def save(self, filepath: str):
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath: str):
        self.network.load_state_dict(torch.load(filepath))
