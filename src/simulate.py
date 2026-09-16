"""
Episode rollout logic shared by training and evaluation. Implements the
'prune -> select -> deliver -> update -> log' loop of Fig. 1 for the
proposed method, and analogous single-step-choice loops for each
baseline.
"""
from __future__ import annotations

import numpy as np

from .environment import LearningPathEnv
from .knowledge_graph import KnowledgeGraph
from .learner_model import LearnerState, sigmoid_center
from .agents.features import state_action_features, state_features, STATE_ACTION_DIM
from .agents.q_learning import QNetwork, ReplayBuffer
from .agents.ppo import PPOPolicy, ValueNetwork

GAMMA_PPO = 0.99   # Eq. 11
GAMMA_Q = 0.95      # Eq. 13
PRUNE_TOP_M = 12


def expected_gain(kg: KnowledgeGraph, s: np.ndarray, action: int, correctness_model) -> float:
    """Closed-form expected mastery gain used as the 'prediction' pi in
    the MAE/RMSE calibration metric -- independent of which agent picked
    the action, so all methods are scored on the same yardstick."""
    ann = kg.resource_annotations[action]
    primary = ann["primary"]
    p_correct = correctness_model(s[primary])
    eta = kg.knowledge_points_of_resource(action)  # {k: weight}

    # Direct + propagation under positive feedback (answered correctly)
    direct_pos = np.zeros(kg.n_knowledge_points, dtype=np.float64)
    for k, w in eta.items():
        direct_pos[k] = LearnerState.ALPHA_PLUS * (1.0 - s[k]) * w
    prop_pos = kg.W.T @ direct_pos
    raw_pos = s + direct_pos + prop_pos
    s_next_pos = np.clip(sigmoid_center(raw_pos), 0.0, 1.0)
    delta_pos = s_next_pos - s
    r_pos = sum(w * delta_pos[k] for k, w in eta.items())

    # Direct + propagation under negative feedback (answered incorrectly)
    direct_neg = np.zeros(kg.n_knowledge_points, dtype=np.float64)
    for k, w in eta.items():
        direct_neg[k] = -LearnerState.ALPHA_MINUS * s[k] * w
    prop_neg = kg.W.T @ direct_neg
    raw_neg = s + direct_neg + prop_neg
    s_next_neg = np.clip(sigmoid_center(raw_neg), 0.0, 1.0)
    delta_neg = s_next_neg - s
    r_neg = sum(w * delta_neg[k] for k, w in eta.items())

    return float(p_correct * r_pos + (1.0 - p_correct) * r_neg)


class OursAgent:
    """Prune-then-PPO agent: Q-network prunes candidates, PPO policy
    selects the final action from the pruned set (Section
    'Reinforcement learning strategy training and recommendation
    generation')."""

    name = "KG-RL"

    def __init__(self, kg: KnowledgeGraph, seed: int = 0, q_lr: float = 0.05,
                 policy_lr: float = 3e-4, value_lr: float = 3e-4):
        self.kg = kg
        self.qnet = QNetwork(seed=seed, lr=q_lr)
        self.policy = PPOPolicy(seed=seed + 1, lr=policy_lr)
        self.valuenet = ValueNetwork(seed=seed + 2, lr=value_lr)
        self.buffer = ReplayBuffer(seed=seed + 3)
        self.rng = np.random.default_rng(seed)

    def prune(self, s: np.ndarray, candidates: list):
        phis = np.stack([state_action_features(self.kg, s, r) for r in candidates])
        q_vals = self.qnet.q_batch(phis)
        order = np.argsort(-q_vals)
        top = order[:min(PRUNE_TOP_M, len(candidates))]
        pruned_ids = [candidates[i] for i in top]
        pruned_phis = phis[top]
        return pruned_ids, pruned_phis

    def choose(self, s: np.ndarray, candidates: list, greedy: bool = False):
        pruned_ids, pruned_phis = self.prune(s, candidates)
        if greedy:
            a, prob = self.policy.greedy_action(pruned_phis, pruned_ids)
            idx = pruned_ids.index(a)
        else:
            a, prob, idx = self.policy.sample_action(pruned_phis, pruned_ids, self.rng)
        return a, prob, pruned_ids, pruned_phis, idx

    # -------------------- training ------------------------------------
    def train_step_q(self, batch_size: int = 64):
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        phis = np.stack([b["phi_sa"] for b in batch])
        targets = []
        for b in batch:
            if b["done"]:
                targets.append(b["reward"])
            else:
                next_phis = b["next_cand_phis"]
                if len(next_phis) == 0:
                    targets.append(b["reward"])
                else:
                    max_next_q = float(np.max(self.qnet.q_batch(next_phis, use_target=True)))
                    targets.append(b["reward"] + GAMMA_Q * max_next_q)
        loss = self.qnet.update_batch(phis, np.array(targets))
        self.qnet.sync_target(tau=0.05)
        return loss

    def train_step_ppo(self, episode_trajectory):
        """episode_trajectory: list of dicts with cand_phis, chosen_idx,
        old_prob, reward, state_phi. Computes discounted returns and
        advantages, then updates policy + value nets."""
        if not episode_trajectory:
            return
        returns = np.zeros(len(episode_trajectory))
        running = 0.0
        for t in reversed(range(len(episode_trajectory))):
            running = episode_trajectory[t]["reward"] + GAMMA_PPO * running
            returns[t] = running

        state_phis = np.stack([e["state_phi"] for e in episode_trajectory])
        baselines = self.valuenet.predict_batch(state_phis)
        advantages = returns - baselines
        std = advantages.std()
        if std > 1e-6:
            advantages = (advantages - advantages.mean()) / std
        advantages = np.clip(advantages, -3.0, 3.0)

        ppo_batch = []
        for e, adv in zip(episode_trajectory, advantages):
            ppo_batch.append({
                "cand_phis": e["cand_phis"],
                "chosen_idx": e["chosen_idx"],
                "old_prob": e["old_prob"],
                "advantage": float(adv),
            })
        self.policy.update(ppo_batch)
        self.valuenet.update_batch(state_phis, returns)


# -----------------------------------------------------------------------
def run_episode_ours(env: LearningPathEnv, kg: KnowledgeGraph, agent: OursAgent,
                      learner_id: str, init_scores: np.ndarray, rng: np.random.Generator,
                      train: bool = False):
    s = env.reset(init_scores, learner_id)
    trajectory = []
    log = {"recommended": [], "rewards": [], "correct": [], "predicted_gain": [],
           "predicted_probs": [], "mastery_gaps": [], "primary_kps": [], "s0": init_scores.copy()}
    correctness_model = env.correctness_model

    done = False
    while not done:
        candidates = env.candidate_actions()
        if not candidates:
            break  # resource pool exhausted for this learner
        a, prob, pruned_ids, pruned_phis, idx = agent.choose(s, candidates, greedy=not train)
        ann = kg.resource_annotations[a]
        primary_kp = ann["primary"]
        p_correct = correctness_model(s[primary_kp])
        pred_gain = expected_gain(kg, s, a, correctness_model)
        s_next, reward, done, info = env.step(a)

        # Gap between mastery objective (0.85) and achieved mastery on target KP
        gap = max(0.0, 0.85 - s_next[primary_kp])

        log["recommended"].append(a)
        log["rewards"].append(reward)
        log["correct"].append(info["correct"])
        log["predicted_gain"].append(pred_gain)
        log["predicted_probs"].append(p_correct)
        log["mastery_gaps"].append(gap)
        log["primary_kps"].append(info["primary_kp"])

        if train:
            next_candidates = env.candidate_actions() if not done else []
            next_phis = (np.stack([state_action_features(kg, s_next, r) for r in next_candidates])
                         if next_candidates else np.zeros((0, STATE_ACTION_DIM)))
            phi_sa = pruned_phis[idx]
            agent.buffer.push({
                "phi_sa": phi_sa, "reward": reward, "done": done,
                "next_cand_phis": next_phis,
            })
            trajectory.append({
                "cand_phis": pruned_phis, "chosen_idx": idx, "old_prob": prob,
                "reward": reward, "state_phi": state_features(s),
            })
        s = s_next

    log["sT"] = s.copy()
    if train:
        agent.train_step_ppo(trajectory)
        for _ in range(4):
            agent.train_step_q()
    return log


def run_episode_baseline(env: LearningPathEnv, kg: KnowledgeGraph, agent, agent_kind: str,
                          learner_id: str, init_scores: np.ndarray, rng: np.random.Generator):
    s = env.reset(init_scores, learner_id)
    log = {"recommended": [], "rewards": [], "correct": [], "predicted_gain": [],
           "predicted_probs": [], "mastery_gaps": [], "primary_kps": [], "s0": init_scores.copy()}
    correctness_model = env.correctness_model

    if agent_kind == "MC":
        agent.reset_learner(learner_id)

    done = False
    while not done:
        candidates = env.candidate_actions()
        if not candidates:
            break  # resource pool exhausted for this learner
        if agent_kind == "Rule":
            a = agent.choose(s, candidates, rng)
        elif agent_kind == "MC":
            a = agent.choose(s, candidates, rng, learner_id)
        elif agent_kind == "CF":
            a = agent.choose(s, candidates, rng, learner_id)
        elif agent_kind == "KG-H":
            a = agent.choose(s, candidates, rng)
        else:
            raise ValueError(agent_kind)

        ann = kg.resource_annotations[a]
        primary_kp = ann["primary"]
        p_correct = correctness_model(s[primary_kp])
        pred_gain = expected_gain(kg, s, a, correctness_model)
        s_next, reward, done, info = env.step(a)

        # Gap between mastery objective (0.85) and achieved mastery on target KP
        gap = max(0.0, 0.85 - s_next[primary_kp])

        if agent_kind == "MC":
            agent.observe(learner_id, info["primary_kp"])
        elif agent_kind == "CF":
            agent.observe(learner_id, a)

        log["recommended"].append(a)
        log["rewards"].append(reward)
        log["correct"].append(info["correct"])
        log["predicted_gain"].append(pred_gain)
        log["predicted_probs"].append(p_correct)
        log["mastery_gaps"].append(gap)
        log["primary_kps"].append(info["primary_kp"])
        s = s_next

    log["sT"] = s.copy()
    return log
