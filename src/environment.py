"""
Learning-path recommendation cast as a Markov Decision Process
(S, A, P, R, gamma) -- see 'Reinforcement learning-based modeling for
personalized learning path recommendation'.

State  s(t): learner mastery vector (Eq. 9)
Action a(t): a learning resource id
Transition P(s'|s,a): LearnerState.apply_interaction (+ forgetting)
Reward R(s,a): weighted mastery-gain sum over involved knowledge points
               (Eq. 10)
"""
from __future__ import annotations

import numpy as np

from .knowledge_graph import KnowledgeGraph
from .learner_model import LearnerState

PREREQ_TAU = 0.5          # mastery threshold to unlock a knowledge point
MASTERED_TAU = 0.85       # a resource whose primary KP is already mastered
                            # this well is de-duplicated out of candidates


class LearningPathEnv:
    """One episode == one learner's guided study session sequence."""

    def __init__(self, kg: KnowledgeGraph, correctness_model=None, rng=None,
                 max_steps: int | None = None):
        self.kg = kg
        self.rng = rng or np.random.default_rng()
        # correctness_model(mastery_of_primary_kp) -> P(correct); defaults
        # to a simple logistic function of current mastery, i.e. better
        # mastery -> more likely to answer correctly.
        self.correctness_model = correctness_model or (
            lambda m: float(np.clip(0.35 + 0.6 * m, 0.05, 0.97))
        )
        # scale episode length to the size of the knowledge graph so that
        # a learner has a realistic chance to work through most concepts.
        self.max_steps = max_steps or max(40, int(2.5 * kg.n_knowledge_points))
        self.learner: LearnerState | None = None
        self.recommended_this_episode: set[int] = set()
        self.t = 0

    # ------------------------------------------------------------------
    def reset(self, init_scores: np.ndarray, learner_id: str) -> np.ndarray:
        self.learner = LearnerState(self.kg, init_scores, learner_id)
        self.recommended_this_episode = set()
        self.t = 0
        return self.learner.s.copy()

    # ------------------------------------------------------------------
    def candidate_actions(self) -> list[int]:
        """Prerequisite-feasible, de-duplicated candidate set A_t (step i
        of the prune-then-select workflow)."""
        s = self.learner.s
        cands = []
        for r, ann in enumerate(self.kg.resource_annotations):
            if r in self.recommended_this_episode:
                continue
            primary = ann["primary"]
            if s[primary] >= MASTERED_TAU:
                continue  # already mastered -> drop (dedup by mastery)
            if not self.kg.is_prerequisite_feasible(s, primary, PREREQ_TAU):
                continue
            cands.append(r)
        if not cands:  # fallback: allow anything not yet recommended
            cands = [r for r in range(self.kg.n_resources)
                      if r not in self.recommended_this_episode]
        return cands

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.learner is not None, "call reset() first"
        ann = self.kg.resource_annotations[action]
        primary = ann["primary"]
        p_correct = self.correctness_model(self.learner.s[primary])
        correct = self.rng.random() < p_correct

        delta = self.learner.apply_interaction(action, correct)
        eta = self.kg.knowledge_points_of_resource(action)
        reward = sum(w * delta[k] for k, w in eta.items())

        self.recommended_this_episode.add(action)
        # Mostly same-session study bursts, with only an occasional small
        # time gap (so the exponential forgetting mechanism, Eq. 8, is
        # exercised without dominating a single continuous episode).
        self.learner.advance_time(delta_days=self.rng.choice([0] * 19 + [1]))
        self.t += 1

        done = self.t >= self.max_steps or float(np.mean(self.learner.s)) >= 0.95
        info = {"correct": correct, "primary_kp": primary}
        return self.learner.s.copy(), float(reward), done, info
