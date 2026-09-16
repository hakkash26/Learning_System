"""
Learner knowledge-state modeling: direct feedback update, graph-based
propagation, sigmoid squashing, and exponential forgetting -- Eqs. 4-8 of
the paper.
"""
from __future__ import annotations

import numpy as np

from .knowledge_graph import KnowledgeGraph


def sigmoid_center(x: np.ndarray) -> np.ndarray:
    """Squash an already-bounded-ish quantity smoothly back into [0, 1].
    We use a centered logistic curve so that inputs near 0.5 pass through
    almost unchanged while overshoot beyond [0, 1] is compressed, matching
    the paper's use of sigma(.) in Eq. 7."""
    return 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))


class LearnerState:
    """Tracks a single learner's mastery vector s(t) and last-visit
    timestamps, and implements the update mechanism of Section 'Learner
    knowledge state modeling method'."""

    ALPHA_PLUS = 0.56     # positive update rate (Eq. 5)
    ALPHA_MINUS = 0.08    # negative update rate (Eq. 5)
    FORGET_MU = 0.08      # forgetting rate (Eq. 8)
    FORGET_AFTER_DAYS = 3

    def __init__(self, kg: KnowledgeGraph, init_scores: np.ndarray, learner_id: str):
        self.kg = kg
        self.learner_id = learner_id
        self.s = init_scores.copy().astype(np.float64)
        self.s0 = init_scores.copy().astype(np.float64)  # remember for AMG
        self.last_visit_day = np.zeros(kg.n_knowledge_points, dtype=np.float64)
        self.day = 0.0

    # ------------------------------------------------------------------
    def advance_time(self, delta_days: float):
        """Apply exponential forgetting (Eq. 8) to knowledge points not
        revisited for >= FORGET_AFTER_DAYS."""
        self.day += delta_days
        stale = (self.day - self.last_visit_day) >= self.FORGET_AFTER_DAYS
        if np.any(stale):
            self.s[stale] = self.s[stale] * np.exp(-self.FORGET_MU * (self.day - self.last_visit_day[stale]))

    # ------------------------------------------------------------------
    def apply_interaction(self, resource_id: int, correct: bool) -> np.ndarray:
        """Apply one learning interaction with `resource_id`. Returns the
        mastery-change vector delta_s(t) = s(t+1) - s(t) (Eq. 10 input)."""
        eta = self.kg.knowledge_points_of_resource(resource_id)  # {k: weight}
        direct = np.zeros(self.kg.n_knowledge_points, dtype=np.float64)
        for k, w in eta.items():
            if correct:
                direct[k] = self.ALPHA_PLUS * (1.0 - self.s[k]) * w
            else:
                direct[k] = -self.ALPHA_MINUS * self.s[k] * w

        # graph-based propagation (Eq. 6): spread change to neighbours
        # weighted by adjacency (prerequisite + semantic edges).
        prop = self.kg.W.T @ direct  # column j: sum_k W[k,j]*direct[k]

        s_before = self.s.copy()
        raw = self.s + direct + prop
        self.s = np.clip(sigmoid_center(raw), 0.0, 1.0)

        for k in eta:
            self.last_visit_day[k] = self.day

        return self.s - s_before

    # ------------------------------------------------------------------
    def amg(self) -> float:
        """Average Mastery Gain (Eq. 21) relative to this learner's own
        initial state."""
        return float(np.mean(self.s - self.s0))
