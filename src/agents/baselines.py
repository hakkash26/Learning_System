"""
Baseline recommenders used for comparison in Table 1 / Table 3 of the
paper: Markov Chain (MC), Collaborative Filtering (CF), Rule-based, and
a Knowledge-Graph-only heuristic (KG-H) that enforces prerequisite
feasibility and semantic proximity without any learned policy. These are
intentionally simple, transparent baselines -- the point of the
simulation is to reproduce the *qualitative ordering* (Rule < CF < MC <
KG-H < KG-RL) described in the paper, not to re-implement AKT / LightGCN /
TA-RL / cDQN exactly (see README for scope notes).
"""
from __future__ import annotations

import numpy as np

from ..knowledge_graph import KnowledgeGraph


class RuleBasedAgent:
    """Recommend the first prerequisite-feasible, not-yet-mastered
    resource in curriculum (topological) order -- a static, non-adaptive
    policy."""

    name = "Rule"

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.order = sorted(range(kg.n_resources),
                             key=lambda r: kg.resource_annotations[r]["primary"])

    def choose(self, s: np.ndarray, candidates: list, rng: np.random.Generator) -> int:
        for r in self.order:
            if r in candidates:
                return r
        return candidates[0]


class MarkovChainAgent:
    """Learns P(next primary KP = j | current focus = i) from training
    episodes, then greedily recommends a candidate whose primary KP has
    the highest transition probability from the learner's most recently
    engaged knowledge point."""

    name = "MC"

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        N = kg.n_knowledge_points
        self.counts = np.ones((N, N)) * 1e-3  # Laplace smoothing
        self.last_kp = {}

    def observe(self, learner_id: str, primary_kp: int):
        prev = self.last_kp.get(learner_id)
        if prev is not None:
            self.counts[prev, primary_kp] += 1.0
        self.last_kp[learner_id] = primary_kp

    def reset_learner(self, learner_id: str):
        self.last_kp[learner_id] = None

    def choose(self, s: np.ndarray, candidates: list, rng: np.random.Generator,
               learner_id: str) -> int:
        prev = self.last_kp.get(learner_id)
        if prev is None:
            # fall back to global mastery-gap heuristic
            scores = {r: 1.0 - s[self.kg.resource_annotations[r]["primary"]] for r in candidates}
        else:
            trans = self.counts[prev]
            scores = {r: trans[self.kg.resource_annotations[r]["primary"]] for r in candidates}
        return max(scores, key=scores.get)


class CollaborativeFilteringAgent:
    """Item-item CF over a learner x resource interaction matrix built
    from training episodes: recommend the candidate most similar
    (cosine) to resources the learner has already engaged with."""

    name = "CF"

    def __init__(self, kg: KnowledgeGraph, n_learners_hint: int = 200):
        self.kg = kg
        self.interactions = {}  # learner_id -> set(resource_id)
        self.item_sim = None

    def observe(self, learner_id: str, resource_id: int):
        self.interactions.setdefault(learner_id, set()).add(resource_id)

    def fit_item_similarity(self):
        M = self.kg.n_resources
        learners = list(self.interactions.keys())
        if not learners:
            self.item_sim = np.eye(M)
            return
        mat = np.zeros((len(learners), M))
        for li, lid in enumerate(learners):
            for r in self.interactions[lid]:
                mat[li, r] = 1.0
        norm = np.linalg.norm(mat, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        matn = mat / norm
        self.item_sim = matn.T @ matn  # M x M cosine similarity

    def choose(self, s: np.ndarray, candidates: list, rng: np.random.Generator,
               learner_id: str) -> int:
        if self.item_sim is None:
            self.fit_item_similarity()
        seen = self.interactions.get(learner_id, set())
        if not seen:
            # cold start: pick candidate with globally most co-occurring peers
            scores = {r: float(self.item_sim[r].sum()) for r in candidates}
        else:
            scores = {}
            for r in candidates:
                scores[r] = float(sum(self.item_sim[r, s_] for s_ in seen))
        return max(scores, key=scores.get)


class KGHeuristicAgent:
    """Pure knowledge-graph heuristic (no learned policy): scores each
    feasible candidate by (a) how under-mastered its primary concept is
    and (b) semantic proximity to concepts the learner has already
    mastered, then picks the top score. Represents 'KG-H' in Table 3."""

    name = "KG-H"

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def choose(self, s: np.ndarray, candidates: list, rng: np.random.Generator) -> int:
        best_r, best_score = None, -1e9
        for r in candidates:
            primary = self.kg.resource_annotations[r]["primary"]
            gap = 1.0 - s[primary]
            sem_row = self.kg.W[primary]
            sem_context = float(np.mean(s[sem_row > 0])) if np.any(sem_row > 0) else 0.0
            score = 0.7 * gap + 0.3 * sem_context
            if score > best_score:
                best_r, best_score = r, score
        return best_r
