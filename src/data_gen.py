"""
Synthetic learner-population generator. Stands in for the diagnostic
entrance test described in 'Dataset construction and processing': each
simulated learner gets an initial mastery vector s(0) in [0,1]^N, drawn
from a Beta distribution so that most knowledge points start low while a
few (e.g. from prior schooling) start partially mastered.
"""
from __future__ import annotations

import numpy as np

from .knowledge_graph import KnowledgeGraph


def generate_learners(kg: KnowledgeGraph, n_learners: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    learners = []
    for i in range(n_learners):
        # each learner has an overall "prior proficiency" level that
        # shifts their whole Beta distribution, simulating heterogeneous
        # starting points across the population.
        prior = rng.uniform(0.05, 0.4)
        a = 1.0 + prior * 4
        b = 4.0
        scores = rng.beta(a, b, size=kg.n_knowledge_points)
        learner_id = f"learner_{i:04d}"
        learners.append((learner_id, scores))
    return learners


def split_learners(learners, train_frac=0.8, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(learners))
    rng.shuffle(idx)
    n = len(learners)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    to_list = lambda ids: [learners[i] for i in ids]
    return to_list(train_idx), to_list(val_idx), to_list(test_idx)
