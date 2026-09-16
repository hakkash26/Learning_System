#!/usr/bin/env python3
"""Quick sanity checks (not a full test suite) for the core building
blocks: knowledge graph construction, learner-state update bounds, and
one training step of the RL agent. Run with:

    python scripts/test_sanity.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.knowledge_graph import KnowledgeGraph
from src.learner_model import LearnerState
from src.environment import LearningPathEnv
from src.simulate import OursAgent, run_episode_ours


def test_knowledge_graph():
    kg = KnowledgeGraph(n_knowledge_points=12, n_resources=30, seed=1).build()
    assert kg.W.shape == (12, 12)
    assert kg.M.shape == (30, 12)
    # prerequisite entries must be exactly 1.0 (Eq. 2)
    for (i, j) in kg.prereq_edges:
        assert kg.W[i, j] == 1.0
    print("test_knowledge_graph: OK "
          f"({len(kg.prereq_edges)} prereq edges, {len(kg.sem_edges)} semantic edges)")


def test_learner_state_bounds():
    kg = KnowledgeGraph(n_knowledge_points=10, n_resources=20, seed=2).build()
    rng = np.random.default_rng(2)
    s0 = rng.uniform(0, 1, size=10)
    learner = LearnerState(kg, s0, "l0")
    for _ in range(50):
        r = rng.integers(0, kg.n_resources)
        learner.apply_interaction(int(r), correct=bool(rng.random() < 0.5))
        assert np.all(learner.s >= 0.0) and np.all(learner.s <= 1.0), "mastery left [0,1]"
    learner.advance_time(10)
    assert np.all(learner.s >= 0.0) and np.all(learner.s <= 1.0), "forgetting left [0,1]"
    print("test_learner_state_bounds: OK")


def test_one_training_episode():
    kg = KnowledgeGraph(n_knowledge_points=10, n_resources=25, seed=3).build()
    env = LearningPathEnv(kg, rng=np.random.default_rng(3))
    agent = OursAgent(kg, seed=3)
    rng = np.random.default_rng(3)
    s0 = np.random.default_rng(3).uniform(0, 0.3, size=10)
    log = run_episode_ours(env, kg, agent, "l0", s0, rng, train=True)
    assert len(log["recommended"]) > 0
    assert len(log["recommended"]) == len(log["rewards"])
    print(f"test_one_training_episode: OK ({len(log['recommended'])} steps, "
          f"sum reward={sum(log['rewards']):.3f})")


if __name__ == "__main__":
    test_knowledge_graph()
    test_learner_state_bounds()
    test_one_training_episode()
    print("\nAll sanity checks passed.")
