from __future__ import annotations

import numpy as np
from ..knowledge_graph import KnowledgeGraph

STATE_ACTION_DIM = 14
STATE_DIM = 4


def state_action_features(kg: KnowledgeGraph, s: np.ndarray, action: int) -> np.ndarray:
    ann = kg.resource_annotations[action]
    primary = ann["primary"]
    secondary = ann["secondary"]

    mastery_primary = s[primary]
    room_to_grow = 1.0 - mastery_primary
    mastery_secondary = float(np.mean([s[k] for k in secondary])) if secondary else mastery_primary

    prereqs = kg.prerequisites(primary)
    prereq_readiness = float(np.mean([s[p] for p in prereqs])) if prereqs else 1.0

    sem_row = kg.W[primary].copy()
    sem_row[kg.W[primary] == 1.0] = 0.0
    if sem_row.sum() > 0:
        semantic_context = float(np.average(s, weights=sem_row))
    else:
        semantic_context = float(np.mean(s))

    overall_mastery = float(np.mean(s))
    is_new = 1.0 if mastery_primary < 0.1 else 0.0

    gap_readiness = room_to_grow * prereq_readiness
    mastery_sq = mastery_primary ** 2
    urgency = 1.0 if (room_to_grow > 0.4 and prereq_readiness >= 0.7) else 0.0

    return np.array([
        1.0,
        mastery_primary,
        mastery_secondary,
        prereq_readiness,
        semantic_context,
        len(secondary) / 2.0,
        overall_mastery,
        room_to_grow,
        is_new,
        gap_readiness,
        mastery_sq,
        urgency,
        room_to_grow * semantic_context,
        prereq_readiness * overall_mastery
    ], dtype=np.float64)


def state_features(s: np.ndarray) -> np.ndarray:
    return np.array([
        1.0,
        float(np.mean(s)),
        float(np.std(s)),
        float(np.min(s)),
    ], dtype=np.float64)
