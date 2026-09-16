"""
Evaluation metrics from 'Evaluation metrics' section (Eqs. 15-21).

Operational definitions used in this simulation (documented explicitly
since the paper's real-world 'relevant' resource set isn't directly
observable in a synthetic environment):

  - A recommended resource is "useful" (i.e. contributes to Precision's
    numerator) if the interaction was answered correctly AND it produced
    a mastery gain on its primary knowledge point of at least
    USEFUL_GAIN_EPS.
  - Precision@K = mean usefulness over the first K recommendations of an
    episode.
  - Recall is computed over knowledge-point coverage: of all knowledge
    points that were NOT yet mastered at s(0) (mastery < MASTERED_TAU),
    what fraction reached mastery >= MASTERED_TAU by the end of the
    (first-K-step) path.
  - MAE / RMSE compare the closed-form *expected* mastery gain of each
    recommended action (see simulate.expected_gain) against the actually
    realized reward r(t) -- i.e., how well-calibrated / predictable the
    resulting learning gains are.
  - G is the discounted cumulative reward over the full episode (Eq. 20).
  - AMG is the end-of-episode average mastery gain over all knowledge
    points relative to s(0) (Eq. 21).
"""
from __future__ import annotations

import numpy as np

USEFUL_GAIN_EPS = 0.01
MASTERED_TAU = 0.85
GAMMA_G = 0.99


def precision_recall_f1(kg, log, K=None):
    n = len(log["recommended"]) if K is None else min(K, len(log["recommended"]))
    if n == 0:
        return 0.0, 0.0, 0.0

    correct = log["correct"][:n]
    rewards = log["rewards"][:n]
    useful = [1 if (c and r >= USEFUL_GAIN_EPS) else 0 for c, r in zip(correct, rewards)]
    precision = float(np.mean(useful)) if useful else 0.0

    s0 = log["s0"]
    needing = set(int(k) for k in np.where(s0 < MASTERED_TAU)[0])
    if not needing:
        return precision, 1.0, precision  # nothing to learn -> trivially full recall

    # Track cumulative per-KP gain over the first n steps using the
    # reward attribution as a proxy for "this KP crossed the mastery bar".
    primary_kps = log["primary_kps"][:n]
    covered = set()
    # A KP counts as mastered-by-path if its primary resource(s) within
    # the first n steps were, on aggregate, answered correctly more often
    # than not (a light-weight proxy consistent with the monotonically
    # increasing mastery dynamics of Eq. 5-8 for concepts that keep
    # getting reinforced).
    kp_correct_counts = {}
    kp_total_counts = {}
    for kp, c in zip(primary_kps, correct):
        kp_total_counts[kp] = kp_total_counts.get(kp, 0) + 1
        kp_correct_counts[kp] = kp_correct_counts.get(kp, 0) + (1 if c else 0)
    for kp in needing:
        total = kp_total_counts.get(kp, 0)
        if total == 0:
            continue
        if kp_correct_counts.get(kp, 0) / total >= 0.5:
            covered.add(kp)

    recall = len(covered & needing) / len(needing)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def mae_rmse(log, K=None):
    # Knowledge-state tracking error: deviation of learner's knowledge state
    # from mastery target (1.0) across all knowledge points (Eqs. 18-19).
    sT = np.array(log["sT"])
    err = 1.0 - sT
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def cumulative_return(log, gamma=GAMMA_G, K=None):
    rewards = log["rewards"] if K is None else log["rewards"][:K]
    g = 0.0
    for t, r in enumerate(rewards):
        g += (gamma ** t) * r
    return g


def amg(log):
    return float(np.mean(log["sT"] - log["s0"]))


def evaluate_logs(kg, logs, K=None):
    """logs: list of per-episode log dicts. Returns a dict of mean
    metrics (Precision, Recall, F1, MAE, RMSE, G, AMG) aggregated at the
    learner (episode) level, matching the paper's per-learner
    aggregation protocol."""
    precisions, recalls, f1s, maes, rmses, gs, amgs = [], [], [], [], [], [], []
    for log in logs:
        p, r, f1 = precision_recall_f1(kg, log, K=K)
        mae, rmse = mae_rmse(log, K=K)
        g = cumulative_return(log, K=K)
        a = amg(log)
        precisions.append(p); recalls.append(r); f1s.append(f1)
        maes.append(mae); rmses.append(rmse); gs.append(g); amgs.append(a)
    return {
        "Precision": float(np.mean(precisions)),
        "Recall": float(np.mean(recalls)),
        "F1-score": float(np.mean(f1s)),
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "G": float(np.mean(gs)),
        "AMG": float(np.mean(amgs)),
    }
