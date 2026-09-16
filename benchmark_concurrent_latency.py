"""
Concurrent Processing Benchmark for KG-RL (Java Learning Path Recommendation)
-------------------------------------------------------------------------------
What this measures:
    The candidate-generation + Q-network-based pruning + action-selection step
    of the KG-RL pipeline (src/environment.py, src/agents/q_learning.py,
    src/agents/features.py), under simulated concurrent load using Python
    threads issuing repeated calls.

What this does NOT measure:
    The PPO policy module specifically (requires torch). This script uses a
    lightweight numpy softmax stand-in for the final policy-selection step,
    matching the same computational shape (linear score -> softmax over the
    pruned candidate set) without requiring torch to be installed.

How to run:
    1. Place this file in the ROOT of your extracted project folder
       (the same folder that contains the `src/` directory).
    2. Make sure numpy is installed:  pip install numpy
    3. Run:  python benchmark_concurrent_latency.py
    4. The script prints a results table and also writes
       bench_results.json in the same folder.

Notes on reproducibility:
    - Latency and throughput numbers WILL differ from any numbers quoted
      elsewhere, because they depend on your machine's CPU speed and load
      at the time you run it. Report the numbers YOUR run actually prints.
    - The knowledge graph / environment configuration (24 knowledge points,
      90 resources) matches the project's default experimental setup in
      scripts/run_simulation.py. Change N_KP / N_RESOURCES below if you
      want to benchmark a different configuration.
"""

import sys
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")  # run this script from the project root

from src.knowledge_graph import KnowledgeGraph
from src.data_gen import generate_learners
from src.environment import LearningPathEnv
from src.agents.q_learning import QNetwork
from src.agents.features import state_action_features

# ---- configuration (edit if you want a different scale) ----
N_KP = 24
N_RESOURCES = 90
N_LEARNERS = 30          # small pool just to get a valid starting state
SEED = 42
WARMUP_CALLS = 30
SINGLE_CALL_SAMPLES = 300
CONCURRENCY_LEVELS = [1, 8, 32]
LOAD_TEST_DURATION_S = 4

def build_environment():
    kg = KnowledgeGraph(n_knowledge_points=N_KP, n_resources=N_RESOURCES, seed=SEED).build()
    learners = generate_learners(kg, n_learners=N_LEARNERS, seed=SEED)
    env = LearningPathEnv(kg, rng=np.random.default_rng(SEED))
    qnet = QNetwork(seed=SEED)
    state = env.reset(learners[0][1], learners[0][0])
    return kg, env, qnet, state

def make_call(kg, env, qnet, state):
    """One full candidate-generation -> Q-pruning -> selection call."""
    def one_call():
        t0 = time.perf_counter()
        candidates = env.candidate_actions()
        phis = np.stack([state_action_features(kg, state, r) for r in candidates])
        q_vals = qnet.q_batch(phis)
        order = np.argsort(-q_vals)
        top = order[: min(15, len(candidates))]
        pruned_phis = phis[top]
        scores = pruned_phis @ qnet.theta
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        _ = np.random.choice(len(probs), p=probs)
        return time.perf_counter() - t0
    return one_call

def load_test(one_call, n_threads, duration_s):
    stop_at = time.time() + duration_s
    latencies = []

    def worker():
        local = []
        while time.time() < stop_at:
            local.append(one_call() * 1000)  # ms
        return local

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(worker) for _ in range(n_threads)]
        for f in futures:
            latencies.extend(f.result())
    return np.array(latencies)

def main():
    print("Building knowledge graph and environment...")
    kg, env, qnet, state = build_environment()
    one_call = make_call(kg, env, qnet, state)

    print(f"Warming up ({WARMUP_CALLS} calls)...")
    for _ in range(WARMUP_CALLS):
        one_call()

    print(f"Measuring single-call latency ({SINGLE_CALL_SAMPLES} calls)...")
    single_ms = np.array([one_call() for _ in range(SINGLE_CALL_SAMPLES)]) * 1000

    results = {
        "single_call_ms": {
            "mean": float(single_ms.mean()),
            "p95": float(np.percentile(single_ms, 95)),
            "min": float(single_ms.min()),
            "max": float(single_ms.max()),
        }
    }

    print("\n%-12s %-14s %-14s %-22s" % ("Threads", "Mean (ms)", "P95 (ms)", "Throughput (calls/s)"))
    for n in CONCURRENCY_LEVELS:
        print(f"Running load test at {n} concurrent thread(s) for {LOAD_TEST_DURATION_S}s...")
        lat = load_test(one_call, n, LOAD_TEST_DURATION_S)
        mean_ms = float(lat.mean())
        p95_ms = float(np.percentile(lat, 95))
        throughput = round(len(lat) / LOAD_TEST_DURATION_S, 1)
        results[f"threads_{n}"] = {
            "n_calls": int(len(lat)),
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "throughput_calls_per_sec": throughput,
        }
        print("%-12d %-14.2f %-14.2f %-22.1f" % (n, mean_ms, p95_ms, throughput))

    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFull results written to bench_results.json")
    print("\nSingle-call latency (uncontended):")
    print(json.dumps(results["single_call_ms"], indent=2))

if __name__ == "__main__":
    main()
