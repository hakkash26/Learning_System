"""
Knowledge graph construction and learning-resource structuring.

Implements the constructs described in the paper's "Knowledge graph
construction and learning resource structuring" section:

  - K = {k1, ..., kN}: knowledge points (nodes)
  - E = E_prereq (directed) U E_sem (undirected, weighted by Jaccard similarity)
  - Adjacency matrix W (Eq. 2): asymmetric for prerequisites, symmetric for
    semantic edges
  - Resource-to-knowledge mapping matrix M (Eq. 3): binary M in {0,1}^{M x N}
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


TOPICS = [
    "Introduction to Java & Environment Setup",
    "Variables, Data Types, and Operators",
    "Control Flow Statements",
    "Methods and Scope",
    "Arrays and String Manipulation",
    "Introduction to Classes and Objects",
    "Encapsulation and Access Modifiers",
    "Inheritance and Method Overriding",
    "Polymorphism and Interfaces",
    "Abstract Classes and Inner Classes",
    "Exception Handling Basics",
    "Custom Exceptions and Throws",
    "Java Generics",
    "Java Collections: Lists & Sets",
    "Java Collections: Maps",
    "File I/O and NIO",
    "Object Serialization",
    "Functional Programming",
    "Streams API",
    "Multithreading Basics",
    "Synchronization",
    "Thread Pools and Executors",
    "JDBC Database Access",
    "Reflection API",
]


@dataclass
class KnowledgeGraph:
    n_knowledge_points: int = 40
    n_resources: int = 150
    theta: float = 0.3          # semantic-edge Jaccard threshold (paper: 0.3)
    branching: int = 2          # avg number of prerequisites per node
    seed: int = 42

    # populated by build()
    knowledge_points: list = field(default_factory=list)
    topics: list = field(default_factory=list)
    prereq_edges: set = field(default_factory=set)   # (i -> j)
    sem_edges: dict = field(default_factory=dict)     # (i, j) -> weight
    W: np.ndarray = None
    M: np.ndarray = None
    resource_sets: dict = field(default_factory=dict)  # ki -> set(resource idx)
    graph: nx.DiGraph = None

    def build(self) -> "KnowledgeGraph":
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)
        N, Mn = self.n_knowledge_points, self.n_resources

        self.knowledge_points = [f"k{i}" for i in range(N)]
        # assign topics roughly evenly, topics broadly ordered by difficulty
        self.topics = [TOPICS[i % len(TOPICS)] for i in range(N)]

        # ---- 1) Prerequisite edges: hybrid expert + data-driven -------
        # Expert seed: curriculum-like layering (a topological "curriculum
        # order") -- node i can only require nodes with smaller index,
        # approximating instructor-defined logical dependency.
        edges = {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (5, 10),
            (10, 11),
            (8, 12),
            (12, 13),
            (13, 14),
            (10, 15),
            (6, 16),
            (9, 17),
            (13, 18),
            (17, 18),
            (2, 19),
            (19, 20),
            (20, 21),
            (15, 22),
            (8, 23),
        }
        self.prereq_edges = {(i, j) for (i, j) in edges if i < N and j < N}

        # ---- 2) Resource <-> knowledge-point mapping -------------------
        # Each resource is annotated with 1-3 knowledge points (primary +
        # optional secondary/peripheral), used to build M and to derive
        # semantic co-occurrence.
        self.resource_sets = {i: set() for i in range(N)}
        resource_annotations = []  # list of dict: {primary, secondary:[...]}
        for r in range(Mn):
            primary = rng.randrange(N)
            n_secondary = rng.choice([0, 0, 1, 1, 2])
            secondary = rng.sample(
                [k for k in range(N) if k != primary], k=min(n_secondary, N - 1)
            )
            resource_annotations.append({"primary": primary, "secondary": secondary})
            self.resource_sets[primary].add(r)
            for s in secondary:
                self.resource_sets[s].add(r)
        self.resource_annotations = resource_annotations

        self.M = np.zeros((Mn, N), dtype=np.float64)
        for r, ann in enumerate(resource_annotations):
            self.M[r, ann["primary"]] = 1.0
            for s in ann["secondary"]:
                self.M[r, s] = 1.0

        # ---- 3) Semantic edges via Jaccard index (Eq. 1) ---------------
        self.sem_edges = {}
        for i, j in itertools.combinations(range(N), 2):
            if (i, j) in self.prereq_edges or (j, i) in self.prereq_edges:
                continue  # a pair is either a prerequisite or a semantic edge, not both
            Ri, Rj = self.resource_sets[i], self.resource_sets[j]
            union = Ri | Rj
            if not union:
                continue
            w = len(Ri & Rj) / len(union)
            if w >= self.theta:
                self.sem_edges[(i, j)] = w

        # ---- 4) Adjacency matrix W (Eq. 2) ------------------------------
        self.W = np.zeros((N, N), dtype=np.float64)
        for (i, j) in self.prereq_edges:
            self.W[i, j] = 1.0
        for (i, j), w in self.sem_edges.items():
            self.W[i, j] = w
            self.W[j, i] = w

        # ---- 5) networkx graph (for prerequisite feasibility queries) --
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(N))
        self.graph.add_edges_from(self.prereq_edges)

        return self

    def _augment_prereqs_from_synthetic_sequences(self, np_rng, n_sequences=300):
        """Simulate learner mastery sequences consistent with the current
        (expert) prerequisite graph, then statistically test whether any
        additional pair (ki, kj) shows a strong "ki learned before kj, and
        P(master kj) rises after ki" pattern; add such edges to E_prereq.
        This mirrors the paper's data-driven edge augmentation without
        requiring an external dataset.
        """
        N = self.n_knowledge_points
        order = list(nx.topological_sort(nx.DiGraph(self.prereq_edges)) ) if self.prereq_edges else list(range(N))
        # ensure all nodes included even if isolated in prereq graph
        missing = [k for k in range(N) if k not in order]
        order = order + missing

        precede_count = np.zeros((N, N))
        after_master_count = np.zeros((N, N))
        after_total_count = np.zeros((N, N))

        for _ in range(n_sequences):
            # random subsequence (learner covers a random subset, in
            # topological-consistent order with small shuffling noise)
            seq = order[:]
            # local shuffles to introduce noise
            for _ in range(N // 4):
                a = np_rng.integers(0, N - 1)
                if np_rng.random() < 0.3:
                    seq[a], seq[a + 1] = seq[a + 1], seq[a]
            mastered_before = set()
            for idx, k in enumerate(seq):
                for m in mastered_before:
                    precede_count[m, k] += 1
                mastered = np_rng.random() < 0.85  # simulate mastery event
                if mastered:
                    for m in mastered_before:
                        after_total_count[m, k] += 1
                        if np_rng.random() < 0.9:  # mastery "boost" signal
                            after_master_count[m, k] += 1
                    mastered_before.add(k)

        with np.errstate(divide="ignore", invalid="ignore"):
            boost_rate = np.where(
                after_total_count > 0, after_master_count / after_total_count, 0
            )
        threshold_count = 0.6 * n_sequences
        threshold_rate = 0.88
        max_augment = max(3, N // 4)  # cap augmentation so the graph stays sparse
        added = 0
        for i in range(N):
            for j in range(N):
                if added >= max_augment:
                    break
                if i == j:
                    continue
                if (
                    precede_count[i, j] >= threshold_count
                    and boost_rate[i, j] >= threshold_rate
                    and (i, j) not in self.prereq_edges
                    and (j, i) not in self.prereq_edges
                ):
                    self.prereq_edges.add((i, j))
                    added += 1

    # ------------------------------------------------------------------
    def knowledge_points_of_resource(self, r: int) -> dict:
        """Return {knowledge_point_index: importance_weight eta} for a
        resource, following the paper's eta rule: primary=1.0,
        core-secondary=0.6, peripheral=0.4 (secondary items are split
        evenly between the two roles for simplicity)."""
        ann = self.resource_annotations[r]
        weights = {ann["primary"]: 1.0}
        for idx, k in enumerate(ann["secondary"]):
            weights[k] = 0.6 if idx == 0 else 0.4
        return weights

    def prerequisites(self, k: int) -> list:
        return [i for (i, j) in self.prereq_edges if j == k]

    def is_prerequisite_feasible(self, mastery: np.ndarray, k: int, tau: float = 0.5) -> bool:
        """A knowledge point (and hence resources primarily tied to it) is
        feasible if all of its prerequisite knowledge points are mastered
        above threshold tau."""
        for p in self.prerequisites(k):
            if mastery[p] < tau:
                return False
        return True
