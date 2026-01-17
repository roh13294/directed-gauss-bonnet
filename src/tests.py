# src/tests.py

from typing import Tuple
import random
import numpy as np

from .counting import build_adj, count_transitive_and_feedback
from .curvature import verify_gauss_bonnet, curvature
from .graphs import planted_feedback_core

Edge = Tuple[int, int]


def random_digraph(n: int, p: float, seed=None):
    rng = random.Random(seed)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                edges.append((i, j))
    return n, edges


def test_gauss_bonnet_random(num_graphs=20):
    for g in range(num_graphs):
        n, edges = random_digraph(8, 0.2, seed=g)
        outN, inN = build_adj(n, edges)
        T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)
        assert verify_gauss_bonnet(n, edges, outN, inN, T_plus, T_minus), \
            f"GB identity failed on graph {g}"
    print("[OK] Gauss–Bonnet identity holds on random graphs.")


def test_planted_core():
    n, edges, core_nodes = planted_feedback_core(
        n_bg=30, n_core=10, p_bg=0.05, p_core=0.4, p_bridge=0.05, seed=10
    )
    outN, inN = build_adj(n, edges)
    T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)
    K = np.array(curvature(n, edges, outN, inN, T_plus, T_minus))

    core_mask = np.zeros(n, dtype=bool)
    core_mask[core_nodes] = True

    print("core avg:", K[core_mask].mean(), "bg avg:", K[~core_mask].mean())


if __name__ == "__main__":
    test_gauss_bonnet_random()
    test_planted_core()
