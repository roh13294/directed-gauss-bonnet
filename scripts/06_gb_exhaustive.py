import itertools
import numpy as np

from src.counting import build_adj, count_transitive_and_feedback, total_faces_from_incidence
from src.curvature import curvature


def run_exhaustive(n=5, lam=1.0):
    nodes = list(range(n))
    all_edges = [(i, j) for i in nodes for j in nodes if i != j]

    passed = 0
    tested = 0

    for k in range(len(all_edges) + 1):
        for subset in itertools.combinations(all_edges, k):
            edges = list(subset)
            outN, inN = build_adj(n, edges)
            T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)
            Fp, Fm = total_faces_from_incidence(T_plus, T_minus)

            K = np.array(curvature(n, edges, outN, inN, T_plus, T_minus, lam))

            chi = n - len(edges) + Fp - Fm
            if abs(K.sum() - chi) < 1e-9:
                passed += 1
            tested += 1

    print(f"Exhaustive GB check (n={n})")
    print(f"Passed {passed} / {tested}")


if __name__ == "__main__":
    run_exhaustive()
