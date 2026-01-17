import numpy as np
import matplotlib.pyplot as plt

from src.counting import (
    build_adj,
    count_transitive_and_feedback,
    total_faces_from_incidence
)
from src.curvature import curvature


def random_digraph(n, p, seed=None):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                edges.append((i, j))
    return n, edges


def run_phase_transition():
    n = 40
    lam = 1.0
    ps = np.linspace(0.01, 0.25, 13)

    F_minus = []
    chi_vals = []
    mean_K = []
    frac_neg = []

    for p in ps:
        n, edges = random_digraph(n, p, seed=42)
        outN, inN = build_adj(n, edges)

        T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)
        Fp, Fm = total_faces_from_incidence(T_plus, T_minus)

        K = np.array(curvature(n, edges, outN, inN, T_plus, T_minus, lam))

        F_minus.append(Fm)
        chi_vals.append(n - len(edges) + Fp - Fm)
        mean_K.append(K.mean())
        frac_neg.append((K < 0).mean())

    # ----- plots -----
    plt.figure(figsize=(6,4))
    plt.plot(ps, F_minus, marker="o")
    plt.xlabel("edge probability p")
    plt.ylabel("avg feedback faces")
    plt.title("Emergence of feedback triangles")
    plt.tight_layout()
    plt.savefig("figs/phase_feedback_faces.png", dpi=300)

    plt.figure(figsize=(6,4))
    plt.plot(ps, mean_K, marker="o")
    plt.axhline(0, linestyle="--", color="k")
    plt.xlabel("edge probability p")
    plt.ylabel("mean curvature")
    plt.title("Mean curvature vs p")
    plt.tight_layout()
    plt.savefig("figs/phase_mean_curvature.png", dpi=300)

    plt.figure(figsize=(6,4))
    plt.plot(ps, frac_neg, marker="o")
    plt.xlabel("edge probability p")
    plt.ylabel("fraction negative curvature")
    plt.title("Negative curvature fraction")
    plt.tight_layout()
    plt.savefig("figs/phase_frac_negative.png", dpi=300)


if __name__ == "__main__":
    run_phase_transition()
