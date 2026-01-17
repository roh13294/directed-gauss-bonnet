import numpy as np
import matplotlib.pyplot as plt

from src.counting import build_adj, count_transitive_and_feedback
from src.curvature import curvature


def random_digraph(n, p, seed=None):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                edges.append((i, j))
    return edges


def run_finite_size_scaling():
    lam = 1.0
    ps = np.linspace(0.05, 0.2, 8)
    sizes = [20, 40, 60, 80]

    plt.figure(figsize=(6,4))

    for n in sizes:
        means = []
        for p in ps:
            edges = random_digraph(n, p, seed=123)
            outN, inN = build_adj(n, edges)
            T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)
            K = np.array(curvature(n, edges, outN, inN, T_plus, T_minus, lam))
            means.append(K.mean())
        plt.plot(ps, means, marker="o", label=f"n={n}")

    plt.axhline(0, linestyle="--", color="k")
    plt.xlabel("edge probability p")
    plt.ylabel("mean curvature")
    plt.title("Finite-size scaling of curvature transition")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/finite_size_scaling.png", dpi=300)


if __name__ == "__main__":
    run_finite_size_scaling()
