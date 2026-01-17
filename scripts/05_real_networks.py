import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.counting import build_adj, count_transitive_and_feedback
from src.curvature import curvature
from src.experiments import (
    visualize_curvature_digraph,
    plot_curvature_histogram,
    print_top_negative_nodes
)


def run_real_network(path="my_edges.csv", lam=1.0):
    df = pd.read_csv(path)
    edges = list(zip(df["src"], df["dst"]))

    nodes = sorted(set(df["src"]).union(set(df["dst"])))
    node_map = {v: i for i, v in enumerate(nodes)}
    edges = [(node_map[u], node_map[v]) for u, v in edges]

    n = len(nodes)
    outN, inN = build_adj(n, edges)
    T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)

    K = np.array(curvature(n, edges, outN, inN, T_plus, T_minus, lam))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    visualize_curvature_digraph(
        G, list(range(n)), K, edges,
        title="Curvature on real directed network",
        save_path="figs/real_network_curvature.png"
    )

    plot_curvature_histogram(
        K,
        title="Curvature distribution (real network)",
        save_path="figs/real_network_hist.png"
    )

    print_top_negative_nodes(list(range(n)), K, k=15)


if __name__ == "__main__":
    run_real_network()
