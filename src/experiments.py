# src/experiments.py

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .counting import build_adj, count_transitive_and_feedback, total_faces_from_incidence
from .curvature import curvature, chi_lambda
from .graphs import to_networkx_digraph

Edge = Tuple[int, int]


def compute_curvature_all(n, edges, lam=1.0):
    outN, inN = build_adj(n, edges)
    T_plus, T_minus = count_transitive_and_feedback(n, outN, inN)
    K = curvature(n, edges, outN, inN, T_plus, T_minus, lam)
    F_plus, F_minus = total_faces_from_incidence(T_plus, T_minus)
    chi = chi_lambda(n, edges, F_plus, F_minus, lam)
    return K, chi, (F_plus, F_minus)


def plot_curvature_heatmap(n, edges, K, title="Curvature Heatmap"):
    G = to_networkx_digraph(n, edges)
    pos = nx.spring_layout(G, seed=42)
    values = np.array(K)

    plt.figure(figsize=(6, 5))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=values, cmap="coolwarm", node_size=330)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")
    nx.draw_networkx_labels(G, pos)

    plt.colorbar(nodes, label="K_lambda(v)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
