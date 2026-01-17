# src/graphs.py

from typing import List, Tuple
import random
import networkx as nx

Edge = Tuple[int, int]


def dag_gnp(n: int, p: float, seed=None):
    rng = random.Random(seed)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                edges.append((i, j))
    return n, edges


def planted_feedback_core(n_bg, n_core, p_bg, p_core, p_bridge, seed=None):
    rng = random.Random(seed)
    n = n_bg + n_core
    edges = []

    # background DAG
    for i in range(n_bg):
        for j in range(i+1, n_bg):
            if rng.random() < p_bg:
                edges.append((i, j))

    core_nodes = list(range(n_bg, n))

    # dense core with cycles
    for u in core_nodes:
        for v in core_nodes:
            if u != v and rng.random() < p_core:
                edges.append((u, v))

    # bridges
    for u in range(n_bg):
        for v in core_nodes:
            if rng.random() < p_bridge:
                if rng.random() < 0.5:
                    edges.append((u, v))
                else:
                    edges.append((v, u))

    edges = list(dict.fromkeys(edges))
    return n, edges, core_nodes


def to_networkx_digraph(n: int, edges: List[Edge]):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G
