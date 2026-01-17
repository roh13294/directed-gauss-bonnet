# src/counting.py

from typing import List, Set, Tuple

Edge = Tuple[int, int]


def build_adj(n: int, edges: List[Edge]):
    outN: List[Set[int]] = [set() for _ in range(n)]
    inN: List[Set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            continue
        outN[u].add(v)
        inN[v].add(u)
    return outN, inN


def count_transitive_and_feedback(
    n: int,
    outN: List[Set[int]],
    inN: List[Set[int]],
):
    T_plus = [0] * n
    T_minus = [0] * n

    # transitive: i → j → k and i → k
    for j in range(n):
        for i in inN[j]:
            common = outN[i].intersection(outN[j])
            for k in common:
                T_plus[i] += 1
                T_plus[j] += 1
                T_plus[k] += 1

    # feedback: i → j → k → i
    for v in range(n):
        for i in outN[v]:
            for j in outN[i]:
                if v in outN[j]:
                    T_minus[v] += 1
                    T_minus[i] += 1
                    T_minus[j] += 1

    return T_plus, T_minus


def total_faces_from_incidence(T_plus, T_minus):
    F_plus = sum(T_plus) // 3
    F_minus = sum(T_minus) // 3
    return F_plus, F_minus
