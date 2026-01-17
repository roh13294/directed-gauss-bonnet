# src/curvature.py

from typing import List, Set, Tuple
from .counting import total_faces_from_incidence

Edge = Tuple[int, int]


def degrees(n: int, outN: List[Set[int]], inN: List[Set[int]]):
    din = [len(inN[v]) for v in range(n)]
    dout = [len(outN[v]) for v in range(n)]
    return din, dout


def chi_lambda(n, edges, F_plus, F_minus, lam=1.0):
    return n - len(edges) + F_plus - lam * F_minus


def curvature(n, edges, outN, inN, T_plus, T_minus, lam=1.0):
    din, dout = degrees(n, outN, inN)
    K = [0.0] * n
    for v in range(n):
        K[v] = 1 - 0.5 * (din[v] + dout[v]) + (T_plus[v] - lam * T_minus[v]) / 3
    return K


def verify_gauss_bonnet(n, edges, outN, inN, T_plus, T_minus, lam=1.0, tol=1e-9):
    K = curvature(n, edges, outN, inN, T_plus, T_minus, lam)
    F_plus, F_minus = total_faces_from_incidence(T_plus, T_minus)
    lhs = sum(K)
    rhs = chi_lambda(n, edges, F_plus, F_minus, lam)
    return abs(lhs - rhs) <= tol
