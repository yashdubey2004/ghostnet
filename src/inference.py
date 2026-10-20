"""Exact inference helpers shared by the Bayesian network and the Markov network.
Both models are represented by a full joint-probability tensor (small: 4*3*3*2*3*3*3 = 1944 cells),
so conditioning and marginalising are just slicing and summing."""
from __future__ import annotations

import numpy as np


def conditional(joint: np.ndarray, vars_: list, query: str, evidence: dict) -> np.ndarray:
    """P(query | evidence). `evidence` maps variable -> state index."""
    evidence = {k: v for k, v in evidence.items() if k != query}
    idx = tuple(evidence[v] if v in evidence else slice(None) for v in vars_)
    sub = joint[idx]
    remaining = [v for v in vars_ if v not in evidence]
    axes = tuple(i for i, v in enumerate(remaining) if v != query)
    p = sub.sum(axis=axes) if axes else sub
    return p / p.sum()


def posterior_batch(joint: np.ndarray, D: np.ndarray) -> np.ndarray:
    """P(vars_[0] | all other variables observed) for every row of D (n, K) -> (n, card0)."""
    cols = tuple(D[:, k] for k in range(D.shape[1]))
    P = joint[(slice(None),) + cols]          # (card0, n)
    return (P / P.sum(0, keepdims=True)).T


def cond_mutual_info(joint: np.ndarray, vars_: list, x: str, y: str, z=()) -> float:
    """I(X; Y | Z) in nats, computed from the joint tensor."""
    keep = [x, y, *z]
    drop = tuple(i for i, v in enumerate(vars_) if v not in keep)
    p = joint.sum(axis=drop) if drop else joint
    order = [v for v in vars_ if v in keep]
    p = np.transpose(p, [order.index(v) for v in keep])
    pz = p.sum(axis=(0, 1), keepdims=True)
    pxz = p.sum(axis=1, keepdims=True)
    pyz = p.sum(axis=0, keepdims=True)
    with np.errstate(all="ignore"):
        t = p * np.log(p * pz / (pxz * pyz))
    return float(np.nansum(t))


def mean_loglik(joint: np.ndarray, D: np.ndarray) -> float:
    """Average log-probability (nats per row) of the rows of D (columns in vars_ order)."""
    idx = tuple(D[:, k] for k in range(D.shape[1]))
    return float(np.log(joint[idx]).mean())
