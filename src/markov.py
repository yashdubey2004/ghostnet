"""Markov network (Unit II: undirected graphical models).

A pairwise Markov random field over the SAME seven discrete variables as the Bayesian network,
but with undirected edges and *factors* instead of conditional probability tables:

    P(x) = (1/Z) * prod_{(i,j) in E} phi_ij(x_i, x_j),     phi_ij = exp(theta_ij[x_i, x_j])

  * structure : Chow-Liu maximum-spanning tree on pairwise mutual information + a few extra edges
  * parameters: exact maximum likelihood. The state space is tiny (1944 states), so Z is computed
                exactly and the log-likelihood is optimised with Adam (autograd does the gradient
                <phi>_data - <phi>_model for us).
  * inference : condition the joint tensor on evidence (same helpers as the Bayesian network)
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .discretize import STATES, VARS
from .inference import conditional, posterior_batch, cond_mutual_info, mean_loglik


def pairwise_mi(D: pd.DataFrame, vars_: list, card: dict) -> dict:
    out = {}
    n = len(D)
    for a, b in itertools.combinations(vars_, 2):
        c = np.zeros((card[a], card[b]))
        np.add.at(c, (D[a].values, D[b].values), 1)
        p = (c + 0.5) / (n + 0.5 * c.size)
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        out[(a, b)] = float((p * np.log(p / (px * py))).sum())
    return out


class MarkovNet:
    def __init__(self, states: dict | None = None):
        self.states = states or STATES
        self.vars = list(self.states)
        self.card = {v: len(s) for v, s in self.states.items()}
        self.edges: list[tuple[str, str]] = []
        self.theta: dict[tuple[str, str], np.ndarray] = {}
        self.history: list[float] = []

    # ---- structure -------------------------------------------------------------------
    def learn_structure(self, D: pd.DataFrame, n_extra: int = 3) -> "MarkovNet":
        mi = pairwise_mi(D, self.vars, self.card)
        G = nx.Graph()
        for (a, b), w in mi.items():
            G.add_edge(a, b, weight=w)
        tree = nx.maximum_spanning_tree(G)
        self.edges = [tuple(e) for e in tree.edges]
        rest = sorted((e for e in mi if e not in self.edges and (e[1], e[0]) not in self.edges),
                      key=lambda e: -mi[e])
        self.edges += rest[:n_extra]
        self.mi = mi
        return self

    # ---- parameter learning (exact ML) -------------------------------------------------------
    def fit(self, D: pd.DataFrame, steps: int = 400, lr: float = 0.05, l2: float = 1e-4) -> "MarkovNet":
        names = self.vars
        card = [self.card[v] for v in names]
        pos = {v: i for i, v in enumerate(names)}
        grid = torch.as_tensor(np.indices(card).reshape(len(card), -1).T, dtype=torch.long)   # (S, nvars)
        flat = np.ravel_multi_index(D[names].values.T, card)
        counts = torch.as_tensor(np.bincount(flat, minlength=int(np.prod(card))), dtype=torch.float64)
        N = counts.sum()
        theta = {e: torch.zeros(self.card[e[0]], self.card[e[1]], dtype=torch.float64, requires_grad=True)
                 for e in self.edges}
        opt = torch.optim.Adam(theta.values(), lr=lr)
        self.history = []
        for _ in range(steps):
            score = sum(theta[e][grid[:, pos[e[0]]], grid[:, pos[e[1]]]] for e in self.edges)
            logZ = torch.logsumexp(score, 0)
            nll = -(counts * (score - logZ)).sum() / N
            loss = nll + l2 * sum((t ** 2).sum() for t in theta.values())
            opt.zero_grad(); loss.backward(); opt.step()
            self.history.append(float(nll.detach()))
        with torch.no_grad():
            score = sum(theta[e][grid[:, pos[e[0]]], grid[:, pos[e[1]]]] for e in self.edges)
            self._joint = torch.softmax(score, 0).numpy().reshape(card)
        self.theta = {e: t.detach().numpy() for e, t in theta.items()}
        return self

    @property
    def joint(self) -> np.ndarray:
        return self._joint

    # ---- inference ---------------------------------------------------------------------------
    def posterior(self, query: str = "Attack", evidence: dict | None = None) -> np.ndarray:
        return conditional(self.joint, self.vars, query, evidence or {})

    def posterior_batch(self, D: pd.DataFrame) -> np.ndarray:
        from .discretize import INDICATORS
        return posterior_batch(self.joint, D[INDICATORS].values)

    def loglik(self, D: pd.DataFrame) -> float:
        return mean_loglik(self.joint, D[self.vars].values)

    def cmi(self, x, y, given=()):
        return cond_mutual_info(self.joint, self.vars, x, y, tuple(given))

    # ---- views -----------------------------------------------------------------------------------
    def factor(self, e: tuple[str, str]) -> pd.DataFrame:
        """Learned factor phi = exp(theta), rescaled so its mean is 1 (ratios are what matter)."""
        phi = np.exp(self.theta[e])
        phi = phi / phi.mean()
        return pd.DataFrame(phi, index=self.states[e[0]], columns=self.states[e[1]])

    def graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.vars)
        G.add_edges_from(self.edges)
        return G

    def factor_graph(self) -> nx.Graph:
        """Bipartite variable-factor graph (square factor nodes named 'f:A-B')."""
        G = nx.Graph()
        G.add_nodes_from(self.vars, kind="variable")
        for a, b in self.edges:
            f = f"f:{a}-{b}"
            G.add_node(f, kind="factor")
            G.add_edge(a, f); G.add_edge(f, b)
        return G


if __name__ == "__main__":
    from .data import generate_synthetic, split_data
    from .discretize import Discretizer
    from .bayes import BayesNet
    tr, te = split_data(generate_synthetic(12000))
    disc = Discretizer().fit(tr)
    Dtr, Dte = disc.transform(tr, True), disc.transform(te, True)
    mn = MarkovNet().learn_structure(Dtr).fit(Dtr)
    bn = BayesNet().fit(Dtr)
    print("MN edges:", mn.edges)
    print("NLL first/last:", round(mn.history[0], 3), round(mn.history[-1], 3))
    print("held-out loglik  BN:", round(bn.loglik(Dte), 4), " MN:", round(mn.loglik(Dte), 4))
    acc = (mn.posterior_batch(Dte.drop(columns="Attack")).argmax(1) == Dte["Attack"].values).mean()
    print("MN class accuracy:", round(float(acc), 4))
