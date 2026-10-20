"""Module 2 - Bayesian investigation engine (Unit I: directed graphical models).

A small discrete Bayesian network. The *structure* encodes explicit causal assumptions
(edit DEFAULT_PARENTS to change them); the CPTs are learned from data by counting with
Laplace smoothing. Inference is exact (the joint tensor is tiny).

    Attack ──► PktRate ──► Duration        (attack type drives rate; rate drives duration)
    Attack ──► Syn, Port, PktSize
    Attack ──► Asym ◄── PktSize            (byte asymmetry depends on attack and packet size)
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .discretize import STATES, VARS, INDICATORS
from .inference import conditional, posterior_batch, cond_mutual_info, mean_loglik

DEFAULT_PARENTS = {
    "Attack": [],
    "PktRate": ["Attack"],
    "Duration": ["Attack", "PktRate"],
    "Syn": ["Attack"],
    "Port": ["Attack"],
    "PktSize": ["Attack"],
    "Asym": ["Attack", "PktSize"],
}


class BayesNet:
    def __init__(self, parents: dict | None = None, states: dict | None = None):
        self.parents = parents or DEFAULT_PARENTS
        self.states = states or STATES
        self.vars = list(self.states)
        self.card = {v: len(s) for v, s in self.states.items()}
        self.cpt: dict[str, np.ndarray] = {}

    # ---- learning -----------------------------------------------------------------
    def fit(self, D: pd.DataFrame, alpha: float = 1.0) -> "BayesNet":
        """Maximum-a-posteriori CPTs (Dirichlet(alpha) prior = Laplace smoothing)."""
        for v in self.vars:
            ps = self.parents.get(v, [])
            counts = np.full([self.card[p] for p in ps] + [self.card[v]], float(alpha))
            np.add.at(counts, tuple(D[p].values for p in ps) + (D[v].values,), 1)
            self.cpt[v] = counts / counts.sum(-1, keepdims=True)
        self._joint = None
        return self

    # ---- joint distribution ---------------------------------------------------------
    @property
    def joint(self) -> np.ndarray:
        """P(X1..Xn) = prod_i P(Xi | parents(Xi)), as an n-dimensional tensor."""
        if getattr(self, "_joint", None) is None:
            letter = {v: "abcdefghij"[i] for i, v in enumerate(self.vars)}
            subs = [("".join(letter[p] for p in self.parents.get(v, [])) + letter[v]) for v in self.vars]
            expr = ",".join(subs) + "->" + "".join(letter[v] for v in self.vars)
            self._joint = np.einsum(expr, *[self.cpt[v] for v in self.vars])
        return self._joint

    # ---- inference ------------------------------------------------------------------
    def posterior(self, query: str = "Attack", evidence: dict | None = None) -> np.ndarray:
        return conditional(self.joint, self.vars, query, evidence or {})

    def posterior_batch(self, D: pd.DataFrame) -> np.ndarray:
        """P(Attack | all indicators) for each row. D has the INDICATORS columns."""
        return posterior_batch(self.joint, D[INDICATORS].values)

    def sequential_update(self, evidence: dict, order: list | None = None) -> pd.DataFrame:
        """Belief about Attack as evidence arrives one indicator at a time (evidential reasoning)."""
        order = [v for v in (order or INDICATORS) if v in evidence]
        rows = [{"step": "prior", **dict(zip(self.states["Attack"], self.posterior("Attack", {})))}]
        seen: dict = {}
        for v in order:
            seen[v] = evidence[v]
            post = self.posterior("Attack", seen)
            rows.append({"step": f"+ {v}={self.states[v][evidence[v]]}", **dict(zip(self.states["Attack"], post))})
        return pd.DataFrame(rows)

    # ---- structure / independence -----------------------------------------------------
    def graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(self.vars)
        for v, ps in self.parents.items():
            for p in ps:
                G.add_edge(p, v)
        return G

    def d_separated(self, x: str, y: str, given=()) -> bool:
        """Graphical test: is x independent of y given `given`, according to the DAG alone?"""
        return bool(nx.is_d_separator(self.graph(), {x}, {y}, set(given)))

    def cmi(self, x: str, y: str, given=()) -> float:
        """Numerical I(x;y|given) under the learned joint, to compare with the graphical claim."""
        return cond_mutual_info(self.joint, self.vars, x, y, tuple(given))

    def loglik(self, D: pd.DataFrame) -> float:
        return mean_loglik(self.joint, D[self.vars].values)

    def cpt_frame(self, v: str) -> pd.DataFrame:
        ps = self.parents.get(v, [])
        if not ps:
            return pd.DataFrame([self.cpt[v]], columns=self.states[v], index=["(prior)"])
        import itertools
        rows, idx = [], []
        for combo in itertools.product(*[range(self.card[p]) for p in ps]):
            idx.append(", ".join(f"{p}={self.states[p][i]}" for p, i in zip(ps, combo)))
            rows.append(self.cpt[v][combo])
        return pd.DataFrame(rows, columns=self.states[v], index=idx)


if __name__ == "__main__":
    from .data import generate_synthetic, split_data
    from .discretize import Discretizer
    tr, te = split_data(generate_synthetic(12000))
    disc = Discretizer().fit(tr)
    bn = BayesNet().fit(disc.transform(tr, True))
    post = bn.posterior_batch(disc.transform(te))
    pred = post.argmax(1)
    truth = disc.transform(te, True)["Attack"].values
    print("BN class accuracy:", (pred == truth).mean())
    print("prior:", bn.posterior("Attack", {}).round(3))
    print("evidence PktRate=high,Syn=many ->", bn.posterior("Attack", {"PktRate": 2, "Syn": 1}).round(3))
    print("Syn _||_ Duration | Attack ?  d-sep:", bn.d_separated("Syn", "Duration", ["Attack"]),
          " CMI:", round(bn.cmi("Syn", "Duration", ["Attack"]), 4))
    print("Syn _||_ Duration (marginal) ? d-sep:", bn.d_separated("Syn", "Duration"),
          " MI:", round(bn.cmi("Syn", "Duration"), 4))
