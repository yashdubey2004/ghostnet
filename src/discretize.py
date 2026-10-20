"""Turns continuous flow features into the small discrete indicators used by the
Bayesian network (Module 2) and the Markov network (Unit II)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ATTACK_CLASSES

# Variable order matters: it is the axis order of every joint-probability tensor.
STATES = {
    "Attack":   ATTACK_CLASSES,                                   # latent cause (hidden at test time)
    "PktRate":  ["low", "mid", "high"],                           # packets per second
    "Duration": ["short", "mid", "long"],                         # flow duration
    "Syn":      ["<=1 SYN", "many SYN"],                          # SYN flag count
    "Port":     ["web", "remote-access", "other"],                # destination port class
    "PktSize":  ["small", "mid", "large"],                        # mean packet length
    "Asym":     ["response-heavy", "balanced", "request-heavy"],  # share of bytes sent forward
}
VARS = list(STATES.keys())
INDICATORS = VARS[1:]
CARD = {v: len(s) for v, s in STATES.items()}


def _port_class(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), 2)
    out[np.isin(p, [80, 443, 8080, 8000, 8443])] = 0
    out[np.isin(p, [21, 22, 23, 3389])] = 1
    return out


class Discretizer:
    """Quantile edges are fitted on the training split only."""
    _QUANT = {"PktRate": "pkts_per_s", "Duration": "duration", "PktSize": "mean_pkt_len"}

    def fit(self, df: pd.DataFrame) -> "Discretizer":
        self.edges = {v: np.quantile(df[c], [0.25, 0.75]) for v, c in self._QUANT.items()}
        return self

    def transform(self, df: pd.DataFrame, with_label: bool = False) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        if with_label:
            out["Attack"] = df["label"].map({c: i for i, c in enumerate(ATTACK_CLASSES)}).astype(int)
        for v, c in self._QUANT.items():
            out[v] = np.digitize(df[c].values, self.edges[v])
        out["Syn"] = (df["syn_cnt"].values > 1).astype(int)
        out["Port"] = _port_class(df["dst_port"].values)
        share = df["fwd_bytes"].values / (df["fwd_bytes"].values + df["bwd_bytes"].values + 1.0)
        out["Asym"] = np.digitize(share, [0.33, 0.66])
        cols = (["Attack"] if with_label else []) + INDICATORS
        return out[cols].astype(int)

    def describe(self) -> dict:
        return {v: [float(e) for e in edges] for v, edges in self.edges.items()}
