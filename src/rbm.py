"""Module 3 - Restricted Boltzmann Machine (Unit III: RBMs, Unit IV: Gibbs sampling / Markov chains).

A binary-binary RBM trained with Contrastive Divergence (CD-k).
Continuous features (already scaled to [0,1]) are turned into bits by a thermometer encoding,
so the visible layer is genuinely binary and the model can be *sampled* with real Gibbs sampling.

  Energy:        E(v,h) = -b_v.v - b_h.h - v^T W h
  Free energy:   F(v)   = -b_v.v - sum_j softplus(b_h_j + (v^T W)_j)        (p(v) ∝ exp(-F(v)))
  Gibbs step:    h ~ p(h|v) = sigmoid(v W + b_h);   v ~ p(v|h) = sigmoid(h W^T + b_v)
  CD-k gradient: dW ≈ <v h>_data - <v h>_(k-step reconstruction)

Anomaly score = free energy: flows the RBM finds unlikely (high F) look unlike benign traffic.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn


class ThermometerEncoder:
    """x in [0,1]  ->  n_bits bits where bit j is 1 iff x >= (j+0.5)/n_bits."""

    def __init__(self, n_bits: int = 10):
        self.n_bits = n_bits
        self.t = (np.arange(n_bits) + 0.5) / n_bits

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (X[:, :, None] >= self.t[None, None, :]).reshape(len(X), -1).astype(np.float32)

    def decode(self, B: np.ndarray) -> np.ndarray:
        B = np.asarray(B).reshape(len(B), -1, self.n_bits)
        return B.sum(2) / self.n_bits


class RBM:
    def __init__(self, n_vis: int, n_hid: int = 64, seed: int = 0):
        self.n_vis, self.n_hid, self.seed = n_vis, n_hid, seed
        g = torch.Generator().manual_seed(seed)
        self.W = 0.01 * torch.randn(n_vis, n_hid, generator=g)
        self.bv = torch.zeros(n_vis)
        self.bh = torch.zeros(n_hid)

    # ---- conditionals ---------------------------------------------------------------
    def p_h(self, v):
        return torch.sigmoid(v @ self.W + self.bh)

    def p_v(self, h):
        return torch.sigmoid(h @ self.W.T + self.bv)

    @staticmethod
    def _bern(p, gen=None):
        return (torch.rand(p.shape, generator=gen) < p).float()

    def free_energy(self, v):
        return -(v @ self.bv) - Fn.softplus(v @ self.W + self.bh).sum(1)

    # ---- training: contrastive divergence -----------------------------------------------
    @torch.no_grad()
    def fit(self, X, epochs: int = 40, lr: float = 0.05, k: int = 1, batch: int = 128,
            wd: float = 1e-4, eval_sets: dict | None = None) -> dict:
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32)
        n = len(X)
        p = X.mean(0).clamp(0.01, 0.99)
        self.bv = torch.log(p / (1 - p))                     # sensible visible-bias init
        vW, vbv, vbh = torch.zeros_like(self.W), torch.zeros_like(self.bv), torch.zeros_like(self.bh)
        hist = {"epoch": [], "recon_mse": [], "fe_train": []}
        for name in (eval_sets or {}):
            hist[f"fe_{name}"] = []
        for ep in range(epochs):
            mom = 0.5 if ep < 5 else 0.9
            perm = torch.randperm(n)
            err = 0.0
            for i in range(0, n, batch):
                v0 = X[perm[i:i + batch]]
                B = len(v0)
                ph0 = self.p_h(v0)                              # positive phase
                h = self._bern(ph0)
                for _ in range(k):                              # k steps of Gibbs sampling
                    vk = self._bern(self.p_v(h))
                    phk = self.p_h(vk)
                    h = self._bern(phk)
                dW = (v0.T @ ph0 - vk.T @ phk) / B - wd * self.W
                dbv = (v0 - vk).mean(0)
                dbh = (ph0 - phk).mean(0)
                vW = mom * vW + lr * dW; vbv = mom * vbv + lr * dbv; vbh = mom * vbh + lr * dbh
                self.W += vW; self.bv += vbv; self.bh += vbh
                err += ((v0 - self.p_v(ph0)) ** 2).sum().item()  # mean-field reconstruction error
            hist["epoch"].append(ep + 1)
            hist["recon_mse"].append(err / (n * X.shape[1]))
            hist["fe_train"].append(self.free_energy(X[:2000]).mean().item())
            for name, E in (eval_sets or {}).items():
                hist[f"fe_{name}"].append(self.free_energy(torch.as_tensor(E, dtype=torch.float32)).mean().item())
        return hist

    # ---- scoring / sampling ---------------------------------------------------------------
    @torch.no_grad()
    def score(self, X) -> np.ndarray:
        """Free energy (higher = less likely under the benign model = more anomalous)."""
        return self.free_energy(torch.as_tensor(X, dtype=torch.float32)).numpy()

    @torch.no_grad()
    def gibbs_chain(self, v0, steps: int = 40, seed: int = 0):
        """Run a Gibbs chain from one visible vector. Returns (V (steps+1, n_vis), FE (steps+1,))."""
        gen = torch.Generator().manual_seed(seed)
        v = torch.as_tensor(np.asarray(v0), dtype=torch.float32).reshape(1, -1)
        V, FE = [v.clone()], [self.free_energy(v).item()]
        for _ in range(steps):
            h = self._bern(self.p_h(v), gen)
            v = self._bern(self.p_v(h), gen)
            V.append(v.clone()); FE.append(self.free_energy(v).item())
        return torch.cat(V).numpy(), np.array(FE)

    @torch.no_grad()
    def generate(self, n: int = 1000, burn: int = 500, seed: int = 0) -> np.ndarray:
        """Draw samples by running n parallel Gibbs chains from random bits for `burn` steps."""
        gen = torch.Generator().manual_seed(seed)
        v = (torch.rand(n, self.n_vis, generator=gen) < 0.5).float()
        for _ in range(burn):
            h = self._bern(self.p_h(v), gen)
            v = self._bern(self.p_v(h), gen)
        return v.numpy()


if __name__ == "__main__":
    import time
    from sklearn.metrics import roc_auc_score
    from .data import generate_synthetic, split_data, Preprocessor
    tr, te = split_data(generate_synthetic(12000))
    pre = Preprocessor().fit(tr)
    enc = ThermometerEncoder(10)
    Xtr = enc.encode(pre.transform(tr[tr.label == "Benign"]))
    Xte = enc.encode(pre.transform(te))
    yte = (te.label != "Benign").values
    t = time.time()
    rbm = RBM(Xtr.shape[1], 64)
    h = rbm.fit(Xtr, epochs=40, eval_sets={"benign_test": Xte[~yte], "attack_test": Xte[yte]})
    print(f"trained in {time.time()-t:.1f}s")
    print("recon_mse first/last:", round(h["recon_mse"][0], 4), round(h["recon_mse"][-1], 4))
    print("FE benign/attack (last):", round(h["fe_benign_test"][-1], 2), round(h["fe_attack_test"][-1], 2))
    print("RBM free-energy AUC:", round(roc_auc_score(yte, rbm.score(Xte)), 4))
    V, FE = rbm.gibbs_chain(Xte[yte][0], steps=40)
    print("Gibbs chain FE start->end:", round(FE[0], 2), "->", round(FE[-1], 2))
