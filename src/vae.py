"""Module 4 - Variational Autoencoder (Unit V: generative models).

Encoder q(z|x) = N(mu(x), diag(sigma^2(x))),  decoder p(x|z) = N(dec(z), s^2 I)  (s = `sigma`).
Loss (negative ELBO) = ||x - dec(z)||^2 / (2 s^2) + beta * KL(q(z|x) || N(0, I)).

Uses:
  * anomaly score  -> per-flow reconstruction error of a VAE trained on benign traffic only
  * explanation    -> per-feature reconstruction error (which features look unusual?)
  * generation     -> decode z ~ N(0, I) into synthetic flow records
  * visualisation  -> latent means mu(x), projected to 2-D
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 64, d_z: int = 6, sigma: float = 0.05, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.d_in, self.d_z, self.sigma, self.seed = d_in, d_z, sigma, seed
        self.enc = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid), nn.ReLU())
        self.mu = nn.Linear(d_hid, d_z)
        self.lv = nn.Linear(d_hid, d_z)
        self.dec = nn.Sequential(nn.Linear(d_z, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_in), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)      # reparameterisation trick
        return self.dec(z), mu, lv

    def loss(self, x, beta: float = 1.0):
        xr, mu, lv = self(x)
        rec = ((xr - x) ** 2).sum(1) / (2 * self.sigma ** 2)
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return (rec + beta * kl).mean(), rec.mean(), kl.mean()

    def fit(self, X, epochs: int = 60, lr: float = 2e-3, batch: int = 128, beta: float = 1.0) -> dict:
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        hist = {"epoch": [], "loss": [], "recon": [], "kl": []}
        for ep in range(epochs):
            self.train()
            perm = torch.randperm(len(X))
            tot = np.zeros(3)
            for i in range(0, len(X), batch):
                xb = X[perm[i:i + batch]]
                loss, rec, kl = self.loss(xb, beta)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += np.array([loss.item(), rec.item(), kl.item()]) * len(xb)
            tot /= len(X)
            hist["epoch"].append(ep + 1)
            for key, val in zip(("loss", "recon", "kl"), tot):
                hist[key].append(float(val))
        self.eval()
        return hist

    @torch.no_grad()
    def reconstruct(self, X) -> np.ndarray:
        mu, _ = self.encode(torch.as_tensor(X, dtype=torch.float32))
        return self.dec(mu).numpy()

    @torch.no_grad()
    def score(self, X) -> np.ndarray:
        """Mean squared reconstruction error per flow (higher = more anomalous)."""
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.reconstruct(X)) ** 2).mean(1)

    @torch.no_grad()
    def feature_errors(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.reconstruct(X)) ** 2

    @torch.no_grad()
    def latent(self, X) -> np.ndarray:
        mu, _ = self.encode(torch.as_tensor(X, dtype=torch.float32))
        return mu.numpy()

    @torch.no_grad()
    def generate(self, n: int = 1000, seed: int = 0, noise: bool = True) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self.d_z, generator=g)
        x = self.dec(z)
        if noise:                                               # sample from p(x|z), not just its mean
            x = x + self.sigma * torch.randn(x.shape, generator=g)
        return x.clamp(0, 1).numpy()


if __name__ == "__main__":
    import time
    from sklearn.metrics import roc_auc_score
    from .data import generate_synthetic, split_data, Preprocessor
    tr, te = split_data(generate_synthetic(12000))
    pre = Preprocessor().fit(tr)
    Xb = pre.transform(tr[tr.label == "Benign"])
    Xte = pre.transform(te)
    yte = (te.label != "Benign").values
    t = time.time()
    vae = VAE(Xb.shape[1]); h = vae.fit(Xb, epochs=60)
    print(f"trained in {time.time()-t:.1f}s  loss {h['loss'][0]:.1f} -> {h['loss'][-1]:.1f}  KL {h['kl'][-1]:.2f}")
    print("VAE recon-error AUC:", round(roc_auc_score(yte, vae.score(Xte)), 4))
    S = vae.generate(1000)
    print("synthetic shape", S.shape, "mean abs diff of feature means:",
          round(float(np.abs(S.mean(0) - Xb.mean(0)).mean()), 4))
