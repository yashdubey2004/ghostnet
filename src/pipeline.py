"""Glue: trains every GhostNet model on one dataset and scores new flows.

    bundle = train_all(df)           # trains baseline, BN, MN, RBM, VAE (benign), VAE (all traffic)
    scores = score_frame(bundle, df) # every model's score for every flow
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import fit_baselines, rf_attack_prob, lr_attack_prob
from .bayes import BayesNet
from .data import ATTACK_CLASSES, Preprocessor, split_data
from .discretize import Discretizer, INDICATORS
from .evaluate import compare_models, fidelity, thresholds_from_benign
from .markov import MarkovNet
from .rbm import RBM, ThermometerEncoder
from .vae import VAE


@dataclass
class Bundle:
    source: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    pre: Preprocessor
    disc: Discretizer
    enc: ThermometerEncoder
    baselines: dict
    bn: BayesNet
    mn: MarkovNet
    rbm: RBM
    vae: VAE
    vae_all: VAE
    hist: dict = field(default_factory=dict)
    ref: dict = field(default_factory=dict)          # sorted benign-train scores for percentiles
    train_scores: pd.DataFrame | None = None
    test_scores: pd.DataFrame | None = None
    thresholds: dict = field(default_factory=dict)
    metrics: pd.DataFrame | None = None
    per_class: pd.DataFrame | None = None
    ll: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)


def _pct(ref_sorted: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100.0 * np.searchsorted(ref_sorted, s) / len(ref_sorted)


def snap(b: Bundle, X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn generated [0,1] vectors into valid flow records: back to raw units, round packet/SYN
    counts and ports to integers, and return both the records and their re-scaled version."""
    raw = b.pre.inverse(X)
    return raw, b.pre.transform(raw)


def score_frame(b: Bundle, df: pd.DataFrame) -> pd.DataFrame:
    """All model outputs for each flow in df (index preserved)."""
    X = b.pre.transform(df)
    out = pd.DataFrame(index=df.index)
    out["rbm_fe"] = b.rbm.score(b.enc.encode(X))
    out["vae_err"] = b.vae.score(X)
    out["rbm_pct"] = _pct(b.ref["rbm_fe"], out["rbm_fe"].values)
    out["vae_pct"] = _pct(b.ref["vae_err"], out["vae_err"].values)
    out["anomaly_pct"] = (out["rbm_pct"] + out["vae_pct"]) / 2
    out["rf_attack"] = rf_attack_prob(b.baselines["rf"], X)
    out["lr_attack"] = lr_attack_prob(b.baselines["lr"], X)
    D = b.disc.transform(df)
    bn_post = b.bn.posterior_batch(D)
    mn_post = b.mn.posterior_batch(D)
    out["bn_attack"] = 1 - bn_post[:, 0]
    out["mn_attack"] = 1 - mn_post[:, 0]
    for i, c in enumerate(ATTACK_CLASSES):
        out[f"bn_p_{c}"] = bn_post[:, i]
    out["bn_pred"] = [ATTACK_CLASSES[i] for i in bn_post.argmax(1)]
    return out


def train_all(df: pd.DataFrame, source: str = "synthetic", n_hid: int = 64, rbm_epochs: int = 40,
              vae_epochs: int = 100, seed: int = 0, log=print) -> Bundle:
    log("Splitting data (stratified 70/30)...")
    train_df, test_df = split_data(df, seed=seed)
    pre = Preprocessor().fit(train_df)
    Xtr, Xte = pre.transform(train_df), pre.transform(test_df)
    benign = (train_df["label"] == "Benign").values
    yte = (test_df["label"] != "Benign").values

    log("Phase 1: baseline classifiers...")
    baselines = fit_baselines(Xtr, train_df["label"].values, seed)

    log("Phase 2: Bayesian network + Markov network...")
    disc = Discretizer().fit(train_df)
    Dtr, Dte = disc.transform(train_df, True), disc.transform(test_df, True)
    bn = BayesNet().fit(Dtr)
    mn = MarkovNet().learn_structure(Dtr).fit(Dtr)

    log("Phase 3: RBM (contrastive divergence)...")
    enc = ThermometerEncoder(10)
    Btr, Bte = enc.encode(Xtr), enc.encode(Xte)
    rbm = RBM(Btr.shape[1], n_hid, seed)
    rbm_hist = rbm.fit(Btr[benign], epochs=rbm_epochs,
                       eval_sets={"benign_test": Bte[~yte], "attack_test": Bte[yte]})

    log("Phase 4: VAEs (benign-only for detection, all traffic for generation)...")
    vae = VAE(Xtr.shape[1], d_z=4, sigma=0.1, seed=seed)   # low capacity on purpose: anomalies reconstruct badly
    vae_hist = vae.fit(Xtr[benign], epochs=vae_epochs)
    vae_all = VAE(Xtr.shape[1], seed=seed + 1)
    vae_all_hist = vae_all.fit(Xtr, epochs=vae_epochs)

    b = Bundle(source, train_df, test_df, pre, disc, enc, baselines, bn, mn, rbm, vae, vae_all,
               hist={"rbm": rbm_hist, "vae": vae_hist, "vae_all": vae_all_hist, "mn": mn.history})

    log("Scoring + evaluation...")
    tr_ben = train_df[benign]
    ref_rbm = rbm.score(enc.encode(pre.transform(tr_ben)))
    ref_vae = vae.score(pre.transform(tr_ben))
    b.ref = {"rbm_fe": np.sort(ref_rbm), "vae_err": np.sort(ref_vae)}
    b.train_scores = score_frame(b, train_df)
    b.test_scores = score_frame(b, test_df)
    b.thresholds = thresholds_from_benign(b.train_scores[benign])
    b.metrics, b.per_class = compare_models(b.test_scores, test_df["label"], b.thresholds)
    b.ll = {"bn": bn.loglik(Dte), "mn": mn.loglik(Dte)}

    log("Fidelity of generated data...")
    Xte_ben = Xte[~yte]
    b.fidelity = {
        "VAE detector (benign-only) vs benign test": fidelity(Xte_ben, snap(b, vae.generate(len(Xte_ben), seed=1))[1]),
        "VAE (all traffic) vs test": fidelity(Xte, snap(b, vae_all.generate(len(Xte), seed=2))[1]),
        "RBM Gibbs samples vs benign test":
            fidelity(Xte_ben, snap(b, enc.decode(rbm.generate(len(Xte_ben), burn=300, seed=3)))[1]),
    }
    log("Done.")
    return b


def save_bundle(b: Bundle, path="artifacts/bundle.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(b, f)


def load_bundle(path="artifacts/bundle.pkl") -> Bundle | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
