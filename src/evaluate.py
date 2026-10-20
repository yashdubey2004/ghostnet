"""Evaluation helpers: detection metrics, per-class detection rates, and fidelity of generated data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score

from .data import FEATURES

MODEL_COLS = {
    "Logistic regression (supervised)": ("lr_attack", "prob"),
    "Random forest (supervised)": ("rf_attack", "prob"),
    "Bayesian network": ("bn_attack", "prob"),
    "Markov network": ("mn_attack", "prob"),
    "RBM free energy (unsupervised)": ("rbm_fe", "anom"),
    "VAE recon. error (unsupervised)": ("vae_err", "anom"),
    "RBM + VAE ensemble (unsupervised)": ("anomaly_pct", "anom"),
}


def thresholds_from_benign(ref_scores: pd.DataFrame, q: float = 95.0) -> dict:
    """Unsupervised models: flag anything above the q-th percentile of BENIGN TRAIN scores.
    Probabilistic models: 0.5."""
    th = {}
    for name, (col, kind) in MODEL_COLS.items():
        th[name] = 0.5 if kind == "prob" else float(np.percentile(ref_scores[col], q))
    return th


def compare_models(scores: pd.DataFrame, labels: pd.Series, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = (labels != "Benign").astype(int).values
    rows, per_class = [], []
    for name, (col, _) in MODEL_COLS.items():
        s = scores[col].values
        pred = (s > thresholds[name]).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        fpr = float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
        rows.append({"Model": name, "ROC AUC": roc_auc_score(y, s), "PR AUC": average_precision_score(y, s),
                     "Precision": p, "Recall": r, "F1": f, "False-pos. rate": fpr})
        rec = {"Model": name}
        for cls in sorted(set(labels) - {"Benign"}):
            m = (labels == cls).values
            rec[cls] = float(pred[m].mean())
        per_class.append(rec)
    return pd.DataFrame(rows).set_index("Model"), pd.DataFrame(per_class).set_index("Model")


# ---------------------------------------------------------------------------------------
# Fidelity of synthetic data
# ---------------------------------------------------------------------------------------
def ks_per_feature(real: np.ndarray, synth: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"feature": FEATURES,
                         "KS statistic": [ks_2samp(real[:, i], synth[:, i]).statistic for i in range(real.shape[1])]})


def corr_distance(real: np.ndarray, synth: np.ndarray) -> float:
    """Frobenius distance between the two correlation matrices (0 = identical dependence structure)."""
    a = np.nan_to_num(np.corrcoef(real.T)); b = np.nan_to_num(np.corrcoef(synth.T))
    return float(np.linalg.norm(a - b))


def c2st_auc(real: np.ndarray, synth: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test: how well can a forest tell real from synthetic?
    0.5 = indistinguishable, 1.0 = trivially distinguishable."""
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth), 3000)
    a = real[rng.choice(len(real), n, replace=False)]
    b = synth[rng.choice(len(synth), n, replace=False)]
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean())


def fidelity(real: np.ndarray, synth: np.ndarray) -> dict:
    ks = ks_per_feature(real, synth)
    return {"ks_mean": float(ks["KS statistic"].mean()), "ks_max": float(ks["KS statistic"].max()),
            "corr_dist": corr_distance(real, synth), "c2st_auc": c2st_auc(real, synth)}


def make_report(b) -> str:
    """Markdown experiment report from a trained Bundle."""
    m = b.metrics.copy()
    lines = ["# GhostNet - experiment report", "",
             f"* Data source: **{b.source}**  ({len(b.train_df)} train / {len(b.test_df)} test flows)",
             f"* Train class counts: {b.train_df.label.value_counts().to_dict()}",
             f"* Unsupervised models (RBM, VAE) are trained on benign training flows only "
             f"({int((b.train_df.label == 'Benign').sum())} flows); thresholds = 95th percentile of benign-train scores.",
             "", "## Detection performance (held-out test set)", "", m.round(3).to_markdown(), "",
             "## Detection rate per attack class (fraction flagged)", "", b.per_class.round(3).to_markdown(), "",
             "## Probabilistic graphical models", "",
             f"* Held-out average log-likelihood (nats/flow): Bayesian net {b.ll['bn']:.3f}, Markov net {b.ll['mn']:.3f}",
             f"* Markov network edges: {b.mn.edges}", "",
             "## Synthetic-data fidelity (lower KS / corr-dist is better; C2ST AUC near 0.5 is best)", "",
             pd.DataFrame(b.fidelity).T.round(3).to_markdown(), "",
             "## Caveats", "",
             "* Synthetic records are samples from learned models. They are NOT real attack traffic.",
             "* Bayesian/Markov net posteriors reflect the learned data and the assumed structure, "
             "not a verified causal reconstruction of an attack.",
             "* The built-in demo data are generated, so absolute scores say little about real networks; "
             "re-run on CIC-IDS2017 for real conclusions."]
    return "\n".join(lines)
