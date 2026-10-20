"""Phase 1 baseline: simple supervised intrusion-detection classifiers.
Everything GhostNet adds later should be judged against these."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def fit_baselines(X: np.ndarray, labels, seed: int = 0) -> dict:
    labels = np.asarray(labels)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                n_jobs=-1, random_state=seed).fit(X, labels)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, (labels != "Benign").astype(int))
    return {"rf": rf, "lr": lr}


def rf_attack_prob(rf, X: np.ndarray) -> np.ndarray:
    """P(attack) = 1 - P(Benign) from the multi-class forest."""
    return 1.0 - rf.predict_proba(X)[:, list(rf.classes_).index("Benign")]


def lr_attack_prob(lr, X: np.ndarray) -> np.ndarray:
    return lr.predict_proba(X)[:, 1]
