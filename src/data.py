"""Module 1 - Network data ingestion.

* generate_synthetic(): a labelled synthetic flow generator (works with no download)
* load_flows():          loader for CIC-IDS2017 CSVs (maps columns to our compact feature set)
* Preprocessor:          log transform + min-max scaling, fitted on TRAIN data only
* split_data():          stratified train/test split
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "pkts_per_s", "bytes_per_s", "mean_pkt_len", "syn_cnt", "iat_mean", "dst_port",
]
# all features get log1p before scaling
LOG_FEATURES = list(FEATURES)  # every feature is heavy-tailed or categorical-ish; log scaling separates ports like 22/53/80/443

ATTACK_CLASSES = ["Benign", "DoS", "PortScan", "BruteForce"]
# Coarse, *hypothesised* stage label per attack family (used only for the dashboard timeline)
STAGE_OF = {
    "Benign": "-",
    "PortScan": "Reconnaissance",
    "BruteForce": "Credential access",
    "DoS": "Impact",
}


# --------------------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------------------
def _ln(rng, mean, sigma, m):
    return rng.lognormal(np.log(mean), sigma, m)


def _benign(m, rng):
    fwd = 1 + rng.poisson(_ln(rng, 8, 0.8, m)).astype(float)
    bwd = 1 + rng.poisson(_ln(rng, 10, 0.9, m)).astype(float)
    fb = fwd * _ln(rng, 350, 0.5, m)
    bb = bwd * _ln(rng, 700, 0.6, m)
    dur = _ln(rng, 1.5, 1.1, m)
    syn = (rng.random(m) < 0.9).astype(float)
    port = rng.choice([80, 443, 53, 8080, 22, 3306], size=m, p=[.30, .45, .12, .06, .05, .02]).astype(float)
    big = rng.random(m) < 0.03  # occasional bulk transfers: benign but unusual
    fwd[big] *= 20; bwd[big] *= 20; fb[big] *= 40; bb[big] *= 40; dur[big] *= 10
    return fwd, bwd, fb, bb, dur, syn, port


def _dos(m, rng):
    fwd = 1 + rng.poisson(_ln(rng, 250, 0.5, m)).astype(float)
    bwd = rng.poisson(0.7, m).astype(float)
    fb = fwd * _ln(rng, 70, 0.2, m)
    bb = bwd * _ln(rng, 60, 0.3, m)
    dur = _ln(rng, 0.4, 0.7, m)
    syn = np.maximum(1, np.round(fwd * rng.uniform(0.5, 0.95, m)))
    port = rng.choice([80, 443], size=m, p=[.7, .3]).astype(float)
    return fwd, bwd, fb, bb, dur, syn, port


def _portscan(m, rng):
    fwd = 1 + rng.poisson(0.5, m).astype(float)
    bwd = rng.poisson(0.3, m).astype(float)
    fb = fwd * _ln(rng, 58, 0.1, m)
    bb = bwd * 60.0
    dur = _ln(rng, 0.002, 0.8, m)
    syn = np.ones(m)
    port = rng.integers(1, 65536, m).astype(float)
    return fwd, bwd, fb, bb, dur, syn, port


def _brute(m, rng):
    fwd = 10 + rng.poisson(_ln(rng, 12, 0.3, m)).astype(float)
    bwd = 8 + rng.poisson(_ln(rng, 10, 0.3, m)).astype(float)
    fb = fwd * _ln(rng, 120, 0.3, m)
    bb = bwd * _ln(rng, 140, 0.3, m)
    dur = _ln(rng, 2.5, 0.4, m)
    syn = np.ones(m)
    port = rng.choice([22, 21], size=m, p=[.7, .3]).astype(float)
    return fwd, bwd, fb, bb, dur, syn, port


_MAKERS = {"Benign": _benign, "DoS": _dos, "PortScan": _portscan, "BruteForce": _brute}


def generate_synthetic(n: int = 12000, seed: int = 7, p_attack=(0.07, 0.07, 0.06),
                       stealth_frac: float = 0.06) -> pd.DataFrame:
    """Synthetic labelled flows. p_attack is the share of (DoS, PortScan, BruteForce).
    `stealth_frac` of attack flows look statistically benign (so no model reaches 100%)."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(ATTACK_CLASSES, size=n, p=[1 - sum(p_attack), *p_attack])

    cols = {k: np.zeros(n) for k in ["fwd", "bwd", "fb", "bb", "dur", "syn", "port"]}
    for cls in ATTACK_CLASSES:
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        vals = _MAKERS[cls](len(idx), rng)
        for k, v in zip(cols, vals):
            cols[k][idx] = v
    # stealthy attacks: benign-looking features, attack label retained
    atk = np.where(labels != "Benign")[0]
    stealth = atk[rng.random(len(atk)) < stealth_frac]
    if len(stealth):
        vals = _benign(len(stealth), rng)
        for k, v in zip(cols, vals):
            cols[k][stealth] = v

    dur = np.clip(cols["dur"], 1e-4, None)
    tot = cols["fwd"] + cols["bwd"]
    df = pd.DataFrame({
        "duration": dur,
        "fwd_pkts": cols["fwd"], "bwd_pkts": cols["bwd"],
        "fwd_bytes": cols["fb"], "bwd_bytes": cols["bb"],
        "pkts_per_s": tot / dur,
        "bytes_per_s": (cols["fb"] + cols["bb"]) / dur,
        "mean_pkt_len": (cols["fb"] + cols["bb"]) / np.maximum(tot, 1),
        "syn_cnt": cols["syn"],
        "iat_mean": dur / np.maximum(tot, 1),
        "dst_port": cols["port"],
    })

    clients = np.array([f"192.168.1.{i}" for i in range(10, 50)])
    servers = np.array([f"10.0.0.{i}" for i in (5, 6, 7, 8, 9)])
    src = np.empty(n, dtype=object); dst = np.empty(n, dtype=object); ts = np.zeros(n)
    for cls in ATTACK_CLASSES:
        idx = np.where(labels == cls)[0]
        m = len(idx)
        if m == 0:
            continue
        if cls == "Benign":
            src[idx] = rng.choice(clients, m)
            dst[idx] = rng.choice(servers, m, p=[.35, .25, .15, .15, .10])
            ts[idx] = rng.uniform(0, 3600, m)
        elif cls == "DoS":
            src[idx] = rng.choice(["203.0.113.66", "203.0.113.67"], m)
            dst[idx] = "10.0.0.5"
            ts[idx] = rng.uniform(2600, 3000, m)
        elif cls == "PortScan":
            src[idx] = "198.51.100.23"
            dst[idx] = rng.choice(["10.0.0.6", "10.0.0.8"], m)
            ts[idx] = rng.uniform(600, 900, m)
        else:
            src[idx] = "203.0.113.99"
            dst[idx] = "10.0.0.7"
            ts[idx] = rng.uniform(1500, 2100, m)
    df.insert(0, "dst_ip", dst); df.insert(0, "src_ip", src); df.insert(0, "timestamp", ts)
    df["label"] = labels
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "flow_id", np.arange(len(df)))
    return df


# --------------------------------------------------------------------------------------
# Real data loader (CIC-IDS2017)
# --------------------------------------------------------------------------------------
_CIC_MAP = {
    "flow duration": "duration", "total fwd packets": "fwd_pkts", "total backward packets": "bwd_pkts",
    "total length of fwd packets": "fwd_bytes", "total length of bwd packets": "bwd_bytes",
    "flow packets/s": "pkts_per_s", "flow bytes/s": "bytes_per_s", "packet length mean": "mean_pkt_len",
    "syn flag count": "syn_cnt", "flow iat mean": "iat_mean", "destination port": "dst_port",
    "label": "label", "source ip": "src_ip", "destination ip": "dst_ip", "timestamp": "timestamp",
}


def _map_label(s: str) -> str | None:
    s = str(s).strip().lower()
    if s == "benign":
        return "Benign"
    if "dos" in s or "ddos" in s:
        return "DoS"
    if "portscan" in s:
        return "PortScan"
    if "patator" in s:
        return "BruteForce"
    return None  # other attack families are dropped (our Bayesian net has 4 states)


def load_flows(path_or_df, max_per_class: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Load one or more CIC-IDS2017 CSVs (path, list of paths, or DataFrame) into our schema."""
    if isinstance(path_or_df, pd.DataFrame):
        raw = path_or_df.copy()
    else:
        paths = path_or_df if isinstance(path_or_df, (list, tuple)) else [path_or_df]
        raw = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw = raw.rename(columns={c: _CIC_MAP[c] for c in raw.columns if c in _CIC_MAP})
    missing = [f for f in FEATURES if f not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns (after CIC mapping): {missing}")
    if "label" in raw.columns:
        raw["label"] = raw["label"].map(_map_label)
        raw = raw.dropna(subset=["label"])
    else:
        raw["label"] = "Unlabeled"
    for f in FEATURES:
        raw[f] = pd.to_numeric(raw[f], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    raw = raw[(raw["duration"] >= 0)]
    raw["duration"] = raw["duration"] / 1e6   # CIC stores microseconds
    raw["iat_mean"] = raw["iat_mean"] / 1e6
    if "src_ip" not in raw.columns:
        raw["src_ip"] = "n/a"
    if "dst_ip" not in raw.columns:
        raw["dst_ip"] = "n/a"
    if "timestamp" in raw.columns:
        t = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw["timestamp"] = (t - t.min()).dt.total_seconds()
        raw["timestamp"] = raw["timestamp"].fillna(0)
    else:
        raw["timestamp"] = np.arange(len(raw), dtype=float)  # row order as pseudo-time
    caps = max_per_class or {"Benign": 8000, "DoS": 2000, "PortScan": 2000, "BruteForce": 2000}
    if "Unlabeled" not in set(raw["label"]):
        parts = []
        for lab, g in raw.groupby("label"):
            parts.append(g.sample(n=min(len(g), caps.get(lab, 2000)), random_state=seed))
        raw = pd.concat(parts)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw.insert(0, "flow_id", np.arange(len(raw)))
    return raw[["flow_id", "timestamp", "src_ip", "dst_ip", *FEATURES, "label"]]


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
class Preprocessor:
    """log1p on heavy-tailed features, then min-max to [0, 1] using robust (0.5/99.5 pct) bounds
    estimated on the training split only (no test leakage)."""

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        Z = self._log(df)
        self.lo = np.percentile(Z, 0.5, axis=0)
        self.hi = np.percentile(Z, 99.5, axis=0)
        return self

    @staticmethod
    def _log(df: pd.DataFrame) -> np.ndarray:
        Z = df[FEATURES].astype(float).copy()
        Z[LOG_FEATURES] = np.log1p(Z[LOG_FEATURES].clip(lower=0))
        return Z.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        Z = self._log(df)
        return np.clip((Z - self.lo) / (self.hi - self.lo + 1e-9), 0, 1).astype(np.float32)

    def inverse(self, X: np.ndarray) -> pd.DataFrame:
        Z = np.asarray(X) * (self.hi - self.lo + 1e-9) + self.lo
        out = pd.DataFrame(Z, columns=FEATURES)
        out[LOG_FEATURES] = np.expm1(out[LOG_FEATURES]).clip(lower=0)
        for c in ("fwd_pkts", "bwd_pkts", "syn_cnt"):
            out[c] = out[c].round()
        out["dst_port"] = out["dst_port"].round().clip(1, 65535)
        return out


def split_data(df: pd.DataFrame, test_size: float = 0.3, seed: int = 0):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


if __name__ == "__main__":
    d = generate_synthetic(6000)
    print(d.head())
    print(d.label.value_counts())
    tr, te = split_data(d)
    pre = Preprocessor().fit(tr)
    print(pre.transform(te).shape, pre.transform(te).min(), pre.transform(te).max())
