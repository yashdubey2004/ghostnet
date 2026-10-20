# GhostNet — probabilistic cyber-threat investigation and generative simulation

DL2 solo project. GhostNet does not just classify traffic as malicious or benign: it estimates **why** an
event looks suspicious (Bayesian / Markov networks), scores **how unusual** it is (RBM, VAE), and
**simulates** traffic with generative models (VAE, RBM Gibbs sampling), all inside one Streamlit dashboard.

## 0. Quick start (5 minutes)

```bash
cd ghostnet
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # trains everything on built-in synthetic traffic (~25 s), writes artifacts/
streamlit run app.py            # opens the dashboard in your browser
```

Everything works with **no dataset download**. When the demo works, switch to real data (section 3).

## 1. Project layout — what goes where

```
ghostnet/
├── requirements.txt
├── run_pipeline.py        # CLI: train all models, save artifacts/bundle.pkl + artifacts/report.md
├── app.py                 # Module 5: Streamlit dashboard
├── data/                  # put CIC-IDS2017 CSVs here
├── artifacts/             # created by run_pipeline.py / the app (trained models, report)
└── src/
    ├── data.py            # Module 1: synthetic generator, CIC-IDS2017 loader, preprocessing, split
    ├── baseline.py        # Phase 1: logistic regression + random forest
    ├── discretize.py      # continuous features -> discrete indicators for the graphical models
    ├── inference.py       # exact inference helpers (condition, marginalise, mutual information)
    ├── bayes.py           # Module 2 / Unit I: Bayesian network
    ├── markov.py          # Unit II: Markov network (factor graph, exact ML)
    ├── rbm.py             # Module 3 / Units III+IV: RBM, contrastive divergence, Gibbs sampling
    ├── vae.py             # Module 4 / Unit V: VAE
    ├── evaluate.py        # metrics, per-class detection, fidelity of generated data, report text
    ├── pipeline.py        # trains and scores everything (Bundle)
    └── viz.py             # Plotly figures for the dashboard
```

Create each file at exactly that path and paste the code in; `src/__init__.py` is an empty file.

## 2. Build it phase by phase (run each module on its own)

Each module has a self-test at the bottom. Run from the `ghostnet/` folder.

| Phase | Files | Command | What you should see |
|---|---|---|---|
| 1 Data + baseline | `data.py`, `baseline.py` | `python -m src.data` | class counts, scaled data in [0,1]; baselines reach about 0.97 ROC AUC |
| 2 Bayesian investigation | `discretize.py`, `inference.py`, `bayes.py` | `python -m src.bayes` | ~98% 4-class accuracy; posterior jumps to DoS given high packet rate + many SYNs; d-separation vs mutual-information check |
| 3 RBM + Gibbs | `rbm.py` | `python -m src.rbm` | reconstruction error falls; benign free energy < attack free energy; free energy falls along a Gibbs chain |
| 4 VAE + synthetic data | `vae.py`, `evaluate.py` | `python -m src.vae` | loss falls; anomaly AUC about 0.96; synthetic feature means close to real |
| Unit II Markov net | `markov.py` | `python -m src.markov` | learned edges (Chow-Liu + extras); BN vs MN held-out log-likelihood |
| 5 Dashboard | `pipeline.py`, `viz.py`, `app.py` | `python run_pipeline.py` then `streamlit run app.py` | seven tabs, all interactive |

## 3. Using real data (CIC-IDS2017)

1. Download the CIC-IDS2017 CSVs (use the **TrafficLabelling** version if you want IPs and timestamps for the
   host graph and timeline; the *MachineLearningCVE* version has no IPs).
2. Put one or two day-files in `data/` (e.g. a Wednesday DoS file and a Friday PortScan file).
3. Train: `python run_pipeline.py --csv data/<file1>.csv data/<file2>.csv`
   or upload a file in the dashboard sidebar and press **Train GhostNet**.

The loader maps CIC column names to GhostNet's 11 features, converts microseconds to seconds, drops inf/NaN rows,
keeps Benign / DoS / DDoS / PortScan / FTP- and SSH-Patator, and sub-samples per class. Other attack families are
dropped because the Bayesian network has four states. UNSW-NB15 would need a small column map added to `_CIC_MAP` in `data.py`.

## 4. How each part maps to your syllabus

| Unit | Where | What it does |
|---|---|---|
| I Directed models | `bayes.py` | CPTs by counting, joint = product of CPTs, evidence updates, d-separation vs numerical conditional MI |
| II Markov networks | `markov.py` | Pairwise factors over the same variables, Chow-Liu structure, exact maximum-likelihood fit, factor-graph view |
| III RBMs | `rbm.py` | Energy / free energy, hidden-unit conditionals, free energy as anomaly score |
| IV Markov chains | `rbm.py` | Gibbs sampling (`gibbs_chain`, `generate`), CD-k training, live chain in the dashboard |
| V Generative models | `vae.py` | ELBO, reparameterisation, latent space, sampling synthetic flows, fidelity tests |

## 5. Results on the built-in demo data (seed 7, 12,000 flows)

| Model | ROC AUC | F1 |
|---|---|---|
| Logistic regression | 0.968 | 0.946 |
| Random forest | 0.970 | 0.965 |
| Bayesian network | 0.970 | 0.952 |
| Markov network | 0.972 | 0.952 |
| RBM free energy (unsupervised) | 0.958 | 0.809 |
| VAE reconstruction error (unsupervised) | 0.966 | 0.872 |
| RBM + VAE ensemble | 0.966 | 0.847 |

Honest reading of these numbers, worth writing in your report:

* **The ceiling is about 93–94% recall by construction.** 6% of synthetic attack flows were generated to look
  benign ("stealth"), so no model can catch them. That is why every supervised model has nearly the same recall.
* **Supervised models are better on pure detection.** The point of GhostNet is explanation and simulation, not beating a random forest.
  The unsupervised models never saw an attack label, yet reach AUC 0.96, at the cost of more false positives (5–7%).
* **Bayesian and Markov nets match the baselines** while giving an interpretable posterior over *which* attack family explains the event.
* **VAE capacity matters.** A higher-capacity VAE reconstructed attacks too well and detection AUC fell from 0.97 to 0.92,
  so the detector is deliberately small (4 latent dims), while the generator uses 6.
* **Generation is only partly successful.** The VAE reproduces marginals reasonably (mean KS about 0.09), but a classifier
  can still tell its output from real flows (AUC near 1), partly because real flows obey exact relations
  (e.g. packets/s = packets / duration) that the VAE only approximates. The CD-1 RBM is a good anomaly detector but a poor
  sampler (mean KS about 0.48). These are genuine findings, report them rather than hiding them.
* These are **synthetic** data. Re-run on CIC-IDS2017 before drawing any real-world conclusion.

## 6. Things to avoid (from your project plan)

* Do not claim generated records are real attack traffic. The dashboard says so, keep that wording in your report.
* Do not claim the timeline reconstructs the true attack sequence. It shows when flows of a family were observed;
  the stage names are a coarse hypothesised mapping.
* Do not add a GAN or autoregressive model until you have evaluated the core. Suggested order if time remains:
  (1) persistent contrastive divergence or CD-10 for the RBM and compare sample fidelity, (2) a MADE-style autoregressive
  density estimator on the 11 scaled features, (3) a small GAN with the same fidelity tests in `evaluate.py`.

## 7. Customising

* Change causal assumptions: edit `DEFAULT_PARENTS` in `bayes.py`, retrain, and compare held-out log-likelihood.
* Change discretisation bins or states: `discretize.py`. Keep `STATES` and the bayes/markov variable order in sync.
* RBM size and training: `n_hid`, `rbm_epochs`, `k` in `pipeline.py` / `rbm.py`.
* Add features: append to `FEATURES` in `data.py` and extend the loader map.

## 8. Troubleshooting

* `ModuleNotFoundError: src` → run commands from inside the `ghostnet/` folder, and use `python -m src.xxx`.
* Dashboard says no trained models → run `python run_pipeline.py` or press **Train GhostNet**.
* CIC CSV: "missing required columns" → you used a file without the standard CIC headers.
* PyTorch install is slow → on a machine without a GPU, install the CPU build from pytorch.org.
