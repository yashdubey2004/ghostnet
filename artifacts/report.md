# GhostNet - experiment report

* Data source: **synthetic demo traffic**  (8400 train / 3600 test flows)
* Train class counts: {'Benign': 6681, 'PortScan': 612, 'DoS': 595, 'BruteForce': 512}
* Unsupervised models (RBM, VAE) are trained on benign training flows only (6681 flows); thresholds = 95th percentile of benign-train scores.

## Detection performance (held-out test set)

| Model                             |   ROC AUC |   PR AUC |   Precision |   Recall |    F1 |   False-pos. rate |
|:----------------------------------|----------:|---------:|------------:|---------:|------:|------------------:|
| Logistic regression (supervised)  |     0.968 |    0.961 |       0.957 |    0.935 | 0.946 |             0.011 |
| Random forest (supervised)        |     0.97  |    0.962 |       0.997 |    0.935 | 0.965 |             0.001 |
| Bayesian network                  |     0.97  |    0.96  |       0.97  |    0.935 | 0.952 |             0.007 |
| Markov network                    |     0.972 |    0.96  |       0.968 |    0.936 | 0.952 |             0.008 |
| RBM free energy (unsupervised)    |     0.956 |    0.921 |       0.736 |    0.864 | 0.795 |             0.08  |
| VAE recon. error (unsupervised)   |     0.966 |    0.951 |       0.824 |    0.929 | 0.874 |             0.051 |
| RBM + VAE ensemble (unsupervised) |     0.965 |    0.942 |       0.776 |    0.921 | 0.842 |             0.068 |

## Detection rate per attack class (fraction flagged)

| Model                             |   BruteForce |   DoS |   PortScan |
|:----------------------------------|-------------:|------:|-----------:|
| Logistic regression (supervised)  |        0.945 | 0.933 |      0.927 |
| Random forest (supervised)        |        0.945 | 0.933 |      0.927 |
| Bayesian network                  |        0.945 | 0.933 |      0.927 |
| Markov network                    |        0.945 | 0.937 |      0.927 |
| RBM free energy (unsupervised)    |        0.694 | 0.933 |      0.939 |
| VAE recon. error (unsupervised)   |        0.922 | 0.933 |      0.931 |
| RBM + VAE ensemble (unsupervised) |        0.89  | 0.933 |      0.935 |

## Probabilistic graphical models

* Held-out average log-likelihood (nats/flow): Bayesian net -4.269, Markov net -4.280
* Markov network edges: [('Attack', 'PktSize'), ('Attack', 'Port'), ('Attack', 'Syn'), ('Attack', 'PktRate'), ('Attack', 'Asym'), ('PktRate', 'Duration'), ('PktSize', 'Asym'), ('Attack', 'Duration'), ('Syn', 'Asym')]

## Synthetic-data fidelity (lower KS / corr-dist is better; C2ST AUC near 0.5 is best)

|                                           |   ks_mean |   ks_max |   corr_dist |   c2st_auc |
|:------------------------------------------|----------:|---------:|------------:|-----------:|
| VAE detector (benign-only) vs benign test |     0.109 |    0.323 |       1.52  |      0.999 |
| VAE (all traffic) vs test                 |     0.092 |    0.206 |       0.701 |      0.997 |
| RBM Gibbs samples vs benign test          |     0.349 |    0.497 |       2.5   |      1     |

## Caveats

* Synthetic records are samples from learned models. They are NOT real attack traffic.
* Bayesian/Markov net posteriors reflect the learned data and the assumed structure, not a verified causal reconstruction of an attack.
* The built-in demo data are generated, so absolute scores say little about real networks; re-run on CIC-IDS2017 for real conclusions.