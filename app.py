"""GhostNet dashboard (Module 5).   Run with:   streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src import viz
from src.data import ATTACK_CLASSES, FEATURES, STAGE_OF, generate_synthetic, load_flows
from src.discretize import INDICATORS, STATES
from src.evaluate import fidelity, ks_per_feature, make_report
from src.pipeline import load_bundle, save_bundle, snap, train_all

st.set_page_config(page_title="GhostNet", page_icon="👻", layout="wide")
st.title("👻 GhostNet")
st.caption("Probabilistic cyber-threat investigation and generative simulation framework")

ENS = "RBM + VAE ensemble (unsupervised)"

# --------------------------------------------------------------------------------- sidebar: data + training
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Synthetic demo traffic", "Upload CIC-IDS2017 CSV"], label_visibility="collapsed")
    if source == "Synthetic demo traffic":
        n_events = st.slider("Flows to generate", 4000, 30000, 12000, 1000)
        seed = st.number_input("Random seed", 0, 9999, 7)
        upload = None
    else:
        upload = st.file_uploader("CIC-IDS2017 flow CSV", type="csv")
        st.caption("Columns are mapped automatically; flows are sub-sampled per class.")
    st.header("2 · Training")
    rbm_epochs = st.slider("RBM epochs (CD-1)", 10, 80, 40, 5)
    vae_epochs = st.slider("VAE epochs", 20, 150, 100, 10)
    if st.button("Train GhostNet", type="primary", width="stretch"):
        try:
            if source == "Synthetic demo traffic":
                df, src_name = generate_synthetic(int(n_events), int(seed)), "synthetic demo traffic"
            elif upload is not None:
                df, src_name = load_flows(pd.read_csv(upload, low_memory=False)), "uploaded CIC-IDS2017 data"
            else:
                st.warning("Upload a CSV first."); st.stop()
            with st.spinner("Training baseline, Bayesian net, Markov net, RBM and VAEs..."):
                st.session_state["bundle"] = train_all(df, src_name, rbm_epochs=rbm_epochs,
                                                       vae_epochs=vae_epochs, log=lambda m: None)
                save_bundle(st.session_state["bundle"])
            st.success("Trained.")
        except Exception as e:  # show a readable message instead of a stack trace
            st.error(f"Training failed: {e}")

if "bundle" not in st.session_state:
    saved = load_bundle()
    if saved is not None:
        st.session_state["bundle"] = saved
b = st.session_state.get("bundle")
if b is None:
    st.info("No trained models yet. Click **Train GhostNet** in the sidebar (about 30 seconds on the demo data), "
            "or run `python run_pipeline.py` once and reload this page.")
    st.stop()

te, sc = b.test_df, b.test_scores
ev = te.join(sc)
y_te = (te["label"] != "Benign").values
st.sidebar.caption(f"Loaded: {b.source} · {len(b.train_df):,} train / {len(te):,} test flows")

tabs = st.tabs(["Data", "Investigate an event", "Network & timeline", "Graphical models",
                "RBM", "VAE & synthetic data", "Evaluation"])

# --------------------------------------------------------------------------------- tab 1: data
with tabs[0]:
    c1, c2 = st.columns([1, 2])
    c1.subheader("Class balance")
    c1.plotly_chart(viz.class_counts(pd.concat([b.train_df, te])), width="stretch")
    c2.subheader("Feature distributions by class")
    feat = c2.selectbox("Feature", FEATURES, index=FEATURES.index("pkts_per_s"))
    c2.plotly_chart(viz.feature_hist(pd.concat([b.train_df, te]), feat), width="stretch")
    st.caption("Models are fitted on the 70% training split only; scaling and discretisation bounds also come from "
               "training data. Unsupervised models (RBM, VAE) see **benign training flows only**.")
    st.dataframe(te.head(200), width="stretch", height=240)

# --------------------------------------------------------------------------------- tab 2: investigate
with tabs[1]:
    mode = st.radio("Which events?", ["All test events", "Flagged by RBM+VAE ensemble", "Ground-truth attacks"],
                    horizontal=True)
    thr_e = b.thresholds[ENS]
    mask = {"All test events": np.ones(len(te), bool),
            "Flagged by RBM+VAE ensemble": (sc["anomaly_pct"] > thr_e).values,
            "Ground-truth attacks": y_te}[mode]
    ids = list(te.index[mask][:500])
    if not ids:
        st.warning("No events match this filter."); st.stop()
    sel = st.selectbox("Event", ids, format_func=lambda i: (
        f"#{te.loc[i, 'flow_id']} · {te.loc[i, 'src_ip']} → {te.loc[i, 'dst_ip']}:{int(te.loc[i, 'dst_port'])}"
        f" · true label: {te.loc[i, 'label']}"))
    row, srow = te.loc[sel], sc.loc[sel]

    m = st.columns(5)
    m[0].metric("RBM anomaly (pct of benign)", f"{srow['rbm_pct']:.0f}")
    m[1].metric("VAE anomaly (pct of benign)", f"{srow['vae_pct']:.0f}")
    m[2].metric("Random-forest P(attack)", f"{srow['rf_attack']:.2f}")
    m[3].metric("Bayes-net P(attack)", f"{srow['bn_attack']:.2f}")
    m[4].metric("Most likely explanation", srow["bn_pred"])
    st.caption("Percentile = share of benign training flows that look *less* unusual than this one.")

    left, right = st.columns(2)
    D = b.disc.transform(te.loc[[sel]])
    obs = {v: int(D[v].iloc[0]) for v in INDICATORS}
    with left:
        st.markdown("**Observed indicators**")
        st.dataframe(pd.DataFrame({"indicator": INDICATORS, "state": [STATES[v][obs[v]] for v in INDICATORS]}),
                     hide_index=True, width="stretch")
        use = st.multiselect("Evidence to condition on", INDICATORS, default=INDICATORS)
        evidence = {v: obs[v] for v in use}
        bn_post, mn_post = b.bn.posterior("Attack", evidence), b.mn.posterior("Attack", evidence)
        st.plotly_chart(viz.posterior_bar(bn_post, "Bayesian network: P(Attack | evidence)"), width="stretch")
        st.caption("Markov-network posterior: " + ", ".join(f"{c} {p:.1%}" for c, p in zip(ATTACK_CLASSES, mn_post)))
    with right:
        st.plotly_chart(viz.sequential_fig(b.bn.sequential_update(evidence)), width="stretch")
        err = b.vae.feature_errors(b.pre.transform(te.loc[[sel]]))[0]
        st.plotly_chart(viz.feature_error_bar(err), width="stretch")
    with st.expander("Raw flow record"):
        st.dataframe(te.loc[[sel]], width="stretch")
    st.caption("Posteriors are conditional on the *assumed* network structure and the training data. "
               "They rank plausible explanations; they do not prove what caused the event.")

# --------------------------------------------------------------------------------- tab 3: network & timeline
with tabs[2]:
    if ev["src_ip"].nunique() <= 1:
        st.warning("This dataset has no source/destination IP columns, so no host graph can be drawn. "
                   "Use the CIC-IDS2017 *TrafficLabelling* CSVs (they include IPs and timestamps).")
    else:
        t0, t1 = float(ev["timestamp"].min()), float(ev["timestamp"].max())
        rng_t = st.slider("Time window (s)", t0, t1, (t0, t1))
        score_choice = st.radio("Colour connections by", ["Bayes-net P(attack)", "Ensemble anomaly percentile"], horizontal=True)
        col, thr = (("bn_attack", 0.5) if score_choice.startswith("Bayes") else ("anomaly_pct", thr_e))
        w = ev[(ev["timestamp"] >= rng_t[0]) & (ev["timestamp"] <= rng_t[1])]
        st.plotly_chart(viz.network_graph(w, col, thr), width="stretch")
        st.subheader("Observed vs hypothesised attack activity")
        a, c = st.columns(2)
        a.plotly_chart(viz.timeline(w), width="stretch")
        c.plotly_chart(viz.stage_bars(w), width="stretch")
        st.caption("Each point is a flow the Bayesian network assigns to an attack family. Stage names "
                   f"({', '.join(f'{k}→{v}' for k, v in STAGE_OF.items() if k != 'Benign')}) are a coarse, "
                   "hypothesised mapping from attack family. The timeline shows when such flows were *observed*; it does "
                   "not reconstruct a verified causal attack chain.")

# --------------------------------------------------------------------------------- tab 4: graphical models
with tabs[3]:
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Bayesian network (directed, Unit I)")
        st.plotly_chart(viz.bn_graph(b.bn.graph()), width="stretch")
        node = st.selectbox("Show conditional probability table for", b.bn.vars, index=1)
        st.dataframe(b.bn.cpt_frame(node).style.format("{:.3f}"), width="stretch")
        st.markdown("**Conditional-independence check**")
        x, y_ = st.columns(2)
        vx = x.selectbox("X", b.bn.vars, index=3); vy = y_.selectbox("Y", b.bn.vars, index=2)
        given = st.multiselect("Given", [v for v in b.bn.vars if v not in (vx, vy)], default=["Attack"])
        if vx != vy:
            st.write(f"d-separated by the graph: **{b.bn.d_separated(vx, vy, given)}**  ·  "
                     f"I({vx};{vy} | {', '.join(given) or '∅'}) = **{b.bn.cmi(vx, vy, given):.4f}** nats")
            st.caption("If the graph says d-separated, the model's mutual information is exactly 0 by construction.")
    with g2:
        st.subheader("Markov network (undirected, Unit II)")
        st.plotly_chart(viz.factor_graph_fig(b.mn.factor_graph()), width="stretch")
        edge = st.selectbox("Learned factor φ(A,B)", b.mn.edges, format_func=lambda e: f"{e[0]} — {e[1]}")
        st.plotly_chart(viz.heatmap(b.mn.factor(edge), f"φ({edge[0]}, {edge[1]})  (>1 = states favoured together)"),
                        width="stretch")
        st.caption("Structure: Chow-Liu tree on mutual information + 3 extra edges. Parameters: exact maximum "
                   "likelihood (partition function computed by enumeration).")
    st.divider()
    st.subheader("What-if evidence builder")
    cols = st.columns(len(INDICATORS))
    wi = {}
    for c_, v in zip(cols, INDICATORS):
        pick = c_.selectbox(v, ["(unobserved)"] + STATES[v], key=f"wi_{v}")
        if pick != "(unobserved)":
            wi[v] = STATES[v].index(pick)
    p1, p2 = st.columns(2)
    p1.plotly_chart(viz.posterior_bar(b.bn.posterior("Attack", wi), "Bayesian network"), width="stretch")
    p2.plotly_chart(viz.posterior_bar(b.mn.posterior("Attack", wi), "Markov network"), width="stretch")
    st.metric("Held-out log-likelihood (nats/flow): Bayesian net vs Markov net",
              f"{b.ll['bn']:.3f}  vs  {b.ll['mn']:.3f}")

# --------------------------------------------------------------------------------- tab 5: RBM
with tabs[4]:
    h = b.hist["rbm"]
    a, c = st.columns(2)
    a.plotly_chart(viz.training_curves(h, ["recon_mse"], "RBM reconstruction error (CD-1)"), width="stretch")
    c.plotly_chart(viz.training_curves(h, ["fe_benign_test", "fe_attack_test"], "Mean free energy during training",
                                       {"fe_benign_test": "benign (test)", "fe_attack_test": "attack (test)"}),
                   width="stretch")
    st.caption("As training proceeds, benign flows get low free energy (high probability) while attacks stay high. "
               "The gap is what the RBM uses as an anomaly score.")
    st.plotly_chart(viz.score_hist(sc["rbm_fe"].values, te["label"].values, "RBM free energy on the test set",
                                   b.thresholds["RBM free energy (unsupervised)"]), width="stretch")
    st.subheader("Watch Gibbs sampling")
    s1, s2, s3 = st.columns(3)
    start = s1.selectbox("Start the chain from", ["Random bits", "A test attack flow", "A test benign flow"])
    steps = s2.slider("Gibbs steps", 5, 100, 40)
    cseed = s3.number_input("Chain seed", 0, 999, 0)
    Bte = b.enc.encode(b.pre.transform(te))
    if start == "Random bits":
        v0 = (np.random.default_rng(int(cseed)).random(Bte.shape[1]) < 0.5).astype(np.float32)
    elif start == "A test attack flow":
        v0 = Bte[np.flatnonzero(y_te)[int(cseed) % int(y_te.sum())]]
    else:
        v0 = Bte[np.flatnonzero(~y_te)[int(cseed) % int((~y_te).sum())]]
    V, FE = b.rbm.gibbs_chain(v0, steps, seed=int(cseed))
    st.plotly_chart(viz.gibbs_fig(b.enc.decode(V), FE), width="stretch")
    st.caption("Each step samples hidden units given the visible bits, then visible bits given the hidden units "
               "(a Markov chain whose stationary distribution is the RBM). Free energy usually falls as the state "
               "moves toward benign-like configurations.")
    st.info("Finding to report: with CD-1 training the RBM works well as an *anomaly detector* but its Gibbs samples "
            "drift from the data distribution (see the fidelity table in the Evaluation tab). Try more CD steps or "
            "persistent CD as an extension.")

# --------------------------------------------------------------------------------- tab 6: VAE
with tabs[5]:
    a, c = st.columns(2)
    a.plotly_chart(viz.training_curves(b.hist["vae"], ["loss", "recon", "kl"], "VAE detector: negative ELBO terms"),
                   width="stretch")
    sub = b.train_df.sample(min(2500, len(b.train_df)), random_state=0)
    Z = b.vae_all.latent(b.pre.transform(sub))
    Z2 = PCA(2).fit_transform(Z) if Z.shape[1] > 2 else Z
    c.plotly_chart(viz.latent_scatter(Z2, sub["label"].values), width="stretch")
    st.divider()
    st.subheader("Generate synthetic flow records")
    g1, g2, g3 = st.columns(3)
    which = g1.selectbox("Generator", ["VAE trained on all traffic", "VAE trained on benign only"])
    n_syn = g2.slider("Records", 100, 5000, 1000, 100)
    gseed = g3.number_input("Seed", 0, 9999, 0, key="gseed")
    vae_g = b.vae_all if which.startswith("VAE trained on all") else b.vae
    ref_mask = np.ones(len(te), bool) if which.startswith("VAE trained on all") else ~y_te
    Xr = b.pre.transform(te)[ref_mask]
    raw_syn, Xs = snap(b, vae_g.generate(n_syn, seed=int(gseed)))
    st.dataframe(raw_syn.round(3).head(100), width="stretch", height=240)
    st.download_button("Download synthetic records (CSV)", raw_syn.to_csv(index=False), "ghostnet_synthetic.csv",
                       "text/csv")
    f = fidelity(Xr, Xs)
    k = st.columns(4)
    k[0].metric("Mean KS (lower is better)", f"{f['ks_mean']:.3f}")
    k[1].metric("Worst-feature KS", f"{f['ks_max']:.3f}")
    k[2].metric("Correlation distance", f"{f['corr_dist']:.2f}")
    k[3].metric("Real-vs-synthetic AUC (0.5 = ideal)", f"{f['c2st_auc']:.3f}")
    ks = ks_per_feature(Xr, Xs)
    fsel = st.selectbox("Compare distribution of", FEATURES, index=FEATURES.index("duration"), key="fsel")
    d1, d2 = st.columns([2, 1])
    d1.plotly_chart(viz.real_vs_synth(Xr, Xs, fsel), width="stretch")
    d2.dataframe(ks.style.format({"KS statistic": "{:.3f}"}), hide_index=True, width="stretch", height=300)
    st.warning("Synthetic records are samples from a learned model. They are **not** real attack traffic, and a "
               "classifier can usually still tell them apart from real flows (AUC near 1 above).")

# --------------------------------------------------------------------------------- tab 7: evaluation
with tabs[6]:
    st.subheader("Detection performance on the held-out test set")
    st.dataframe(b.metrics.style.format("{:.3f}"), width="stretch")
    st.caption("Supervised models and the Bayesian/Markov nets use attack labels when fitting. RBM and VAE are "
               "unsupervised (benign-only training); their threshold is the 95th percentile of benign training scores.")
    st.subheader("Fraction of each attack class flagged")
    st.dataframe(b.per_class.style.format("{:.2f}"), width="stretch")
    roc_scores = {"Random forest": sc["rf_attack"], "Bayesian net": sc["bn_attack"], "RBM": sc["rbm_fe"],
                  "VAE": sc["vae_err"], "Ensemble": sc["anomaly_pct"]}
    st.plotly_chart(viz.roc_fig(te["label"].values, roc_scores), width="stretch")
    st.subheader("Fidelity of generated data")
    st.dataframe(pd.DataFrame(b.fidelity).T.style.format("{:.3f}"), width="stretch")
    st.download_button("Download experiment report (Markdown)", make_report(b), "ghostnet_report.md", "text/markdown")
