"""Plotly figure builders used by the Streamlit dashboard."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

from .data import ATTACK_CLASSES, FEATURES, STAGE_OF

COLORS = {"Benign": "#8c9aa8", "DoS": "#d1495b", "PortScan": "#edae49", "BruteForce": "#00798c"}


def _layout(fig, h=420, title=None):
    fig.update_layout(height=h, title=title, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


# ---------------------------------------------------------------------------------- data / network
def class_counts(df: pd.DataFrame):
    c = df["label"].value_counts().reindex(ATTACK_CLASSES).dropna().reset_index()
    c.columns = ["label", "flows"]
    return _layout(px.bar(c, x="label", y="flows", color="label", color_discrete_map=COLORS), 300)


def feature_hist(df: pd.DataFrame, feature: str):
    d = df.copy()
    d["value"] = np.log10(d[feature].clip(lower=1e-6))
    fig = px.histogram(d, x="value", color="label", color_discrete_map=COLORS, barmode="overlay",
                       nbins=60, opacity=0.65, labels={"value": f"log10({feature})"})
    return _layout(fig, 320)


def network_graph(ev: pd.DataFrame, score_col: str, thr: float, max_edges: int = 400):
    """Hosts as nodes, aggregated connections as edges. Red = suspicious (mean score > thr)."""
    agg = (ev.groupby(["src_ip", "dst_ip"])
             .agg(n=("flow_id", "size"), s=(score_col, "mean")).reset_index())
    agg = agg.sort_values("n", ascending=False).head(max_edges)
    G = nx.Graph()
    for r in agg.itertuples():
        G.add_edge(r.src_ip, r.dst_ip, n=r.n, s=r.s)
    if G.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(G, seed=3, k=0.9 / np.sqrt(max(G.number_of_nodes(), 1)), iterations=80)
    traces = []
    for sus, color, name in [(False, "rgba(140,154,168,0.35)", "normal"), (True, "#d1495b", "suspicious")]:
        xs, ys = [], []
        for a, b, d in G.edges(data=True):
            if (d["s"] > thr) == sus:
                xs += [pos[a][0], pos[b][0], None]; ys += [pos[a][1], pos[b][1], None]
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", name=f"{name} connections",
                                 line=dict(width=2.2 if sus else 0.8, color=color), hoverinfo="skip"))
    nodes = list(G.nodes)
    nscore = [np.average([d["s"] for _, _, d in G.edges(n, data=True)],
                         weights=[d["n"] for _, _, d in G.edges(n, data=True)]) for n in nodes]
    traces.append(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name="hosts",
        marker=dict(size=[8 + 3 * np.sqrt(G.degree(n)) for n in nodes], color=nscore, colorscale="YlOrRd",
                    cmin=0, cmax=max(1.0, max(nscore)), showscale=True, colorbar=dict(title="mean score"),
                    line=dict(width=1, color="white")),
        text=[f"{n}<br>degree {G.degree(n)}<br>mean score {s:.2f}" for n, s in zip(nodes, nscore)],
        hoverinfo="text"))
    fig = go.Figure(traces)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout(fig, 520)


def timeline(ev: pd.DataFrame, class_col: str = "bn_pred", max_points: int = 4000):
    d = ev[ev[class_col] != "Benign"]
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)
    d = d.assign(stage=d[class_col].map(STAGE_OF))
    fig = px.scatter(d, x="timestamp", y="src_ip", color=class_col, color_discrete_map=COLORS,
                     hover_data=["dst_ip", "stage"], labels={"timestamp": "time (s)", "src_ip": "source host"})
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    return _layout(fig, 380)


def stage_bars(ev: pd.DataFrame, class_col: str = "bn_pred", bins: int = 24):
    d = ev[ev[class_col] != "Benign"].copy()
    if d.empty:
        return go.Figure()
    d["window"] = pd.cut(d["timestamp"], bins=bins).apply(lambda i: round(i.mid, 0))
    g = d.groupby(["window", class_col], observed=True).size().reset_index(name="flows")
    g["hypothesised stage"] = g[class_col].map(STAGE_OF)
    fig = px.bar(g, x="window", y="flows", color=class_col, color_discrete_map=COLORS,
                 hover_data=["hypothesised stage"], labels={"window": "time window (s)"})
    return _layout(fig, 320)


# ---------------------------------------------------------------------------------- graphical models
def posterior_bar(post: np.ndarray, title="P(Attack | evidence)"):
    fig = go.Figure(go.Bar(x=list(ATTACK_CLASSES), y=post, marker_color=[COLORS[c] for c in ATTACK_CLASSES],
                           text=[f"{p:.1%}" for p in post], textposition="outside"))
    fig.update_yaxes(range=[0, 1.1], title="probability")
    return _layout(fig, 300, title)


def sequential_fig(seq: pd.DataFrame):
    long = seq.melt(id_vars="step", var_name="Attack", value_name="probability")
    fig = px.bar(long, x="step", y="probability", color="Attack", color_discrete_map=COLORS)
    fig.update_xaxes(categoryorder="array", categoryarray=list(seq["step"]), tickangle=-30)
    return _layout(fig, 360, "Belief update as evidence arrives")


def _centre_layout(G, centre="Attack"):
    others = [n for n in G.nodes if n != centre]
    return nx.shell_layout(G, nlist=[[centre], others])


def bn_graph(G: nx.DiGraph):
    pos = _centre_layout(G)
    fig = go.Figure()
    for a, b in G.edges:
        fig.add_annotation(x=pos[b][0], y=pos[b][1], ax=pos[a][0], ay=pos[a][1], xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=1.5,
                           arrowcolor="#6b7785", standoff=16)
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in G.nodes], y=[pos[n][1] for n in G.nodes], mode="markers+text",
                             text=list(G.nodes), textposition="top center",
                             marker=dict(size=34, color=["#d1495b" if n == "Attack" else "#00798c" for n in G.nodes],
                                         line=dict(width=2, color="white")), hoverinfo="text"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout(fig, 420, "Bayesian network (arrows = assumed causal influence)")


def factor_graph_fig(FG: nx.Graph):
    pos = nx.spring_layout(FG, seed=5, k=0.8)
    xs, ys = [], []
    for a, b in FG.edges:
        xs += [pos[a][0], pos[b][0], None]; ys += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#aab4be", width=1), hoverinfo="skip"))
    for kind, sym, col, size in [("variable", "circle", "#00798c", 30), ("factor", "square", "#edae49", 12)]:
        nodes = [n for n, d in FG.nodes(data=True) if d["kind"] == kind]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                                 mode="markers+text" if kind == "variable" else "markers",
                                 text=nodes if kind == "variable" else None, textposition="top center",
                                 hovertext=nodes, hoverinfo="text", name=kind,
                                 marker=dict(size=size, symbol=sym, color=col, line=dict(width=1, color="white"))))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout(fig, 440, "Markov network as a factor graph (squares = factors)")


def heatmap(df: pd.DataFrame, title: str):
    fig = px.imshow(df.values, x=list(df.columns), y=list(df.index), color_continuous_scale="RdBu_r",
                    text_auto=".2f", aspect="auto")
    return _layout(fig, 300, title)


# ---------------------------------------------------------------------------------- RBM / VAE
def training_curves(hist: dict, keys: list, title: str, labels: dict | None = None):
    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Scatter(x=hist["epoch"] if "epoch" in hist else list(range(len(hist[k]))), y=hist[k],
                                 mode="lines", name=(labels or {}).get(k, k)))
    fig.update_xaxes(title="epoch")
    return _layout(fig, 320, title)


def gibbs_fig(decoded: np.ndarray, fe: np.ndarray):
    """decoded: (steps+1, n_features) values in [0,1] after decoding the bits; fe: free energy per step."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Heatmap(z=decoded.T, x=list(range(decoded.shape[0])), y=FEATURES, colorscale="Viridis",
                             zmin=0, zmax=1, colorbar=dict(title="scaled value", len=0.6, y=0.75)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(fe))), y=fe, mode="lines+markers", name="free energy",
                             line=dict(color="#d1495b")), row=2, col=1)
    fig.update_xaxes(title="Gibbs step", row=2, col=1)
    fig.update_yaxes(title="F(v)", row=2, col=1)
    return _layout(fig, 520, "Gibbs sampling: visible state (decoded) and free energy per step")


def latent_scatter(Z2: np.ndarray, labels, title="VAE latent space (PCA of posterior means)"):
    d = pd.DataFrame({"PC1": Z2[:, 0], "PC2": Z2[:, 1], "label": labels})
    fig = px.scatter(d, x="PC1", y="PC2", color="label", color_discrete_map=COLORS, opacity=0.6)
    fig.update_traces(marker=dict(size=5))
    return _layout(fig, 420, title)


def real_vs_synth(real: np.ndarray, synth: np.ndarray, feature: str):
    i = FEATURES.index(feature)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=real[:, i], name="real", opacity=0.6, nbinsx=40, histnorm="probability density",
                               marker_color="#8c9aa8"))
    fig.add_trace(go.Histogram(x=synth[:, i], name="synthetic", opacity=0.6, nbinsx=40, histnorm="probability density",
                               marker_color="#00798c"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=f"{feature} (log-scaled to [0,1])")
    return _layout(fig, 300)


def score_hist(scores: np.ndarray, labels, title: str, thr: float | None = None):
    d = pd.DataFrame({"score": scores, "class": np.where(np.asarray(labels) == "Benign", "Benign", "Attack")})
    fig = px.histogram(d, x="score", color="class", barmode="overlay", opacity=0.65, nbins=60,
                       color_discrete_map={"Benign": "#8c9aa8", "Attack": "#d1495b"})
    if thr is not None:
        fig.add_vline(x=thr, line_dash="dash", annotation_text="threshold")
    return _layout(fig, 300, title)


def feature_error_bar(err: np.ndarray):
    d = pd.DataFrame({"feature": FEATURES, "squared error": err})
    return _layout(px.bar(d, x="feature", y="squared error", color="squared error", color_continuous_scale="YlOrRd"),
                   260, "Which features does the VAE find unusual?")


def roc_fig(y_true, score_dict: dict):
    y = (np.asarray(y_true) != "Benign").astype(int)
    fig = go.Figure()
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_curve(y, s)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#aaa"), showlegend=False))
    fig.update_xaxes(title="false-positive rate"); fig.update_yaxes(title="true-positive rate")
    return _layout(fig, 440, "ROC curves (test set)")
