"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ngl_constants as C
from ngl_evaluation import t_label

_LOCK = dict(fixedrange=True)


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(**_LOCK)
    fig.update_yaxes(**_LOCK)
    return fig


# --- Experiment 1: die Treppe und die geglättete Kurve ------------------------------------------------------------------------------------

def build_hand_figure(curves: dict, h: dict) -> go.Figure:
    """Links: Regret gegen den Vorhersage-Abstand Δ = ĉ₁ − ĉ₂ (harte Treppe, geglättete Kurve); rechts: Steigung des Regrets (Treppe: überall null) und der geglätteten Kurve."""
    x = curves["x"]
    neg, pos = x <= 0, x > 0
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Regret", "Steigung nach ĉ₁"), horizontal_spacing=0.1)
    # Treppe: zwei Stufen, bei Δ = 0 gehört der Punkt zur unteren Stufe (Gleichstand: Weg 1)
    fig.add_trace(go.Scatter(x=x[neg], y=curves["hard"][neg], mode="lines", line=dict(color=C.COLOR_STEP, width=3), name="harter Löser (Treppe)",
                             hovertemplate="Δ = %{x:.2f}: Regret %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[pos], y=curves["hard"][pos], mode="lines", line=dict(color=C.COLOR_STEP, width=3), showlegend=False,
                             hovertemplate="Δ = %{x:.2f}: Regret %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.0], y=[2.0], mode="markers", marker=dict(color="white", line=dict(color=C.COLOR_STEP, width=2), size=8), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=curves["smooth"], mode="lines", line=dict(color=C.COLOR_SMOOTH, width=3), name=f"geglättet, T = {t_label(h['T'])}",
                             hovertemplate="Δ = %{x:.2f}: erwarteter Regret %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[h["delta"]], y=[h["regret"]], mode="markers", marker=dict(color="black", size=10, symbol="diamond"), name="Ihre Vorhersage (hart)",
                             hovertemplate="Ihr Δ: harter Regret %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[h["delta"]], y=[h["smooth_regret"]], mode="markers", marker=dict(color="black", size=9, symbol="circle-open", line=dict(width=2)), name="Ihre Vorhersage (geglättet)",
                             hovertemplate="Ihr Δ: erwarteter Regret %{y:.3f}<extra></extra>"), row=1, col=1)
    # Steigungen
    fig.add_trace(go.Scatter(x=x[neg], y=np.zeros(int(neg.sum())), mode="lines", line=dict(color=C.COLOR_STEP, width=3), showlegend=False, hovertemplate="Δ = %{x:.2f}: Steigung 0<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x[pos], y=np.zeros(int(pos.sum())), mode="lines", line=dict(color=C.COLOR_STEP, width=3), showlegend=False, hovertemplate="Δ = %{x:.2f}: Steigung 0<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=curves["slope"], mode="lines", line=dict(color=C.COLOR_SMOOTH, width=3), showlegend=False,
                             hovertemplate="Δ = %{x:.2f}: Steigung %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[h["delta"]], y=[h["grad_soft"]], mode="markers", marker=dict(color="black", size=9, symbol="circle-open", line=dict(width=2)), showlegend=False,
                             hovertemplate="Ihr Δ: Steigung %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Vorhersage-Abstand Δ = ĉ₁ − ĉ₂", row=1, col=1, range=list(C.HAND_X_RANGE))
    fig.update_xaxes(title_text="Δ", row=1, col=2, range=list(C.HAND_X_RANGE))
    fig.update_yaxes(range=[-0.15, 2.15], row=1, col=1)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=95), legend=dict(orientation="h", y=-0.32, x=0.0))
    return _lock(fig)


# --- Experiment 2: Anteil unveränderter Wege ----------------------------------------------------------------------------------------------

def build_share_figure(shares: dict, selected: float, band: dict | None = None) -> go.Figure:
    """Balken: Anteil der Testfälle (in %), in denen der Weg bei einer Störung um höchstens ±δ % derselbe bleibt (live, eine Instanz); die gewählte Stärke ist hervorgehoben.

    `band`: {δ: Median über die Seeds in Bruchteilen} aus der Vormessung (als Rauten), wenn die Welt der Messung entspricht."""
    keys = sorted(shares)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"±{t_label(k)} %" for k in keys], y=[100 * shares[k]["weg"] for k in keys], marker_color=[C.COLOR_STEP if k == selected else "#bbbbbb" for k in keys],
                         name="diese Instanz (live)", hovertemplate="%{x}: Weg unverändert in %{y:.1f} %<extra></extra>"))
    if band:
        bk = [k for k in keys if k in band]
        fig.add_trace(go.Scatter(x=[f"±{t_label(k)} %" for k in bk], y=[100 * band[k] for k in bk], mode="markers", marker=dict(symbol="diamond", size=10, color="black"),
                                 name="Median über 20 Zufalls-Seeds (vorgerechnet)", hovertemplate="%{x}: Median %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Testfälle mit gleichem Weg (%)", range=[0, 105])
    fig.update_xaxes(title_text="relative Störung der Vorhersage")
    fig.update_layout(height=330, margin=dict(l=40, r=20, t=20, b=95), legend=dict(orientation="h", y=-0.3, x=0.0))
    return _lock(fig)


# --- Experiment 3: Lernkurven -------------------------------------------------------------------------------------------------------------------

def build_learning_figure(run) -> go.Figure:
    """Regret auf den Testfällen nach t Adam-Schritten: KQ (Start), SPO+ und das geglättete Training je Temperatur; Orakel als Linie."""
    steps = sorted(run.curve)
    last = max(steps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, last], y=[100 * run.regret["kq"]] * 2, mode="lines", line=dict(color=C.COLOR_KQ, dash="dash"), name="KQ (Start)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, last], y=[100 * run.regret["orakel"]] * 2, mode="lines", line=dict(color=C.COLOR_ORACLE, dash="dot"), name="Orakel", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=steps, y=[100 * run.curve[s] for s in steps], mode="lines+markers", line=dict(color=C.COLOR_SPO, width=2, dash="dashdot"), name="SPO+",
                             hovertemplate="SPO+, Schritt %{x}: %{y:.2f} %<extra></extra>"))
    for T, curve in run.curve_gibbs.items():
        s = sorted(curve)
        fig.add_trace(go.Scatter(x=s, y=[100 * curve[k] for k in s], mode="lines+markers", line=dict(color=C.TEMP_COLORS.get(T, "#999999"), width=3), name=f"geglättet, T = {t_label(T)}",
                                 hovertemplate=f"T = {t_label(T)}, Schritt " + "%{x}: %{y:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Adam-Schritte")
    fig.update_yaxes(title_text="Regret auf den Testfällen (%)")
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=20, b=95), legend=dict(orientation="h", y=-0.28, x=0.0))
    return _lock(fig)


# --- Experiment 4: Temperatur und Kostenmaßstab -------------------------------------------------------------------------------------------

def build_scale_figure(base: dict, scaled: dict, factor: int, r_kq: float) -> go.Figure:
    """Regret gegen die Temperatur (logarithmische Achse) für die Welt mit Kosten mal 1 und mal `factor` (eine Instanz, live); KQ als Linie."""
    fig = go.Figure()
    xs = sorted(base)
    fig.add_trace(go.Scatter(x=xs, y=[base[t] for t in xs], mode="lines+markers", line=dict(color=C.COLOR_SPO, width=3), name="Kosten mal 1",
                             hovertemplate="T = %{x:g}: %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=sorted(scaled), y=[scaled[t] for t in sorted(scaled)], mode="lines+markers", line=dict(color=C.COLOR_SMOOTH, width=3), name=f"Kosten mal {factor}",
                             hovertemplate="T = %{x:g}: %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[min(xs), max(xs)], y=[r_kq] * 2, mode="lines", line=dict(color=C.COLOR_KQ, dash="dash"), name="KQ (beide Fälle gleich)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Temperatur T (logarithmisch, in Kosteneinheiten)", type="log")
    fig.update_yaxes(title_text="Regret auf den Testfällen (%)")
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=20, b=95), legend=dict(orientation="h", y=-0.3, x=0.0))
    return _lock(fig)
