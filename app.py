"""Warum der Gradient fehlt: der Löser ist eine Treppe - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Der Regret des vom Löser gewählten Weges ist stückweise konstant, sein Gradient fast überall null. Die Gibbs-Glättung (Verteilung über alle Wege, Temperatur T) macht daraus eine glatte
Funktion mit Gradient; im 5 x 5 Gitter werden alle 70 Wege aufgezählt. Aufbau nach DEMO-PLAYBOOK §2. Kern ohne Streamlit: ngl_rng, ngl_grid, ngl_world, ngl_losses, ngl_training, ngl_gibbs, ngl_runs.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import ngl_bands as B
import ngl_constants as C
from ngl_evaluation import (
    Settings,
    band_rows,
    band_str,
    de,
    best_temperature,
    hand_curves,
    hand_state,
    learning_rows,
    reach_share,
    share_band,
    t_label,
    verdict_core,
    verdict_hand,
    verdict_learning,
    verdict_perturbation,
    verdict_scale,
    world_shares,
)
from ngl_layout import dunkles_design, kopf_css
from ngl_pdf_export import build_pdf
from ngl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ngl_runs import RunSpec, gibbs_sweep, run_world
from ngl_visualization import build_hand_figure, build_learning_figure, build_scale_figure, build_share_figure

st.set_page_config(page_title="Nullgradient und Glättung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Training läuft: KQ, SPO+ und vier geglättete Verfahren mit je 300 Adam-Schritten (das dauert einige Sekunden) …")
def _world_run(deg: int, noise: float, n: int, seed: int):
    return run_world(Settings(deg=deg, noise=noise, n=n, seed=seed).run_spec())


@st.cache_data(show_spinner=False)
def _shares(deg: int, noise: float, n: int, seed: int):
    return world_shares(_world_run(deg, noise, n, seed))


@st.cache_data(show_spinner="Training läuft: die geglättete Regret-Minimierung für sieben Temperaturen, in der Welt und in der Welt mit skalierten Kosten (das dauert einige Sekunden) …")
def _sweep(deg: int, noise: float, n: int, seed: int, scale: int):
    spec = RunSpec(deg=deg, noise=noise, n=n, seed=seed, p=C.N_FEATURES, h=C.GRID_H, steps=C.STEPS, lr=C.LR, n_test=C.N_TEST, scale=float(scale))
    return gibbs_sweep(spec, C.SCALE_TEMPS)


def _pct(x: float) -> str:
    return f"{de(100 * x)} %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ngl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Warum der Gradient fehlt: der Löser ist eine Treppe")
st.markdown(
    """
Wenn die Entscheidung ein **Weg im Netz** ist und ein Modell die Kosten vorhersagt, entscheidet der **Löser**: er wählt den billigsten Weg. Als Funktion der Vorhersage ist der **Regret** dieses Weges eine
**Treppe**: er ändert sich nur, wenn der gewählte Weg wechselt, und ist dazwischen konstant, der Gradient also fast überall **null**. Wer auf dem Regret lernen will, findet keine Richtung.
Eine Abhilfe ist, den Löser zu **glätten**: statt des einen billigsten Weges eine **Gibbs-Verteilung über alle Wege** mit einer Temperatur T. Hier im 5 × 5 Gitter (40 Bögen, 70 Wege): erst ein Fall mit zwei Wegen von Hand,
dann der Anteil der Fälle, in denen eine kleine Störung nichts ändert, dann das Training mit geglättetem Löser und die Frage, wie T zum Kostenmaßstab passt.

Wer es zuerst ohne Fachbegriffe verstehen will, öffnet „💬 In einfachen Worten“. Weiter unten: **🔬 vier Experimente**, **🔗 Brücke: die Ableitung am Ergebnis**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/loeser-treppe-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Modell sagt Kosten voraus, und ein Löser wählt daraus den billigsten Weg. Der Regret dieses Weges, also die Mehrkosten gegenüber dem besten, ändert sich nur, wenn der gewählte Weg wechselt. Dazwischen ist er konstant, wie eine Treppe mit flachen Stufen. Wer das Modell auf den Regret trainieren will, findet auf einer Stufe keine Richtung, in die er drehen soll.

Die Glättung baut daraus eine Rampe: Jeder Weg bekommt ein Gewicht, das mit seinen vorhergesagten Kosten sinkt, und der Regret wird über alle Wege gemittelt. Die Temperatur regelt, wie scharf die billigen Wege bevorzugt werden. Ist sie zu klein, kehrt die Treppe zurück. Ist sie zu groß, verwischt alles. Die Ergebnisse gelten für dieses Gitter, ein lineares Modell und erzeugte Kosten.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/loeser-treppe-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Glättung", expanded=True):
    st.markdown(
        """
1. **Kosten vorhersagen:** ein lineares Modell macht aus den Merkmalen $x$ einen Kostenvektor $\\hat c$ für alle 40 Bögen. Die Kostenwelt ist nichtlinear (Grad 1 bis 6) und kann verrauscht sein.
2. **Der harte Löser** wählt den billigsten Weg $w(\\hat c)$; bewertet mit den **wahren** Kosten ergibt sich der **Regret** $c^\\top w(\\hat c) - z^*(c)$. Er ist in $\\hat c$ stückweise konstant.
3. **Der geglättete Löser** gibt jedem Weg das Gewicht $\\pi_T(w) \\propto \\exp(-\\hat c^\\top w / T)$. Die **erwarteten** Kosten $\\mathbb E_{\\pi_T}[c^\\top w]$ sind in $\\hat c$ überall glatt, ihr Gradient ist die Kovarianz von Wegkosten und Weg geteilt durch $-T$. Mit $T \\to 0$ wird daraus wieder der harte Löser, mit $T \\to \\infty$ der Mittelwert aller Wege.
4. **Training:** Adam auf den erwarteten Kosten (genauer: erwartete minus beste Kosten), Start bei Kleinsten Quadraten (KQ), 300 Schritte; gemessen wird nach dem Training mit dem **harten** Löser auf 1000 Testfällen.
5. **Rechnen:** im 5 × 5 Gitter werden alle 70 Wege aufgezählt. Dieselben Gewichte lassen sich auf dem azyklischen Gitter auch ohne Wegliste durch eine Vorwärts-Rückwärts-Rechnung bestimmen; sie dient in den Tests als zweiter Rechenweg.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste: Schnellstart und Kostenwelt -------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kostenwelt**")
    deg = st.select_slider("Grad der Wahrheit", options=list(C.DEG_CHOICES), key="deg_slider",
                           help="1 = die Kosten hängen linear von den Merkmalen ab (ein lineares Modell kann sie exakt abbilden); je größer, desto stärker weicht die Wahrheit von jedem linearen Modell ab.")
    noise = st.select_slider("Rauschstufe", options=list(C.NOISE_CHOICES), key="noise_slider", format_func=lambda v: de(v),
                             help="Jeder Bogen kostet das Mittel mal einen Faktor aus [1 − e, 1 + e]. Bei 0 sind die Kosten ohne Rauschen, bei 0,5 liegt auch das Orakel beim Regret deutlich über null.")
    n = st.select_slider("Trainingsfälle n", options=list(C.N_CHOICES), key="n_slider",
                         help="Zahl der Fälle (Merkmale mit beobachteten Kosten); das lineare Modell hat 6 × 40 = 240 Parameter.")
    seed = st.number_input("Zufalls-Seed der Welt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt generieren", width="stretch", on_click=randomize_seed)
    st.caption("Die Regler der Experimente (Abstand Δ und Temperatur im Handbeispiel, Störungsgröße, Kostenfaktor) stehen beim jeweiligen Experiment.")

deg, noise, n, seed = int(deg), float(noise), int(n), int(seed)
run = _world_run(deg, noise, n, seed)
shares = _shares(deg, noise, n, seed)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis dieser Welt")
st.caption(f"Eine Instanz (Zufalls-Seed {seed}): {n} Trainingsfälle, {C.N_TEST} Testfälle, normierter Regret = Summe der Regrets der gewählten Wege geteilt durch die Summe der besten Wegkosten; "
           "gemessen mit dem harten Löser nach 300 Adam-Schritten.")
m1, m2, m3 = st.columns(3)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Regret Orakel", _pct(run.regret["orakel"]), help="Das Orakel kennt die mittleren Kosten der Testfälle; sein Regret ist der unvermeidbare Teil (Rauschen).")
m2.metric("Regret KQ (Start)", _pct(run.regret["kq"]), delta=f"{de(100 * (run.regret['kq'] - run.regret['orakel']))} Punkte über dem Orakel", delta_color="off")
m3.metric("Regret SPO+", _pct(run.regret["spo"]), delta=f"{de(100 * (run.regret['spo'] - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse",
          help="SPO+ ist der konvexe Ersatzverlust für den Regret, mit dem Löser in der Schleife (Vergleichsspalte).")
gcols = st.columns(len(C.TRAIN_TEMPS))
for col, T in zip(gcols, C.TRAIN_TEMPS):
    r = run.regret_gibbs[T]
    col.metric(f"Geglättet, T = {t_label(T)}", _pct(r), delta=f"{de(100 * (r - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse")

settings = Settings(deg, noise, n, seed, float(st.session_state.get("delta_slider", C.DEFAULT_DELTA)), float(st.session_state.get("hand_t_slider", C.DEFAULT_HAND_T)),
                    float(st.session_state.get("pert_slider", C.DEFAULT_PERT)), int(st.session_state.get("scale_slider", C.DEFAULT_SCALE)))
if st.button("📄 PDF dieser Welt erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, run, shares))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="nullgradient-glaettung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Warum steht das Lernen auf dem Regret still, und was ändert die Glättung?")
kind, text = verdict_core(1.0, shares[1.0]["regret"], 100 * run.regret["kq"], C.TRAIN_TEMPS[0], 100 * run.regret_gibbs[C.TRAIN_TEMPS[0]], 1.0, 100 * run.regret_gibbs[1.0])
getattr(st, kind)(text)
st.caption("Das Verdikt gilt für diese eine Instanz und das lineare Modell; Mediane über 20 Zufalls-Seeds stehen in den Experimenten 2 bis 4, wo die Welt der Messung entspricht, und in der README.")

st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: die Treppe von Hand, zwei Wege")
st.markdown(
    "Zwei Wege mit je einem Bogen. Die **wahren Kosten** sind $c = (3, 5)$: Weg 1 ist der richtige. Stellen Sie den **Vorhersage-Abstand** $\\Delta = \\hat c_1 - \\hat c_2$ ein "
    "(nur er zählt; $\\hat c = (4, 3)$ ist $\\Delta = 1$) und eine **Temperatur** $T$ für die Glättung."
)
h1, h2 = st.columns(2)
with h1:
    delta = st.slider("Vorhersage-Abstand Δ = ĉ₁ − ĉ₂", C.DELTA_MIN, C.DELTA_MAX, key="delta_slider", step=C.DELTA_STEP, format="%.2f",
                      help="Negativ: Weg 1 (der richtige) hat die kleineren vorhergesagten Kosten. Positiv: der Löser wählt den falschen Weg 2. 0: Gleichstand, der Löser nimmt Weg 1.")
with h2:
    hand_t = st.select_slider("Temperatur T (Handbeispiel)", options=list(C.HAND_T_CHOICES), key="hand_t_slider", format_func=t_label,
                              help="Je kleiner T, desto näher ist die geglättete Kurve an der Treppe; je größer, desto flacher.")
hs = hand_state(float(delta), float(hand_t))
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Harter Löser wählt", f"Weg {hs['chosen']}", help="Der Weg mit den kleineren vorhergesagten Kosten; bei Gleichstand Weg 1.")
k2.metric("Harter Regret", de(hs["regret"]), help="Wahre Kosten des gewählten Weges minus die besten wahren Kosten (3): 0 oder 2.")
k3.metric("Steigung des harten Regrets", "springt" if hs["tie"] else "0", help="Der harte Regret ist zwischen den Sprüngen konstant: die Steigung ist null, nur bei Δ = 0 springt er.")
k4.metric("Gewichte Weg 1 / Weg 2", f"{de(hs['w1'], 4)} / {de(hs['w2'], 4)}", help="Gibbs-Gewichte bei der gewählten Temperatur.")
k5.metric("Erwartete Kosten", de(hs["ecost"], 4), help="3 mal Gewicht Weg 1 plus 5 mal Gewicht Weg 2.")
st.plotly_chart(build_hand_figure(hand_curves(float(hand_t)), hs), width="stretch", key="hand_chart")
st.caption("Links: der harte Regret (rot) ist eine Treppe mit der Stufe bei Δ = 0 (offener Kreis: dort gilt die untere Stufe); die geglättete Kurve (gelb) ist der erwartete Regret 2 · Gewicht(Weg 2). "
           "Rechts: die Steigung der Treppe ist überall null (bis auf den Sprung bei Δ = 0), die der geglätteten Kurve ist überall positiv, fern von Δ = 0 aber sehr klein.")
kind, text = verdict_hand(hs)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: wie oft ändert eine kleine Störung etwas?")
st.markdown(
    "Die KQ-Vorhersagen der 1000 Testfälle werden um einen zufälligen Bruchteil bis zur gewählten **Störungsgröße** verändert (jeder Eintrag einzeln, dieselbe Zufallsrichtung für alle Größen). "
    "Gezählt wird, in wie vielen Fällen der gewählte Weg danach **derselbe** ist; dort ist der Regret konstant und ein Differenzenquotient null."
)
pert = st.select_slider("Störungsgröße (relativ, in %)", options=list(C.PERT_CHOICES), key="pert_slider", format_func=t_label,
                        help="Jeder Eintrag der Vorhersage wird mit 1 + δ · u multipliziert, u gleichverteilt zwischen −1 und 1; δ ist die gewählte Größe.")
pert = float(pert)
band_med = None
if B.has_cell(deg, n, noise):
    band_med = {p: share_band(deg, n, noise, p)["weg"]["median"] for p in C.BAND_PERTS}
st.plotly_chart(build_share_figure(shares, pert, band_med), width="stretch", key="share_chart")
kind, text = verdict_perturbation(pert, shares[pert]["weg"], shares[pert]["regret"])
getattr(st, kind)(text)
if B.has_cell(deg, n, noise):
    sb = share_band(deg, n, noise, pert)
    st.caption(
        f"Vorgerechnet (Grad {deg}, Rauschstufe {de(noise)}, n = {n}, {sb['seeds']} Zufalls-Seeds, Störung ±{t_label(pert)} %): Anteil gleicher Wege im Median {de(100 * sb['weg']['median'])} % "
        f"(Minimum {de(100 * sb['weg']['min'])} %, Maximum {de(100 * sb['weg']['max'])} %)."
    )
else:
    st.caption(f"Für Grad {deg}, Rauschstufe {de(noise)} und n = {n} gibt es keine vorgerechnete Messung über Zufalls-Seeds; die Balken zeigen diese eine Instanz.")

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Lernen mit Glättung")
st.markdown(
    "Dasselbe lineare Modell, derselbe Start (KQ), dieselbe Lernrate und **300 Adam-Schritte**; verschieden ist nur das Lernziel: SPO+ oder der erwartete Regret unter der Gibbs-Verteilung mit Temperatur T. "
    "Nach jedem Teil der Schritte wird mit dem **harten** Löser auf den Testfällen gemessen. Bei **T = 0,001** ist die Verteilung fast der harte Löser: der Gradient ist fast überall null."
)
st.plotly_chart(build_learning_figure(run), width="stretch", key="learning_chart")
reach_rows = [{"Temperatur T": t_label(T), "Trainingsfälle mit mehr als einem Weg über 1 % Gewicht (am Start)": f"{de(100 * reach_share(run.grid.paths(), run.coef_kq, run.data.X_train, T))} %"} for T in C.TRAIN_TEMPS]
st.dataframe(pd.DataFrame(reach_rows), hide_index=True, width="stretch")
st.caption("Nur in Fällen, in denen die Verteilung mehr als einen Weg nennenswert gewichtet, hat der geglättete Regret eine Steigung, die in das Training eingeht (Start: KQ-Parameter).")
table3 = pd.DataFrame([{"Verfahren": r["Verfahren"], "Regret nach 300 Schritten (diese Instanz)": f"{de(r['regret'])} %"} for r in learning_rows(run)])
st.dataframe(table3, hide_index=True, width="stretch")
kind, text = verdict_learning(100 * run.regret["kq"], 100 * run.regret["spo"], {T: 100 * r for T, r in run.regret_gibbs.items()})
getattr(st, kind)(text)
if B.has_cell(deg, n, noise):
    rows3 = band_rows(deg, n, noise)
    st.markdown(f"**Vorgerechnet** (Grad {deg}, Rauschstufe {de(noise)}, n = {n}, {rows3[0]['seeds']} Zufalls-Seeds): Regret in Prozent, Median [Minimum; Maximum], dazu in wie vielen Seeds besser als KQ.")
    st.dataframe(pd.DataFrame([{"Verfahren": r["Verfahren"], "Regret (Median [Min; Max])": band_str(r["band"]) + " %",
                                "besser als KQ": "–" if r["besser"] is None else f"{r['besser']} von {r['seeds']} Seeds"} for r in rows3]), hide_index=True, width="stretch")
    st.caption("Die Unterschiede zwischen den mittleren Temperaturen liegen innerhalb der Streuung über die Seeds; eine Rangfolge unter ihnen folgt aus der Messung nicht.")
else:
    st.caption(f"Für Grad {deg}, Rauschstufe {de(noise)} und n = {n} gibt es keine vorgerechnete Messung über Zufalls-Seeds (vorgerechnet sind die vier Voreinstellungen).")

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Temperatur und Kostenmaßstab")
st.markdown(
    "Dieselbe Welt, aber alle Kosten mit einem **Faktor** multipliziert (die Lernrate wird mit ihm skaliert). T hat die Einheit der Kosten: das passende T sollte mit dem Maßstab wandern. "
    "Gezeigt ist der Regret nach 300 Schritten für sieben Temperaturen, je eine Zehnerpotenz Abstand."
)
scale = st.select_slider("Kostenfaktor", options=list(C.SCALE_CHOICES), key="scale_slider", help="Alle Kosten der Welt (Training und Test) werden mit diesem Faktor multipliziert; der Regret in Prozent bleibt vergleichbar.")
scale = int(scale)
sw1 = _sweep(deg, noise, n, seed, 1)
swf = _sweep(deg, noise, n, seed, scale)
base = {T: 100 * r for T, r in sw1["gibbs"].items()}
scaled = {T: 100 * r for T, r in swf["gibbs"].items()}
st.plotly_chart(build_scale_figure(base, scaled, scale, 100 * sw1["kq"]), width="stretch", key="scale_chart")
st.dataframe(pd.DataFrame([{"Temperatur T": t_label(T), "Regret, Kosten mal 1": f"{de(base[T])} %", f"Regret, Kosten mal {scale}": f"{de(scaled[T])} %"} for T in C.SCALE_TEMPS]), hide_index=True, width="stretch")
kind, text = verdict_scale(scale, base, scaled, 100 * sw1["kq"])
getattr(st, kind)(text)
if B.has_scale_cell(deg, n, noise) and scale in C.BAND_SCALES:
    b1, bf = B.scale_cell(noise, 1), B.scale_cell(noise, scale)
    st.caption(
        f"Vorgerechnet (Grad 6, n = 1000, Rauschstufe {de(noise)}, {len(b1['regret_kq'])} Zufalls-Seeds, Median des Regrets): beste Temperatur bei Kosten mal 1 T = "
        f"{t_label(best_temperature({float(t): v['median'] for t, v in b1['summary'].items()})[0])}, bei Kosten mal {scale} T = {t_label(best_temperature({float(t): v['median'] for t, v in bf['summary'].items()})[0])}."
    )
else:
    st.caption("Die Messung über Zufalls-Seeds liegt für Grad 6, n = 1000 und die Kostenfaktoren 1 und 10 vor (Details in der README).")

st.markdown("---")

# --- Brücke -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔗 Brücke: die Ableitung am Ergebnis")
st.markdown(
    """
Die Glättung **ändert das Problem**, um einen Gradienten zu bekommen. Der andere Weg lässt das Problem, wie es ist, und **leitet seine Lösung ab**: die implizite Differentiation am Optimum.
[`kegel-ableitung-demo`](https://github.com/sebastian-hanisch/kegel-ableitung-demo) macht das für Kegelprogramme: eine Gleichung für die Ableitung, eine Lösung des Gleichungssystems.
Nach der README dieser Demo gilt die Ableitung nur für **eindeutige, reguläre Lösungen**; an Gleichständen hat die Lösung einen **Knick** (dort entscheidet das Rauschen des Lösers, welche Seite die Ableitung nennt), und bei nicht eindeutigen Lösungen ist das System **singulär**.

Das ist die Lage der Treppe: bei einem kürzesten-Weg-LP ist die Lösung in den Kosten stückweise konstant (Ableitung null, wo sie existiert, eigene Überlegung, in Experiment 2 gemessen); die Stufen liegen genau bei den **Gleichständen** zweier Wege (Δ = 0 in Experiment 1), und dort ist die Lösung nicht eindeutig.
Die Demo rechnet kein Gitter; hier wird nichts von ihr nachgebaut. Ein zweiter, entgegengesetzter Weg, ein Verfahren abzuleiten, ist das Entrollen: [`entrollte-netze-demo`](https://github.com/sebastian-hanisch/entrollte-netze-demo) (Ableitung durch den Algorithmus, nicht am Ergebnis).
"""
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
_main = B.cell(6, 1000, 0.5)
_lin = B.cell(1, 1000, 0.0)
_small = B.cell(6, 100, 0.5)
_sc0 = B.scale_cell(0.0, 1)
_sc5 = B.scale_cell(0.5, 1)
_below5 = [T for T in C.SCALE_TEMPS if _sc5["summary"][f"{T:g}"]["median"] < _sc5["summary_kq"]["median"]]
_below0 = [T for T in C.SCALE_TEMPS if _sc0["summary"][f"{T:g}"]["median"] < _sc0["summary_kq"]["median"]]
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Alle Wege lassen sich aufzählen** | Das Gitter h × h hat binomial(2 (h − 1), h − 1) Wege: {math.comb(8, 4)} bei h = 5, {math.comb(10, 5)} bei h = 6, {math.comb(12, 6)} bei h = 7, {math.comb(18, 9)} bei h = 10. Die Aufzählung wächst damit schnell. Auf dem **azyklischen** Gitter geht es auch ohne Wegliste (Vorwärts-Rückwärts, in den Tests als zweiter Rechenweg); für allgemeine Polytope (Wege mit Nebenbedingungen) gibt es diese Struktur nicht, und die Demo behandelt sie nicht. |
| **T ist nicht null** | Bei T = 0,001 endet der Regret in der Messung (Grad 6, n = 1000, Rauschstufe 0,5, Median über {len(_main['regret_kq'])} Seeds) bei {de(_main['summary']['regret_gibbs_0.001']['median'])} % gegen {de(_main['summary']['regret_kq']['median'])} % für KQ: der Gradient ist fast überall null und das Lernen bringt nichts, hier sogar Verlust. |
| **T ist nicht zu groß** | Bei Kosten mit 5. bis 95. Perzentil 1,0 bis 3,9 liegt der Regret für T = 100 bei {de(_sc5['summary']['100']['median'])} % und für T = 1000 bei {de(_sc5['summary']['1000']['median'])} % (Rauschstufe 0,5, {len(_sc5['regret_kq'])} Seeds, Median; KQ {de(_sc5['summary_kq']['median'])} %): auch zu grobe Glättung verliert. Unter KQ liegt der Median-Regret bei Rauschstufe 0,5 für T von {t_label(_below5[0])} bis {t_label(_below5[-1])}, ohne Rauschen für T von {t_label(_below0[0])} bis {t_label(_below0[-1])}. |
| **T passt zum Kostenmaßstab** | Die Zone gehört zu dieser Kostenskala (Kosten mit 5. bis 95. Perzentil 1,0 bis 3,9); Experiment 4 zeigt, dass sie mit dem Faktor wandert. Ein T aus einer anderen Skala zu übernehmen kann nach beiden Seiten danebenliegen. |
| **Die Wahrheit ist nicht linear** | Grad 1, Rauschstufe 0, n = 1000: KQ hat {de(_lin['summary']['regret_kq']['median'])} % Regret, SPO+ {de(_lin['summary']['regret_spo']['median'])} %, T = 1 {de(_lin['summary']['regret_gibbs_1']['median'])} % (Median, {len(_lin['regret_kq'])} Seeds): hier gibt es nichts zu gewinnen. |
| **Genug Daten** für 240 Parameter | Grad 6, Rauschstufe 0,5, n = 100: KQ {de(_small['summary']['regret_kq']['median'])} %, SPO+ {de(_small['summary']['regret_spo']['median'])} % (in {_small['spo_besser_als_kq']} von {len(_small['regret_kq'])} Seeds besser als KQ), T = 1 {de(_small['summary']['regret_gibbs_1']['median'])} % (in {_small['gibbs_besser_als_kq']['1']} von {len(_small['regret_kq'])} Seeds besser). |
| **Eine Familie, eine Modellklasse** | Nur 5 × 5, nur diese Kostenformel, nur lineares Modell, nur Gibbs-Glättung (keine quadratische Regularisierung des LP), Adam mit fester Lernrate. Aussagen über andere Netze, Modelle oder Löser sind nicht gemessen. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/loeser-treppe-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Fälle $(x_i, c_i)$ mit Merkmalen $x_i$ und beobachteten Bogenkosten $c_i$ und die Menge $W$ der Wege; gesucht ist ein Modell $\hat c(x)$, dessen gewählter Weg wenig Regret hat. Der harte Regret hat fast überall den Gradienten null; die Glättung mit der Temperatur $T$ gibt ihm einen.

| Symbol | Bedeutung |
|---|---|
| $p = 5$, $d = 40$ | Zahl der Merkmale, Zahl der Bögen im 5 × 5 Gitter |
| $W$, $\lvert W \rvert = 70$ | die Menge der Wege, je 8 Bögen |
| $c$, $\hat c$ | wahre und vorhergesagte Bogenkosten |
| $w(\hat c)$ | der Weg mit den kleinsten vorhergesagten Kosten; bei Gleichstand rechts vor unten |
| $z^*(c)$ | die besten wahren Kosten $\min_{w \in W} c^\top w$ |
| $\ell$, $\ell_T$ | Regret des harten Lösers, Regret der Glättung |
| $T > 0$ | Temperatur der Glättung, in Kosteneinheiten |
| $\pi_T(w \mid \hat c)$ | Gibbs-Gewicht des Weges $w$ |
| $\Delta$ | Vorhersage-Abstand $\hat c_1 - \hat c_2$ im Handbeispiel mit zwei Wegen |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/loeser-treppe-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Regret (harter Löser).**

$$\ell(\hat c, c) = c^\top w(\hat c) - z^*(c)$$

Er ändert sich nur, wenn sich der gewählte Weg ändert: in $\hat c$ stückweise konstant, der Gradient ist, wo er existiert, null.

**Gibbs-Glättung.**

$$\pi_T(w \mid \hat c) = \frac{\exp(-\hat c^\top w / T)}{\sum_{w' \in W} \exp(-\hat c^\top w' / T)}, \qquad \ell_T(\hat c, c) = \sum_{w \in W} \pi_T(w \mid \hat c)\, c^\top w - z^*(c)$$

Für $T \to 0$ geht $\pi_T$ auf den billigsten Weg, für $T \to \infty$ auf die Gleichverteilung über die Wege.

**Gradient.** Mit $\mathbb E$ und $\operatorname{Cov}$ unter $\pi_T$:

$$\frac{\partial \ell_T}{\partial \hat c} = -\frac{1}{T}\,\operatorname{Cov}\big(c^\top w,\; w\big) = -\frac{1}{T}\Big( \mathbb E[(c^\top w)\, w] - \mathbb E[c^\top w]\, \mathbb E[w] \Big)$$

Für das lineare Modell $\hat c = \theta^\top (1, x)$ folgt der Gradient nach $\theta$ aus der Kettenregel, gemittelt über die Fälle. Training: Adam (Lernrate 0,05, 300 Schritte, Start bei der KQ-Lösung).

**Handbeispiel.** Zwei Wege mit je einem Bogen, $c = (3, 5)$, $\Delta = \hat c_1 - \hat c_2$. Der harte Regret ist $0$ für $\Delta \le 0$ und $2$ für $\Delta > 0$. Das Gewicht von Weg 2 ist $\sigma(\Delta / T)$ mit $\sigma(u) = 1/(1 + e^{-u})$, der geglättete Regret also $2\,\sigma(\Delta/T)$, seine Steigung nach $\hat c_1$ ist $\tfrac{2}{T}\,\sigma'(\Delta/T) > 0$ für jedes $\Delta$. Bei $\hat c = (4, 3)$, $T = 1$: Gewichte $0{,}2689$ und $0{,}7311$ (gerundet), erwartete Kosten $3 \cdot (1 - \sigma(1)) + 5 \cdot \sigma(1) = 4{,}4621$ mit den exakten Gewichten; die gerundeten Gewichte ergäben $4{,}4622$.

**Zweiter Rechenweg (Summe-Produkt).** Auf dem azyklischen Gitter sei $a_e = -\hat c_e / T$, $\alpha_v$ die Log-Summe der Teilwege vom Start nach $v$ und $\beta_v$ die der Teilwege von $v$ zum Ziel:

$$\alpha_{h(e)} = \log\!\sum_{e': h(e') = h(e)} \exp\big(\alpha_{t(e')} + a_{e'}\big), \qquad \beta_{t(e)} = \log\!\sum_{e': t(e') = t(e)} \exp\big(a_{e'} + \beta_{h(e')}\big), \qquad \mathbb E[w_e] = \exp\big(\alpha_{t(e)} + a_e + \beta_{h(e)} - \log Z\big)$$

mit $t(e)$ und $h(e)$ Anfangs- und Endknoten des Bogens und $\log Z$ die Log-Summe aller Wege. Die erwarteten Kosten sind $c^\top \mathbb E[w]$.

**Welt.** $x \sim \mathcal N(0, I_p)$, $B \in \{0,1\}^{d \times p}$ mit unabhängigen fairen Münzwürfen, mittlere Kosten $m_j(x) = \big((Bx)_j / \sqrt{p} + 3\big)^{\deg} / 3{,}5^{\deg} + 1$, beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$.

**Literatur.** Für diese Demo wurde keine Quelle gelesen; die Gibbs-Glättung (Boltzmann-Verteilung über die Wege) und die Formeln sind selbst hergeleitet und in den Tests nachgerechnet. Das Gitter und die Kostenformel sind aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft; die SPO+-Spalte stammt aus `spo-kuerzester-weg-demo` (Elmachtoub und Grigas, Management Science 68(1), 2022, nur Abstract gelesen).
        """
    )

sync_query_params({
    "deg_slider": deg, "noise_slider": noise, "n_slider": n, "seed_input": seed,
    "delta_slider": float(delta), "hand_t_slider": float(hand_t), "pert_slider": float(pert), "scale_slider": int(scale),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html)."
)
