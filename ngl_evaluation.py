"""Kennzahlen, Verdikt-Logik und Tabellen für die App (keine Streamlit-Abhängigkeit): Handbeispiel, Lauf einer Welt, vorgerechnete Bänder.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.
Zahlen erscheinen mit Dezimalkomma (`de`). Regret-Werte sind in Prozent, Anteile als Bruchteil 0 bis 1 (Anzeige mit `de(100 * x)`)."""

from dataclasses import dataclass

import numpy as np

import ngl_bands as B
import ngl_constants as C
from ngl_gibbs import gibbs_parts, gibbs_weights
from ngl_grid import two_paths
from ngl_runs import RunSpec, summarize, unchanged_shares
from ngl_training import add_intercept, predict


def _shown(value: float, digits: int = 1) -> float:
    """Der Wert so, wie er mit `digits` Stellen angezeigt wird (Vergleiche stimmen mit dem überein, was die Leser sehen)."""
    return float(f"{float(value):.{digits}f}")


def de(value: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; negative Nullen (-0,0) erscheinen als 0,0."""
    text = f"{float(value):.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


def t_label(T: float) -> str:
    """Temperatur ohne überflüssige Nullen: 0,001 / 0,1 / 1 / 3 / 10."""
    return f"{T:g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    deg: int = C.DEFAULT_DEG
    noise: float = C.DEFAULT_NOISE
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    delta: float = C.DEFAULT_DELTA
    hand_t: float = C.DEFAULT_HAND_T
    pert: float = C.DEFAULT_PERT
    scale: int = C.DEFAULT_SCALE

    def run_spec(self) -> RunSpec:
        return RunSpec(deg=self.deg, noise=self.noise, n=self.n, seed=self.seed, p=C.N_FEATURES, h=C.GRID_H, steps=C.STEPS, lr=C.LR, n_test=C.N_TEST, temperatures=C.TRAIN_TEMPS)


# --- Experiment 1: das Handbeispiel ------------------------------------------------------------------------------------------------------

def hand_state(delta: float, T: float) -> dict:
    """Handbeispiel: zwei Wege mit je einem Bogen, wahre Kosten c = (3, 5), Vorhersage ch mit Abstand delta = ch_1 - ch_2 (nur der Abstand zählt) und Temperatur T.

    Hart: gewählt wird der Weg mit den kleineren vorhergesagten Kosten, bei Gleichstand Weg 1; der Regret ist 0 (Weg 1) oder 2 (Weg 2) und überall konstant, wo er sich nicht ändert.
    Geglättet: Gibbs-Gewichte über die zwei Wege, erwartete Kosten, geglätteter Regret und Gradient der erwarteten Kosten nach ch_1."""
    solver = two_paths()
    c = np.array([C.HAND_C])
    chat = np.array([[float(delta), 0.0]])
    W, _ = solver.solve(chat)
    chosen = int(np.argmax(W[0])) + 1
    pi = gibbs_weights(chat, solver.W_all, T)[0]
    ecost, grad = gibbs_parts(c, chat, solver.W_all, T)
    best = float(min(C.HAND_C))
    return {
        "delta": float(delta), "T": float(T), "tie": delta == 0, "chosen": chosen, "regret": float(c[0] @ W[0] - best),
        "w1": float(pi[0]), "w2": float(pi[1]), "ecost": float(ecost[0]), "smooth_regret": float(ecost[0] - best), "grad_soft": float(grad[0, 0]),
    }


def hand_curves(T: float, points: int = C.HAND_POINTS) -> dict:
    """Regret gegen den Vorhersage-Abstand delta: harte Treppe und geglättete Kurve (Zwei-Wege-Beispiel), dazu die Steigung der geglätteten Kurve (die Steigung der Treppe ist überall null, außer bei delta = 0, wo sie springt)."""
    x = np.linspace(C.HAND_X_RANGE[0], C.HAND_X_RANGE[1], points)
    x = np.unique(np.concatenate([x, [0.0]]))
    states = [hand_state(v, T) for v in x]
    return {
        "x": x, "hard": np.array([s["regret"] for s in states]), "smooth": np.array([s["smooth_regret"] for s in states]),
        "slope": np.array([s["grad_soft"] for s in states]),
    }


def verdict_hand(h: dict) -> tuple[str, str]:
    """Verdikt des Handbeispiels aus den Zahlen von `hand_state`."""
    soft = (
        f"Geglättet (T = {t_label(h['T'])}): Gewichte {de(h['w1'], 4)} auf Weg 1 und {de(h['w2'], 4)} auf Weg 2, erwartete Kosten {de(h['ecost'], 4)}, "
        f"Steigung nach ĉ₁ {de(h['grad_soft'], 4)} (positiv: ĉ₁ zu senken senkt die erwarteten Kosten)."
    )
    if h["tie"]:
        return "info", "Gleichstand (Δ = 0): beide Wege gelten als gleich teuer, der Löser nimmt Weg 1. Der harte Regret springt hier von 0,0 auf 2,0, eine Steigung gibt es an dieser Stelle nicht. " + soft
    if h["chosen"] == 1:
        return "success", (
            "Der Löser wählt Weg 1, den richtigen: Regret 0,0. Seine Steigung nach ĉ ist null: eine kleine Änderung von Δ ändert nichts. " + soft
        )
    return "warning", (
        "Der Löser wählt Weg 2; er kostet in Wahrheit 5,0 statt 3,0: Regret 2,0. Die Steigung des Regrets nach ĉ ist null, sie sagt nicht, in welche Richtung ĉ zu ändern wäre. " + soft
    )


# --- Experiment 2: Anteil unveränderter Wege -------------------------------------------------------------------------------------------------

def world_shares(run) -> dict:
    """Anteil unveränderter Wege und unveränderten Regrets in dieser Instanz für jede Störungsgröße (Prozent) aus `C.PERT_CHOICES`: {Prozent: {"weg": ..., "regret": ...}}.
    Gestört wird die KQ-Vorhersage der Testfälle; Zufallsrichtung aus dem Zufalls-Seed der Welt."""
    shares = unchanged_shares(run.grid, predict(run.coef_kq, run.data.X_test), run.data.C_test, [p / 100.0 for p in C.PERT_CHOICES], run.spec.seed)
    return {p: shares[p / 100.0] for p in C.PERT_CHOICES}


def verdict_perturbation(pert_pct: float, weg: float, regret: float, n_cases: int = C.N_TEST) -> tuple[str, str]:
    """Verdikt zur Störung: Anteil der Testfälle mit gleichem Weg und mit gleichem Regret (Bruchteile 0 bis 1) bei zufälliger relativer Störung der KQ-Vorhersage um höchstens `pert_pct` Prozent."""
    head = (
        f"Bei einer zufälligen relativen Störung der Vorhersage um höchstens ±{de(pert_pct)} % bleibt der gewählte Weg in {de(100 * weg)} % der {n_cases} Testfälle derselbe; "
        f"der Regret ändert sich in {de(100 * regret)} % der Fälle nicht."
    )
    if _shown(100 * regret) >= 95:
        return "warning", head + " Für diese Fälle wäre ein Differenzenquotient des Regrets null: auf dem Regret allein gibt es dort keine Richtung, in die man das Modell ändern könnte."
    return "info", head + " Bei dieser Größe der Störung wechselt schon ein größerer Teil der Wege: ein Differenzenquotient des Regrets wäre mal null, mal ein Sprung."


def verdict_core(pert_pct: float, regret_share: float, r_kq: float, t_small: float, r_small: float, t_mid: float, r_mid: float) -> tuple[str, str]:
    """Verdikt der Kernfrage (eine Instanz, live): Anteil mit unverändertem Regret bei Störung `pert_pct`, dazu Regret von KQ, vom kleinsten und vom mittleren trainierten T (alle Regret-Werte in Prozent)."""
    base = (
        f"In dieser Instanz bleibt der Regret bei einer Störung um ±{de(pert_pct)} % in {de(100 * regret_share)} % der Testfälle gleich. "
        f"Auf diesem Regret trainiert, ergibt T = {t_label(t_small)} einen Regret von {de(r_small)} % und T = {t_label(t_mid)} einen von {de(r_mid)} %, KQ liegt bei {de(r_kq)} %."
    )
    if _shown(r_mid) < _shown(r_kq) and _shown(r_small) >= _shown(r_kq):
        return "success", base + " Die sehr kleine Temperatur bringt keinen Gewinn gegen KQ, die mittlere schon."
    if _shown(r_mid) < _shown(r_kq):
        return "info", base + " Beide Temperaturen liegen in dieser Instanz unter KQ."
    return "warning", base + " In dieser Instanz bringt auch die mittlere Temperatur keinen Gewinn gegen KQ."


# --- Experiment 3: Lernen mit Glättung -------------------------------------------------------------------------------------------------------

def learning_rows(run) -> list[dict]:
    """Zeilen der Tabelle des Laufs (eine Instanz): Orakel, KQ, SPO+ und das geglättete Training je Temperatur; Regret in Prozent auf den Testfällen."""
    rows = [
        {"Verfahren": "Orakel (kennt die mittleren Kosten)", "regret": 100 * run.regret["orakel"]},
        {"Verfahren": "KQ (Start)", "regret": 100 * run.regret["kq"]},
        {"Verfahren": "SPO+ (300 Schritte)", "regret": 100 * run.regret["spo"]},
    ]
    for T, r in run.regret_gibbs.items():
        rows.append({"Verfahren": f"geglättet, T = {t_label(T)}", "regret": 100 * r, "T": T})
    return rows


def verdict_learning(r_kq: float, r_spo: float, gibbs: dict) -> tuple[str, str]:
    """Verdikt des geglätteten Trainings in einer Instanz: `gibbs` = {T: Regret in Prozent nach 300 Schritten}. Vergleich der kleinsten Temperatur und der besten größeren mit KQ und SPO+."""
    t_small = min(gibbs)
    a = gibbs[t_small]
    others = {t: r for t, r in gibbs.items() if t != t_small}
    if _shown(a) > _shown(r_kq):
        first = f"T = {t_label(t_small)} endet bei {de(a)} % und damit über KQ ({de(r_kq)} %)."
    elif _shown(a) == _shown(r_kq):
        first = f"T = {t_label(t_small)} endet bei {de(a)} %, so hoch wie KQ ({de(r_kq)} %): kein Gewinn."
    else:
        first = f"T = {t_label(t_small)} endet bei {de(a)} %, unter KQ ({de(r_kq)} %)."
    if not others:
        return "info", first
    t_best = min(others, key=lambda t: (_shown(others[t]), t))
    b = others[t_best]
    second = f" Die beste größere Temperatur ist T = {t_label(t_best)} mit {de(b)} % (SPO+: {de(r_spo)} %)."
    if _shown(b) < _shown(r_kq):
        return "success", first + second
    return "warning", first + second + " Sie liegt in dieser Instanz nicht unter KQ."


def reach_share(grid_paths: np.ndarray, coef: np.ndarray, X: np.ndarray, T: float, threshold: float = 0.99) -> float:
    """Anteil der Fälle, in denen die Gibbs-Verteilung zur Vorhersage `Xa @ coef` mehr als einen Weg nennenswert gewichtet: das größte Gewicht liegt unter `threshold`.

    Nur in diesen Fällen hat der geglättete Regret eine Steigung, die für das Training zählt. Ein Maß für die Reichweite der Glättung bei Temperatur T (ein Zahlenwert, kein Gradient)."""
    pi = gibbs_weights(add_intercept(X) @ coef, grid_paths, T)
    return float(np.mean(pi.max(axis=1) < threshold))


# --- Experiment 4: Temperatur und Kostenmaßstab -----------------------------------------------------------------------------------------------

def best_temperature(curve: dict) -> tuple[float, float]:
    """Temperatur mit dem kleinsten Regret in einer Kurve {T: Regret in Prozent}; bei Gleichstand (auf eine Stelle gerundet) die kleinere."""
    t = min(curve, key=lambda k: (_shown(curve[k]), k))
    return t, curve[t]


def verdict_scale(factor: int, base: dict, scaled: dict, r_kq: float) -> tuple[str, str]:
    """Verdikt zum Kostenmaßstab: beste Temperatur bei Faktor 1 (`base`) und bei Faktor `factor` (`scaled`), je {T: Regret in Prozent}; `r_kq` der Regret von KQ (unabhängig vom Faktor)."""
    t1, r1 = best_temperature(base)
    tf, rf = best_temperature(scaled)
    ratio = tf / t1
    text = (
        f"Bei Kosten mal 1 liegt der kleinste Regret bei T = {t_label(t1)} ({de(r1)} %), bei Kosten mal {factor} bei T = {t_label(tf)} ({de(rf)} %); KQ liegt in beiden Fällen bei {de(r_kq)} %. "
        f"Die beste Temperatur hat sich um den Faktor {t_label(ratio)} verschoben."
    )
    if abs(np.log10(ratio) - np.log10(factor)) < 1e-9:
        return "success", text + f" Das ist der Kostenfaktor: T ist eine Größe in Kosteneinheiten und wandert mit dem Maßstab."
    return "info", text + f" Das weicht vom Kostenfaktor {factor} ab; in einer einzelnen Instanz liegen die Regret-Werte benachbarter Temperaturen dicht beieinander."


# --- vorgerechnete Bänder ---------------------------------------------------------------------------------------------------------------------

def band_str(s: dict) -> str:
    return f"{de(s['median'])} [{de(s['min'])}; {de(s['max'])}]"


def band_rows(deg: int, n: int, noise: float) -> list[dict]:
    """Zeilen der Tabelle für eine Zelle: Orakel, KQ, SPO+ und das geglättete Verfahren je Temperatur (Median [Minimum; Maximum] über die Seeds, in Prozent) und wie oft besser als KQ."""
    c = B.cell(deg, n, noise)
    s = c["summary"]
    seeds = len(c["regret_kq"])
    rows = [
        {"Verfahren": "Orakel", "band": s["regret_orakel"], "besser": None},
        {"Verfahren": "KQ", "band": s["regret_kq"], "besser": None},
        {"Verfahren": "SPO+ (300 Schritte)", "band": s["regret_spo"], "besser": c["spo_besser_als_kq"]},
    ]
    for T in C.BAND_TEMPS:
        rows.append({"Verfahren": f"geglättet, T = {t_label(T)}", "band": s[f"regret_gibbs_{T:g}"], "besser": c["gibbs_besser_als_kq"][f"{T:g}"]})
    for r in rows:
        r["seeds"] = seeds
    return rows


def share_band(deg: int, n: int, noise: float, pert_pct: float) -> dict:
    """Anteil unveränderter Wege (Median, Minimum, Maximum über die Seeds, Bruchteil 0 bis 1) bei Störung `pert_pct` Prozent: Weg und Regret."""
    c = B.cell(deg, n, noise)
    key = f"{pert_pct / 100.0:g}"
    return {"weg": summarize(c["anteil_weg"][key]), "regret": summarize(c["anteil_regret"][key]), "seeds": len(c["regret_kq"])}
