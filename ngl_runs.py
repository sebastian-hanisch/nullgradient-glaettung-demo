"""Auswertung einer Welt: ein Lauf je Welt, Zusammenfassung über Zufalls-Seeds (keine Streamlit-Abhängigkeit, nur numpy).

Ein Lauf (`run_world`) zieht Trainings- und Testdaten zu einem Zufalls-Seed, trainiert KQ, SPO+ und das mit der Temperatur T geglättete Verfahren (Gibbs-Verteilung über alle 70 Wege)
und misst auf den Testfällen den normierten Regret; das Orakel kennt die mittleren Kosten der Testfälle. `run_seeds` wiederholt das über feste Seeds, `summarize` gibt
Median, Minimum und Maximum (Median über Seeds, nie ein einzelner Seed als Befund).

Herkunft: Gerüst (RunSpec, WorldRun, cell_result, summarize) aus `spo-kuerzester-weg-demo/skw_runs.py` kopiert und umbenannt; neu sind der Kostenfaktor `scale`, die Temperaturen,
das geglättete Training (`train_gibbs`), der Anteil unveränderter Wege bei Störung (`unchanged_shares`) und der Lauf über viele Temperaturen (`gibbs_sweep`).

Geglättetes Training: Adam mit demselben Start (KQ), derselben Lernrate (mal Kostenfaktor) und denselben Schritten wie bei SPO+; zurückgegeben wird der Parameter NACH dem letzten
Schritt (keine Auswahl über den Trainingsverlust; bei SPO+ wie im Gerüst der Parameter mit dem kleinsten Trainingsverlust unter den besuchten). Der Kostenfaktor `scale` multipliziert
alle Kosten der Welt; die Lernrate wird mit ihm skaliert, damit ein Lauf mit (scale, T * scale) bis auf Rundung derselbe ist wie einer mit (1, T) (die Tests prüfen das).
"""

from dataclasses import dataclass, field, replace

import numpy as np

from ngl_gibbs import gibbs_objective
from ngl_grid import Grid
from ngl_losses import mse_per_arc, normalized_regret, spo_loss
from ngl_rng import SplitMix64, derive_seed
from ngl_training import adam_minimize, fit_least_squares, predict, spo_plus_loss_mean, train_spo_plus, TrainResult
from ngl_world import Dataset, WorldSpec, make_dataset

SNAPSHOT_STEPS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300)
TAG_PERTURB = 4  # Teilstrom für die Störung der Vorhersage (neben Gewichtsmatrix 1, Training 2, Test 3 der Welt)


@dataclass(frozen=True)
class RunSpec:
    """Alles, was einen Lauf festlegt."""

    deg: int = 6
    noise: float = 0.0
    n: int = 1000
    seed: int = 7
    p: int = 5
    h: int = 5
    steps: int = 300
    lr: float = 0.05
    l2: float = 0.0
    n_test: int = 1000
    scale: float = 1.0                  # Kostenfaktor: alle Kosten der Welt mal scale (Lernrate wird mitskaliert)
    temperatures: tuple = ()            # Temperaturen T der geglätteten Verfahren, die der Lauf zusätzlich trainiert

    @property
    def world(self) -> WorldSpec:
        return WorldSpec(p=self.p, deg=self.deg, noise=self.noise)


@dataclass
class WorldRun:
    spec: RunSpec
    grid: Grid
    data: Dataset
    coef_kq: np.ndarray
    coef_spo: np.ndarray
    best_step: int
    train_losses: np.ndarray        # SPO+-Trainingsverlust vor jedem Schritt
    snapshots: dict                 # Schritt -> Parameter nach so vielen Schritten (SPO+)
    regret: dict                    # "orakel", "kq", "spo": normierter Regret auf den Testfällen
    mse: dict                       # "kq", "spo": mittlerer quadratischer Fehler je Bogen auf den Testkosten
    curve: dict                     # Schritt -> Testregret des Parameters nach so vielen Schritten (SPO+)
    train_loss_kq: float            # mittlerer SPO+-Trainingsverlust von KQ
    train_loss_spo: float           # mittlerer SPO+-Trainingsverlust des gewählten Parameters
    case_regret: dict               # "kq", "spo", "orakel": Regret je Testfall in Bruchteilen der besten Kosten (Länge n_test)
    coef_gibbs: dict = field(default_factory=dict)       # T -> Parameter nach dem letzten Schritt
    curve_gibbs: dict = field(default_factory=dict)      # T -> {Schritt: Testregret}
    train_losses_gibbs: dict = field(default_factory=dict)  # T -> geglätteter Trainingsregret vor jedem Schritt
    regret_gibbs: dict = field(default_factory=dict)     # T -> normierter Regret auf den Testfällen nach dem letzten Schritt


def make_world_data(spec: RunSpec, grid: Grid) -> Dataset:
    """Daten der Welt; bei `scale` != 1 sind alle Kosten (Training, Test, Mittel) mit dem Faktor multipliziert."""
    data = make_dataset(spec.world, grid.d, spec.n, spec.n_test, spec.seed)
    if spec.scale == 1.0:
        return data
    s = float(spec.scale)
    return replace(data, C_train=data.C_train * s, C_test=data.C_test * s, M_test=data.M_test * s)


def train_gibbs(paths: np.ndarray, X: np.ndarray, C: np.ndarray, T: float, steps: int = 300, lr: float = 0.05, init=None, snapshot_steps=()) -> TrainResult:
    """Geglättetes Training: Adam auf dem mittleren geglätteten Regret zur Temperatur T, Start bei KQ (oder bei `init`). Das Ergebnis des letzten Schritts steht in `snapshots[steps]`
    (nur wenn `steps` in `snapshot_steps` steht; `TrainResult.coef` ist wie bei SPO+ der beste besuchte Parameter nach dem geglätteten Trainingsverlust)."""
    coef0 = fit_least_squares(X, C) if init is None else np.array(init, dtype=float)
    return adam_minimize(gibbs_objective(paths, X, C, T), coef0, steps, lr, snapshot_steps)


def run_world(spec: RunSpec, snapshot_steps=SNAPSHOT_STEPS, keep_cases: bool = True) -> WorldRun:
    grid = Grid(spec.h)
    data = make_world_data(spec, grid)
    paths = grid.paths()
    lr = spec.lr * spec.scale
    coef_kq = fit_least_squares(data.X_train, data.C_train)
    res = train_spo_plus(grid, data.X_train, data.C_train, spec.steps, lr, spec.l2, init=coef_kq, snapshot_steps=snapshot_steps)
    Ct = data.C_test
    chat = {"kq": predict(coef_kq, data.X_test), "spo": predict(res.coef, data.X_test), "orakel": data.M_test}
    regret = {k: normalized_regret(grid, Ct, v) for k, v in chat.items()}
    mse = {k: mse_per_arc(Ct, chat[k]) for k in ("kq", "spo")}
    curve = {s: normalized_regret(grid, Ct, predict(c, data.X_test)) for s, c in res.snapshots.items()}
    _, zstar = grid.solve(Ct)
    case = {k: spo_loss(grid, Ct, v) / zstar for k, v in chat.items()} if keep_cases else {}
    coef_g, curve_g, losses_g, regret_g = {}, {}, {}, {}
    want = tuple(sorted(set(snapshot_steps) | {spec.steps}))
    for T in spec.temperatures:
        rg = train_gibbs(paths, data.X_train, data.C_train, T, spec.steps, lr, init=coef_kq, snapshot_steps=want)
        coef_g[T] = rg.snapshots[spec.steps]
        losses_g[T] = rg.losses
        curve_g[T] = {s: normalized_regret(grid, Ct, predict(c, data.X_test)) for s, c in rg.snapshots.items() if s in snapshot_steps or s == spec.steps}
        regret_g[T] = curve_g[T][spec.steps]
    return WorldRun(
        spec, grid, data, coef_kq, res.coef, res.best_step, res.losses, res.snapshots, regret, mse, curve,
        spo_plus_loss_mean(grid, data.X_train, data.C_train, coef_kq), spo_plus_loss_mean(grid, data.X_train, data.C_train, res.coef), case,
        coef_g, curve_g, losses_g, regret_g,
    )


def gibbs_sweep(spec: RunSpec, temperatures) -> dict:
    """Regret (Anteil, 0,05 heißt 5 %) von KQ und des geglätteten Verfahrens (Parameter nach dem letzten Schritt) für viele Temperaturen in einer Welt; kein SPO+.

    Rückgabe: {"kq": ..., "orakel": ..., "gibbs": {T: Regret}}. Für Experimente zum Kostenmaßstab (`spec.scale`)."""
    grid = Grid(spec.h)
    data = make_world_data(spec, grid)
    paths = grid.paths()
    lr = spec.lr * spec.scale
    coef_kq = fit_least_squares(data.X_train, data.C_train)
    out = {"kq": normalized_regret(grid, data.C_test, predict(coef_kq, data.X_test)), "orakel": normalized_regret(grid, data.C_test, data.M_test), "gibbs": {}}
    for T in temperatures:
        r = train_gibbs(paths, data.X_train, data.C_train, T, spec.steps, lr, init=coef_kq, snapshot_steps=(spec.steps,))
        out["gibbs"][float(T)] = normalized_regret(grid, data.C_test, predict(r.snapshots[spec.steps], data.X_test))
    return out


def unchanged_shares(grid: Grid, Chat: np.ndarray, C: np.ndarray, deltas, seed: int) -> dict:
    """Anteil der Testfälle, in denen sich der gewählte Weg nicht ändert, wenn jeder Eintrag der Vorhersage um den Bruchteil delta * u verändert wird (u gleichverteilt in [-1, 1],
    je Eintrag eine Zufallszahl, dieselbe Richtung für alle delta; nur die Größe wächst).

    Rückgabe: {delta: {"weg": Anteil mit gleichem Weg, "regret": Anteil mit gleichem Regret}}. Gleicher Weg impliziert gleichen Regret; gleicher Regret kann auch bei anderem Weg gleicher Kosten gelten
    (nur bei Gleichständen der wahren Kosten, praktisch nie). Der Regret ist die Größe, an der ein Differenzengradient hängt."""
    Chat = np.atleast_2d(np.asarray(Chat, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n, d = Chat.shape
    u = 2.0 * SplitMix64(derive_seed(seed, TAG_PERTURB)).uniform(n * d).reshape(n, d) - 1.0
    W0, _ = grid.solve(Chat)
    _, zstar = grid.solve(C)
    reg0 = np.sum(C * W0, axis=1) - zstar
    out = {}
    for delta in deltas:
        W1, _ = grid.solve(Chat * (1.0 + float(delta) * u))
        reg1 = np.sum(C * W1, axis=1) - zstar
        out[float(delta)] = {"weg": float(np.mean(np.all(W1 == W0, axis=1))), "regret": float(np.mean(np.abs(reg1 - reg0) <= 1e-12 * np.maximum(1.0, zstar)))}
    return out


def cell_result(run: WorldRun) -> dict:
    """Die Zahlen eines Laufs in Prozent (Regret); ohne die großen Felder."""
    out = {
        "seed": run.spec.seed,
        "regret_orakel": 100.0 * run.regret["orakel"],
        "regret_kq": 100.0 * run.regret["kq"],
        "regret_spo": 100.0 * run.regret["spo"],
        "mse_kq": run.mse["kq"],
        "mse_spo": run.mse["spo"],
    }
    for T, r in run.regret_gibbs.items():
        out[f"regret_gibbs_{T:g}"] = 100.0 * r
    return out


def run_seeds(spec: RunSpec, seeds) -> list[dict]:
    """Läuft über feste Zufalls-Seeds; je Seed ein Wörterbuch aus `cell_result`."""
    return [cell_result(run_world(replace(spec, seed=int(s)), snapshot_steps=(), keep_cases=False)) for s in seeds]


def summarize(values) -> dict:
    """Median, Minimum, Maximum einer Liste."""
    a = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def count_better(rows: list[dict], better: str = "regret_spo", than: str = "regret_kq") -> int:
    """In wie vielen Seeds ist `better` kleiner als `than` (streng)?"""
    return int(sum(r[better] < r[than] for r in rows))
