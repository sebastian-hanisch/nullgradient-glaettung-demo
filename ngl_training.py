"""Training für die Gitter-Welt: Kleinste Quadrate (KQ) und SPO+ mit Adam, der Löser läuft in jeder Runde (keine Streamlit-Abhängigkeit, nur numpy).

Modell: lineares Modell mit Achsenabschnitt, ch = [1, x] @ coef, coef hat (p + 1) x d Einträge (5 Merkmale, 40 Bögen: 240 Parameter).
SPO+-Training: Start bei der KQ-Lösung, Adam mit fester Lernrate (Standard 0,05), Vollbatch, 300 Schritte (Standard); in jeder Runde ein Löseraufruf
auf allen Trainingsfällen für w(2 ch - c) (der Weg w*(c) wird einmal vorab gelöst).

Zurückgegeben wird der Parameter mit dem kleinsten SPO+-Trainingsverlust unter allen besuchten (Auswahl nur auf Trainingsdaten; der Start, also KQ, gehört dazu: der
Trainingsverlust des Ergebnisses ist nie größer als der von KQ). Die Verlaufsliste enthält den Trainingsverlust vor jedem Schritt.

Kopie aus `spo-kuerzester-weg-demo/skw_training.py` (dort Modul `skw_*`), nur umbenannt (`skw_` zu `ngl_`); die Rechnung ist unverändert, die Tests in `tests/test_training.py` prüfen sie hier erneut.
"""

from dataclasses import dataclass, field

import numpy as np

from ngl_losses import spo_plus_parts

ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), np.asarray(X, float)])


def fit_least_squares(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate: coef = argmin Summe (c - [1, x] coef)^2, über `numpy.linalg.lstsq` (kleinste Norm bei Rangmangel)."""
    coef, *_ = np.linalg.lstsq(add_intercept(X), np.asarray(C, float), rcond=None)
    return coef


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ coef


@dataclass
class TrainResult:
    coef: np.ndarray                     # der gewählte Parameter (kleinster Trainingsverlust unter allen besuchten)
    best_step: int                       # nach wie vielen Adam-Schritten dieser Parameter erreicht war (0 = Start)
    losses: np.ndarray                   # Trainingsverlust vor Schritt 1 bis vor Schritt `steps` (Länge steps)
    snapshots: dict = field(default_factory=dict)  # Schritt -> Parameter nach so vielen Schritten (nur die gewünschten)


def adam_minimize(loss_and_grad, coef0: np.ndarray, steps: int, lr: float = 0.05, snapshot_steps=()) -> TrainResult:
    """Vollbatch-Adam (Kingma und Ba) mit Verzerrungskorrektur. `loss_and_grad(coef) -> (Verlust, Gradient nach coef)`.

    Der Verlust gehört zum Parameter VOR dem Schritt; nach `steps` Schritten wird kein weiterer Verlust ausgewertet.
    `snapshot_steps`: Schritte, nach denen der Parameter festgehalten wird (0 = Start)."""
    coef = np.array(coef0, dtype=float)
    m = np.zeros_like(coef)
    v = np.zeros_like(coef)
    want = set(int(s) for s in snapshot_steps)
    snaps = {0: coef.copy()} if 0 in want else {}
    best_loss, best_coef, best_step = np.inf, coef.copy(), 0
    losses = np.zeros(steps)
    for t in range(1, steps + 1):
        loss, grad = loss_and_grad(coef)
        losses[t - 1] = loss
        if loss < best_loss:
            best_loss, best_coef, best_step = loss, coef.copy(), t - 1
        m = ADAM_B1 * m + (1 - ADAM_B1) * grad
        v = ADAM_B2 * v + (1 - ADAM_B2) * grad * grad
        coef = coef - lr * (m / (1 - ADAM_B1 ** t)) / (np.sqrt(v / (1 - ADAM_B2 ** t)) + ADAM_EPS)
        if t in want:
            snaps[t] = coef.copy()
    return TrainResult(best_coef, best_step, losses, snaps)


def spo_plus_objective(solver, X: np.ndarray, C: np.ndarray, l2: float = 0.0):
    """Zielfunktion des SPO+-Trainings als `loss_and_grad(coef)`: mittlerer SPO+-Verlust (plus l2/2 |coef|^2 im Gradienten, der Verlust bleibt ohne Strafe)."""
    Xa = add_intercept(X)
    C = np.asarray(C, float)
    n = len(Xa)
    Wstar, zstar = solver.solve(C)  # der beste Weg je Fall hängt nicht vom Modell ab: einmal lösen

    def loss_and_grad(coef):
        Chat = Xa @ coef
        W2, _ = solver.solve(2.0 * Chat - C)
        loss_i = np.sum((C - 2.0 * Chat) * W2, axis=1) + 2.0 * np.sum(Chat * Wstar, axis=1) - zstar
        grad = Xa.T @ (2.0 * (Wstar - W2)) / n + l2 * coef
        return float(np.mean(loss_i)), grad

    return loss_and_grad


def train_spo_plus(solver, X: np.ndarray, C: np.ndarray, steps: int = 300, lr: float = 0.05, l2: float = 0.0, init=None, snapshot_steps=()) -> TrainResult:
    """SPO+ mit Adam, Start bei KQ (oder bei `init`). Der Löser läuft in jeder Runde auf allen Trainingsfällen."""
    coef0 = fit_least_squares(X, C) if init is None else np.array(init, dtype=float)
    return adam_minimize(spo_plus_objective(solver, X, C, l2), coef0, steps, lr, snapshot_steps)


def subgradient_of_mean_loss(solver, X: np.ndarray, C: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Teilgradient des mittleren SPO+-Verlustes nach coef (Kettenregel: Xa^T mal Teilgradient nach ch, gemittelt); Hilfsfunktion für Tests."""
    Xa = add_intercept(X)
    _, G = spo_plus_parts(solver, C, Xa @ coef)
    return Xa.T @ G / len(Xa)


def spo_plus_loss_mean(solver, X: np.ndarray, C: np.ndarray, coef: np.ndarray) -> float:
    """Mittlerer SPO+-Verlust eines Parameters auf den Fällen (X, C)."""
    loss, _ = spo_plus_parts(solver, C, predict(coef, X))
    return float(np.mean(loss))
