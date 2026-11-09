"""Gibbs-Glättung des Lösers: aus dem harten Weg eine Verteilung über alle Wege machen (keine Streamlit-Abhängigkeit, nur numpy).

Der Löser wählt zu den vorhergesagten Kosten `ch` den billigsten Weg: eine Treppe in `ch`, überall null Gradient bis auf die Sprünge. Die Glättung ersetzt ihn durch die
Gibbs-Verteilung über alle Wege w mit der Temperatur T > 0 (Boltzmann-Gewichte):

    pi_T(w | ch) = exp(-ch^T w / T) / Z,        Z = Summe über alle Wege exp(-ch^T w' / T).

Statt "der gewählte Weg kostet c^T w(ch)" gilt dann "die erwarteten Kosten sind E[c^T w] unter pi_T". Das ist in `ch` überall glatt. Ableitung (Kettenregel, `softmax`):

    d E[c^T w] / d ch = -(1/T) ( E[(c^T w) w] - E[c^T w] E[w] )        (Kovarianz von Wegkosten und Weg, geteilt durch -T).

Zwei Rechenwege (der zweite ist das Orakel der Tests):
  1. Aufzählung: alle Wege als 0/1-Matrix (im 5 x 5 Gitter 70 Zeilen), Gewichte über log-sum-exp. Das ist, was die App rechnet.
  2. Summe-Produkt auf dem azyklischen Gitter (Vorwärts-Rückwärts): Kantenrandverteilungen und log Z ohne Wegliste, in Zeit proportional zur Zahl der Bögen.
     `edge_marginals` liefert E[w] je Bogen; die erwarteten Kosten sind dann c^T E[w] (linear in den Randverteilungen).

Grenze: Aufzählung wächst mit der Zahl der Wege (Gitter h x h: 2, 6, 20, 70, 252, 924, ...). Die Summe-Produkt-Rechnung braucht die azyklische Struktur; für ein allgemeines
Polytop (Wege mit Nebenbedingungen) gibt es sie nicht.

Eigene Herleitung, nicht aus einer Quelle übernommen; die Ableitung ist in `tests/test_gibbs.py` gegen zentrale Differenzen über die unabhängige Vorwärts-Rückwärts-Rechnung geprüft.
"""

import numpy as np

from ngl_training import add_intercept


def _check_temperature(T: float) -> float:
    T = float(T)
    if not T > 0.0 or not np.isfinite(T):
        raise ValueError("die Temperatur T muss positiv und endlich sein")
    return T


def log_weights(Chat: np.ndarray, paths: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Normierte Log-Gewichte je Weg (n x Wege) und log Z (n,), stabil über log-sum-exp (Abziehen des Maximums verhindert Überlauf bei kleinem T)."""
    T = _check_temperature(T)
    Chat = np.atleast_2d(np.asarray(Chat, dtype=float))
    s = -(Chat @ np.asarray(paths, dtype=float).T) / T
    m = s.max(axis=1, keepdims=True)
    log_z = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
    return s - log_z[:, None], log_z


def gibbs_weights(Chat: np.ndarray, paths: np.ndarray, T: float) -> np.ndarray:
    """Gewichte pi_T (n x Wege), jede Zeile summiert zu 1 (gleiche Rechnung wie `log_weights`, mit einem Exponentieren weniger: das Training ruft sie in jedem Schritt)."""
    T = _check_temperature(T)
    Chat = np.atleast_2d(np.asarray(Chat, dtype=float))
    s = -(Chat @ np.asarray(paths, dtype=float).T) / T
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def expected_cost(C: np.ndarray, Chat: np.ndarray, paths: np.ndarray, T: float) -> np.ndarray:
    """Erwartete wahre Kosten E[c^T w] je Fall unter der Gibbs-Verteilung zu `Chat`."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    pi = gibbs_weights(Chat, paths, T)
    return np.sum(pi * (C @ np.asarray(paths, dtype=float).T), axis=1)


def smoothed_regret(C: np.ndarray, Chat: np.ndarray, paths: np.ndarray, T: float) -> np.ndarray:
    """Geglätteter Regret je Fall: erwartete wahre Kosten minus die besten wahren Kosten (>= 0, für T -> 0 der SPO-Verlust des harten Weges)."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    return expected_cost(C, Chat, paths, T) - (C @ np.asarray(paths, dtype=float).T).min(axis=1)


def gibbs_parts(C: np.ndarray, Chat: np.ndarray, paths: np.ndarray, T: float, cost_paths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Erwartete wahre Kosten je Fall (n,) und ihr Gradient nach `Chat` (n x d) in einem Durchgang.

    `cost_paths` = C @ paths.T (n x Wege), wenn vorab berechnet (die wahren Kosten ändern sich im Training nicht)."""
    T = _check_temperature(T)
    paths = np.asarray(paths, dtype=float)
    cp = np.atleast_2d(np.asarray(C, dtype=float)) @ paths.T if cost_paths is None else cost_paths
    pi = gibbs_weights(Chat, paths, T)
    ecost = np.sum(pi * cp, axis=1)
    ew = pi @ paths
    ecw = (pi * cp) @ paths
    grad = -(ecw - ecost[:, None] * ew) / T
    return ecost, grad


def gibbs_objective(paths: np.ndarray, X: np.ndarray, C: np.ndarray, T: float):
    """Zielfunktion des geglätteten Trainings als `loss_and_grad(coef)` für `ngl_training.adam_minimize`: mittlerer geglätteter Regret und sein Gradient nach coef."""
    T = _check_temperature(T)
    paths = np.asarray(paths, dtype=float)
    Xa = add_intercept(X)
    C = np.asarray(C, dtype=float)
    n = len(Xa)
    cp = C @ paths.T
    best = cp.min(axis=1)

    def loss_and_grad(coef):
        ecost, G = gibbs_parts(C, Xa @ coef, paths, T, cost_paths=cp)
        return float(np.mean(ecost - best)), Xa.T @ G / n

    return loss_and_grad


def edge_marginals(grid, Chat: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Zweite Rechenweise: Summe-Produkt (Vorwärts-Rückwärts) auf dem azyklischen Gitter, ohne Wegliste.

    Rückgabe: Randverteilung je Bogen E[w_e] (n x d) und log Z (n,). Bögen gehen von kleiner zu größer Knotennummer, die Knotennummern sind also eine topologische Ordnung.
    Vorwärts: alpha[v] = log Summe über Teilwege vom Start nach v; rückwärts: beta[v] = log Summe über Teilwege von v zum Ziel; E[w_e] = exp(alpha[tail] - ch_e/T + beta[head] - log Z)."""
    T = _check_temperature(T)
    Chat = np.atleast_2d(np.asarray(Chat, dtype=float))
    n = Chat.shape[0]
    a = -Chat / T  # Log-Gewicht je Bogen
    alpha = np.full((n, grid.nodes), -np.inf)
    alpha[:, 0] = 0.0
    for k in sorted(range(grid.d), key=lambda e: grid.tail[e]):  # nach dem Anfangsknoten: alle Bögen in v sind vorher abgearbeitet
        t, h = grid.tail[k], grid.head[k]
        alpha[:, h] = np.logaddexp(alpha[:, h], alpha[:, t] + a[:, k])
    beta = np.full((n, grid.nodes), -np.inf)
    beta[:, grid.nodes - 1] = 0.0
    for k in sorted(range(grid.d), key=lambda e: -grid.head[e]):  # nach dem Endknoten, absteigend
        t, h = grid.tail[k], grid.head[k]
        beta[:, t] = np.logaddexp(beta[:, t], a[:, k] + beta[:, h])
    log_z = alpha[:, grid.nodes - 1]
    marg = np.exp(alpha[:, grid.tail] + a + beta[:, grid.head] - log_z[:, None])
    return marg, log_z


def expected_cost_fb(grid, C: np.ndarray, Chat: np.ndarray, T: float) -> np.ndarray:
    """Erwartete Kosten über die Summe-Produkt-Rechnung: c^T E[w] (ohne Wegliste)."""
    marg, _ = edge_marginals(grid, Chat, T)
    return np.sum(np.atleast_2d(np.asarray(C, dtype=float)) * marg, axis=1)
