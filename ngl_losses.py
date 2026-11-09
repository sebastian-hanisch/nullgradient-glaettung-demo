"""Verluste für die Gitter-Welt: SPO-Verlust (Regret), SPO+ und sein Teilgradient, Quadratfehler (keine Streamlit-Abhängigkeit, nur numpy).

Alle Funktionen nehmen einen `solver` mit `solver.solve(C) -> (W, z)` (W: gewählte Wege als 0/1-Matrix, z: ihre Kosten unter C), also `ngl_grid.Grid`, `PathSet` oder
jeden anderen Löser derselben Form. Matrizen haben je Zeile einen Fall; Rückgabe je Fall, gemittelt wird beim Aufrufer.

Bezeichnungen (Zielfunktion c^T w, Minimierung): c wahre Kosten, ch vorhergesagte Kosten, w(ch) der unter ch gewählte Weg, w*(c) = w(c) der beste Weg, z*(c) = c^T w*(c).

    SPO-Verlust     l_SPO(ch, c)  = c^T w(ch) - z*(c)                                     (Regret des gewählten Weges, >= 0, stückweise konstant in ch)
    SPO+-Verlust    l_SPO+(ch, c) = max_w (c - 2 ch)^T w + 2 ch^T w*(c) - z*(c)            (konvex in ch, >= l_SPO)
                                  = -z(2 ch - c) + 2 ch^T w*(c) - z*(c)
    Teilgradient    2 (w*(c) - w(2 ch - c))                                               (nach ch)

Herkunft des SPO+-Verlustes: Elmachtoub und Grigas (Management Science 68(1), 2022), nur Abstract-Ebene gelesen; die Formel oben ist hier selbst nachgerechnet
(Brute-Force-Maximum über alle 70 Wege und Differenzenquotient in den Tests), die Satzbedingungen ihrer Konsistenzaussage sind nicht im Primärtext geprüft.

Kopie aus `spo-kuerzester-weg-demo/skw_losses.py` (dort Modul `skw_*`), nur umbenannt (`skw_` zu `ngl_`); die Rechnung ist unverändert, die Tests in `tests/test_losses.py` prüfen sie hier erneut.
"""

import numpy as np


def squared_error(C: np.ndarray, Chat: np.ndarray) -> np.ndarray:
    """Quadratfehler je Fall: Summe über die Bögen von (c - ch)^2."""
    return np.sum((np.asarray(C, float) - np.asarray(Chat, float)) ** 2, axis=1)


def mse_per_arc(C: np.ndarray, Chat: np.ndarray) -> float:
    """Mittlerer quadratischer Fehler je Bogen und Fall (Mittel über alle Einträge)."""
    return float(np.mean((np.asarray(C, float) - np.asarray(Chat, float)) ** 2))


def spo_loss(solver, C: np.ndarray, Chat: np.ndarray) -> np.ndarray:
    """SPO-Verlust je Fall: wahre Kosten des unter ch gewählten Weges minus die besten wahren Kosten."""
    C = np.atleast_2d(np.asarray(C, float))
    W, _ = solver.solve(Chat)
    _, zstar = solver.solve(C)
    return np.sum(C * W, axis=1) - zstar


def normalized_regret(solver, C: np.ndarray, Chat: np.ndarray) -> float:
    """Normierter Regret: Summe der SPO-Verluste geteilt durch die Summe der besten Kosten (0,05 heißt 5 %)."""
    C = np.atleast_2d(np.asarray(C, float))
    _, zstar = solver.solve(C)
    return float(np.sum(spo_loss(solver, C, Chat)) / np.sum(zstar))


def spo_plus_parts(solver, C: np.ndarray, Chat: np.ndarray):
    """SPO+-Verlust je Fall und Teilgradient nach ch (n x d) in einem Durchgang (zwei Löseraufrufe auf C und 2 ch - C)."""
    C = np.atleast_2d(np.asarray(C, float))
    Chat = np.atleast_2d(np.asarray(Chat, float))
    Wstar, zstar = solver.solve(C)
    W2, _ = solver.solve(2.0 * Chat - C)  # der Weg, der (c - 2 ch)^T w maximiert, minimiert (2 ch - c)^T w
    loss = np.sum((C - 2.0 * Chat) * W2, axis=1) + 2.0 * np.sum(Chat * Wstar, axis=1) - zstar
    return loss, 2.0 * (Wstar - W2)


def spo_plus_loss(solver, C: np.ndarray, Chat: np.ndarray) -> np.ndarray:
    """SPO+-Verlust je Fall."""
    return spo_plus_parts(solver, C, Chat)[0]


def spo_plus_subgradient(solver, C: np.ndarray, Chat: np.ndarray) -> np.ndarray:
    """Ein Teilgradient des SPO+-Verlustes nach ch je Fall (n x d): 2 (w*(c) - w(2 ch - c))."""
    return spo_plus_parts(solver, C, Chat)[1]
