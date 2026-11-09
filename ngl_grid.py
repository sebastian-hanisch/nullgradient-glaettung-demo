"""Gitter und Löser: kürzeste Wege von links oben nach rechts unten im h x h Gitter (keine Streamlit-Abhängigkeit, nur numpy).

Das Gitter ist azyklisch (es geht nur nach rechts oder nach unten), der "Löser" ist deshalb eine Rückwärtsrechnung über die Knoten in umgekehrter Reihenfolge,
vektorisiert über viele Kostenvektoren zugleich. Das ist der Löser in der Schleife: jede Trainingsrunde der SPO+-Verfahren ruft `solve` auf allen Fällen auf.

Aufbau (5 x 5): 25 Knoten, d = 2 h (h - 1) = 40 Bögen, jeder Weg hat 2 (h - 1) = 8 Bögen, es gibt binomial(8, 4) = 70 Wege.
Knoten (i, j) hat die Nummer i * h + j (Zeile i von oben, Spalte j von links); Bögen stehen in der Reihenfolge der Knoten, je Knoten erst "rechts", dann "unten".

Gleichstandsregel: bei gleich langen Wegen gewinnt am Knoten "rechts" vor "unten". Das ist genau der lexikographisch kleinste optimale Weg bei der
Aufzählung mit R < D (gilt mathematisch; bei Gleitkommasummen nur, wenn die Gleichstände exakt sind, z. B. ganzzahlige Kosten).

Kopie aus `spo-kuerzester-weg-demo/skw_grid.py` (dort Modul `skw_*`), nur umbenannt (`skw_` zu `ngl_`); die Rechnung ist unverändert, die Tests in `tests/test_grid.py` prüfen sie hier erneut.
"""

import itertools
from dataclasses import dataclass

import numpy as np


class Grid:
    """h x h Gitter mit Bögen nach rechts und unten; `solve` rechnet den kürzesten Weg für jede Zeile einer Kostenmatrix."""

    def __init__(self, h: int = 5):
        if h < 2:
            raise ValueError("das Gitter braucht mindestens 2 x 2 Knoten")
        self.h = h
        self.nodes = h * h
        edges = []
        # je Knoten: Bogen nach rechts, dann Bogen nach unten (die Reihenfolge ist Teil der Gleichstandsregel und der Aufzählung)
        right = {}
        down = {}
        for i in range(h):
            for j in range(h):
                v = i * h + j
                if j + 1 < h:
                    right[v] = len(edges)
                    edges.append((v, v + 1))
                if i + 1 < h:
                    down[v] = len(edges)
                    edges.append((v, v + h))
        self.edges = edges
        self.d = len(edges)
        self.tail = np.array([a for a, _ in edges], dtype=int)
        self.head = np.array([b for _, b in edges], dtype=int)
        self._right = right
        self._down = down
        # Bogennummer je Knoten (-1: es gibt diesen Bogen nicht), vorab als Tabelle für die Wegrekonstruktion
        self._r_idx = np.array([right.get(v, -1) for v in range(self.nodes)])
        self._d_idx = np.array([down.get(v, -1) for v in range(self.nodes)])
        self.steps = 2 * (h - 1)  # Bögen je Weg

    def node_xy(self, v: int) -> tuple[float, float]:
        """Zeichenposition (x nach rechts, y nach unten) eines Knotens."""
        return float(v % self.h), float(v // self.h)

    def solve(self, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kürzester Weg je Zeile von C (n x d). Rückgabe: Wegmatrix W (n x d, Einträge 0/1) und Wegkosten z (n,).

        Rückwärtsrechnung: dist[v] = kleinste Kosten von v zum Ziel. Gleichstand: rechts vor unten (<=)."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        n = C.shape[0]
        if C.shape[1] != self.d:
            raise ValueError(f"erwartet {self.d} Bögen je Zeile, bekommen {C.shape[1]}")
        dist = np.zeros((n, self.nodes))
        go_right = np.zeros((n, self.nodes), dtype=bool)
        for v in range(self.nodes - 2, -1, -1):
            r = self._right.get(v)
            dn = self._down.get(v)
            if r is not None and dn is not None:
                via_r = C[:, r] + dist[:, v + 1]
                via_d = C[:, dn] + dist[:, v + self.h]
                take_r = via_r <= via_d
                dist[:, v] = np.where(take_r, via_r, via_d)
                go_right[:, v] = take_r
            elif r is not None:
                dist[:, v] = C[:, r] + dist[:, v + 1]
                go_right[:, v] = True
            else:
                dist[:, v] = C[:, dn] + dist[:, v + self.h]
        W = np.zeros((n, self.d))
        rows = np.arange(n)
        node = np.zeros(n, dtype=int)
        for _ in range(self.steps):
            k = np.where(go_right[rows, node], self._r_idx[node], self._d_idx[node])
            W[rows, k] = 1.0
            node = self.head[k]
        return W, dist[:, 0]

    def paths(self) -> np.ndarray:
        """Alle Wege als 0/1-Matrix (Wege x Bögen), lexikographisch nach den Zügen (R = 0 vor D = 1) geordnet: die zweite, unabhängige Rechnung (Aufzählung)."""
        out = []
        for downs in itertools.combinations(range(self.steps), self.h - 1):
            moves = [0] * self.steps  # 0 = rechts, 1 = unten
            for m in downs:
                moves[m] = 1
            i = j = 0
            w = np.zeros(self.d)
            for m in moves:
                v = i * self.h + j
                if m == 0:
                    w[self._right[v]] = 1.0
                    j += 1
                else:
                    w[self._down[v]] = 1.0
                    i += 1
            out.append((tuple(moves), w))
        out.sort(key=lambda t: t[0])
        return np.array([w for _, w in out])


@dataclass(frozen=True)
class PathSet:
    """Löser über eine ausdrücklich gegebene Wegmenge (Wege x Bögen): kleinste Kosten, bei Gleichstand der erste Weg der Liste.

    Dient für das Handbeispiel mit zwei Wegen und als Aufzählungs-Löser: `solve` hat dieselbe Form wie `Grid.solve`."""

    W_all: np.ndarray

    @property
    def d(self) -> int:
        return self.W_all.shape[1]

    def solve(self, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C = np.atleast_2d(np.asarray(C, dtype=float))
        values = C @ self.W_all.T
        best = np.argmin(values, axis=1)  # erster kleinster Wert: feste Gleichstandsregel
        return self.W_all[best], values[np.arange(C.shape[0]), best]


def two_paths() -> PathSet:
    """Das Handbeispiel: zwei Wege mit je einem Bogen (Weg 1 = Bogen 1, Weg 2 = Bogen 2)."""
    return PathSet(np.eye(2))
