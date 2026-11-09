"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngl_grid import Grid, two_paths  # noqa: E402


@pytest.fixture(scope="session")
def grid5():
    return Grid(5)


@pytest.fixture(scope="session")
def grid3():
    return Grid(3)


@pytest.fixture(scope="session")
def grid2():
    return Grid(2)


@pytest.fixture
def two():
    """Das Handbeispiel: zwei Wege mit je einem Bogen. Wahre Kosten c = (3, 5), Vorhersage ch = (4, 3)."""
    return two_paths()


@pytest.fixture
def hand_c():
    return np.array([[3.0, 5.0]])


@pytest.fixture
def hand_chat():
    return np.array([[4.0, 3.0]])


def independent_paths(grid):
    """Alle Wege des Gitters über eine ANDERE Konstruktion als `Grid.paths`: Rekursion über Knotenpaare, Bogennummer über das Wörterbuch (Anfang, Ende) -> Nummer.

    Rückgabe: Liste von (Zugfolge als Tuple aus 'R'/'D', Bogennummern) in der Reihenfolge der Rekursion (rechts vor unten), also schon lexikographisch."""
    h = grid.h
    index = {edge: k for k, edge in enumerate(grid.edges)}
    out = []

    def walk(i, j, moves, arcs):
        if (i, j) == (h - 1, h - 1):
            out.append((tuple(moves), tuple(arcs)))
            return
        v = i * h + j
        if j + 1 < h:
            walk(i, j + 1, moves + ["R"], arcs + [index[(v, v + 1)]])
        if i + 1 < h:
            walk(i + 1, j, moves + ["D"], arcs + [index[(v, v + h)]])

    walk(0, 0, [], [])
    return out


def independent_solve(grid, C):
    """Löser durch Aufzählung aller Wege (rekursiv erzeugt, Summe der Bogenkosten je Weg, erster kleinster Wert). Rückgabe (W, z)."""
    paths = independent_paths(grid)
    C = np.atleast_2d(C)
    W = np.zeros_like(C, dtype=float)
    z = np.zeros(len(C))
    for r, c in enumerate(C):
        costs = [sum(c[k] for k in arcs) for _, arcs in paths]
        best = min(range(len(paths)), key=lambda q: costs[q])  # min liefert beim Gleichstand den ersten
        z[r] = costs[best]
        for k in paths[best][1]:
            W[r, k] = 1.0
    return W, z
