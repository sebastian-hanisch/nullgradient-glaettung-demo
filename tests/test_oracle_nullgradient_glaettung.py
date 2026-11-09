"""Orakel-Test: die Gibbs-Gewichte über Aufzählung der 70 Wege (ngl_gibbs, Rechenweg der App) gegen drei unabhängige Rechenwege.

1. Summe-Produkt (Vorwärts-Rückwärts) auf dem azyklischen Gitter, ohne Wegliste: Kantenrandverteilungen und log Z (`ngl_gibbs.edge_marginals`).
2. Eine rekursiv aufgebaute Wegliste in Python-Zahlen (`conftest.independent_paths`, `math.exp`, `math.fsum`), also weder numpy noch `Grid.paths`.
3. scipy `logsumexp` für log Z (nur wenn scipy im Test-venv steht).

Dazu: der Löser (`Grid.solve`) gegen vollständige Aufzählung, der Gradient der erwarteten Kosten gegen zentrale Differenzen der Summe-Produkt-Rechnung (zweiter Rechenweg für die
Zielfunktion, damit weder Wert noch Ableitung aus derselben Aufzählung stammen), und die angezeigten Kennzahlen des Handbeispiels."""

import math

import numpy as np
import pytest

import ngl_gibbs as G
from conftest import independent_paths, independent_solve


def _brute_force_weights(grid, chat_row, T):
    """Gewichte je Weg in der Reihenfolge der rekursiven Wegliste, in reinen Python-Zahlen (log-sum-exp von Hand)."""
    paths = independent_paths(grid)
    scores = [-sum(chat_row[k] for k in arcs) / T for _, arcs in paths]
    m = max(scores)
    z = math.fsum(math.exp(s - m) for s in scores)
    return [math.exp(s - m) / z for s in scores], m + math.log(z), paths


def test_weights_against_python_brute_force_on_random_cases(grid5):
    rng = np.random.default_rng(0)
    P = grid5.paths()
    for case in range(25):
        chat = rng.uniform(0.5, 4.0, 40)
        T = float(rng.choice([0.05, 0.3, 1.0, 5.0]))
        w, logz, paths = _brute_force_weights(grid5, chat, T)
        assert len(w) == 70
        mine = G.gibbs_weights(chat[None, :], P, T)[0]
        assert np.abs(mine - np.array(w)).max() < 1e-12, case
        assert G.log_weights(chat[None, :], P, T)[1][0] == pytest.approx(logz, abs=1e-10)


def test_marginals_and_log_partition_by_forward_backward_match_the_enumeration(grid5):
    rng = np.random.default_rng(1)
    P = grid5.paths()
    for _ in range(25):
        Chat = rng.uniform(0.5, 4.0, (4, 40))
        T = float(rng.choice([0.05, 0.3, 1.0, 5.0]))
        marg, logz_fb = G.edge_marginals(grid5, Chat, T)
        pi = G.gibbs_weights(Chat, P, T)
        assert np.abs(marg - pi @ P).max() < 1e-12
        assert np.abs(logz_fb - G.log_weights(Chat, P, T)[1]).max() < 1e-10
        assert np.allclose(marg.sum(axis=1), 8.0)         # jeder Weg hat 8 Bögen: die Randverteilungen summieren zu 8


def test_edge_marginals_against_brute_force_in_python_numbers(grid5):
    rng = np.random.default_rng(2)
    for _ in range(5):
        chat = rng.uniform(0.5, 4.0, 40)
        w, _, paths = _brute_force_weights(grid5, chat, 0.7)
        expected = np.zeros(40)
        for weight, (_, arcs) in zip(w, paths):
            for k in arcs:
                expected[k] += weight
        marg, _ = G.edge_marginals(grid5, chat[None, :], 0.7)
        assert np.abs(marg[0] - expected).max() < 1e-12


def test_log_partition_against_scipy_logsumexp(grid5):
    special = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(3)
    P = grid5.paths()
    Chat = rng.uniform(0.5, 4.0, (10, 40))
    for T in (0.01, 1.0, 20.0):
        ref = special.logsumexp(-(Chat @ P.T) / T, axis=1)
        assert np.abs(G.log_weights(Chat, P, T)[1] - ref).max() < 1e-10
        assert np.abs(G.edge_marginals(grid5, Chat, T)[1] - ref).max() < 1e-10


def test_forward_backward_survives_small_temperature(grid5):
    Chat = np.random.default_rng(4).uniform(1, 3, (5, 40))
    marg, logz = G.edge_marginals(grid5, Chat, 1e-6)
    assert np.all(np.isfinite(marg)) and np.all(np.isfinite(logz))
    W, _ = grid5.solve(Chat)
    assert np.abs(marg - W).max() < 1e-6                  # eindeutiges Minimum: die Randverteilung ist der harte Weg (log Z ist hier von der Größe 1e7, die Stellen davon gehen in der Differenz verloren)


def test_solver_against_full_enumeration_on_300_cases(grid5):
    """Der harte Löser (Rückwärtsrechnung) gegen die rekursiv erzeugte Wegliste: dieselben Kosten, bei stetigen Kosten derselbe Weg."""
    rng = np.random.default_rng(5)
    C = rng.uniform(0.5, 4.0, (300, 40))
    W, z = grid5.solve(C)
    W_ref, z_ref = independent_solve(grid5, C)
    assert np.abs(z - z_ref).max() < 1e-12
    assert np.array_equal(W, W_ref)


def test_gradient_of_the_expected_cost_against_central_differences_of_forward_backward(grid5):
    """Wert aus der Summe-Produkt-Rechnung (c^T E[w]), Ableitung aus der Aufzählung (Kovarianz): sie müssen sich entsprechen."""
    rng = np.random.default_rng(6)
    P = grid5.paths()
    C, Chat = rng.uniform(1, 3, (3, 40)), rng.uniform(1, 3, (3, 40))
    for T in (0.1, 0.5, 2.0):
        _, g = G.gibbs_parts(C, Chat, P, T)
        h = 1e-6
        for k in (2, 13, 26, 38):
            plus, minus = Chat.copy(), Chat.copy()
            plus[:, k] += h
            minus[:, k] -= h
            fd = (G.expected_cost_fb(grid5, C, plus, T) - G.expected_cost_fb(grid5, C, minus, T)) / (2 * h)
            assert np.abs(fd - g[:, k]).max() < 1e-6, (T, k)


def test_expected_cost_by_enumeration_equals_cost_of_marginals(grid5):
    rng = np.random.default_rng(7)
    P = grid5.paths()
    C, Chat = rng.uniform(1, 3, (8, 40)), rng.uniform(1, 3, (8, 40))
    for T in (0.2, 1.0):
        assert np.abs(G.expected_cost(C, Chat, P, T) - G.expected_cost_fb(grid5, C, Chat, T)).max() < 1e-12


def test_forward_backward_on_the_3_by_3_grid_by_hand(grid3):
    """3 x 3 Gitter: 6 Wege, 12 Bögen; alle Kosten gleich (T = 1): jeder Weg hat das Gewicht 1/6, die Randverteilung ist die Zahl der Wege durch den Bogen geteilt durch 6.

    Von Hand: der Bogen vom Start nach rechts (Knoten 0 nach 1) liegt auf den 3 Wegen, die mit R beginnen (die Züge R, D, D folgen in 3 Anordnungen): Randverteilung 3/6; der Bogen
    von Knoten 1 nach 2 liegt nur auf dem Weg R R D D: 1/6. Danach alle Bögen gegen die Zählung der Wege durch den Bogen."""
    marg, logz = G.edge_marginals(grid3, np.zeros((1, grid3.d)), 1.0)
    assert logz[0] == pytest.approx(math.log(6.0))
    assert marg[0, 0] == pytest.approx(3 / 6)          # Bogen (0 -> 1)
    idx = grid3.edges.index((1, 2))
    assert marg[0, idx] == pytest.approx(1 / 6)       # genau ein Weg: R R D D
    for e, (a, b) in enumerate(grid3.edges):
        count = sum(1 for _, arcs in independent_paths(grid3) if e in arcs)
        assert marg[0, e] == pytest.approx(count / 6)
