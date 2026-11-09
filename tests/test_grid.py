"""Gitter und Löser (ngl_grid): Handinstanzen auf 2 x 2, 3 x 3 und 5 x 5, Zählungen, Gleichstandsregel, Invarianten."""

import itertools
import math

import numpy as np
import pytest

from ngl_grid import Grid, PathSet


def _follow(grid, w):
    """Prüft, dass die Bögen mit w = 1 einen zusammenhängenden Weg von Knoten 0 zu Knoten h*h-1 bilden (rechts oder unten)."""
    arcs = [grid.edges[k] for k in np.flatnonzero(w)]
    node = 0
    for _ in range(len(arcs)):
        nxt = [b for a, b in arcs if a == node]
        assert len(nxt) == 1, "an jedem Knoten genau ein ausgehender Bogen des Weges"
        node = nxt[0]
    return node == grid.h * grid.h - 1 and len(arcs) == grid.steps


def test_counts_of_the_5x5_grid(grid5):
    assert grid5.nodes == 25
    assert grid5.d == 40 == 2 * 5 * 4
    assert grid5.steps == 8
    P = grid5.paths()
    assert P.shape == (70, 40) and math.comb(8, 4) == 70
    assert set(P.sum(axis=1)) == {8.0}
    assert len({tuple(r) for r in P}) == 70


def test_two_by_two_grid_by_hand(grid2):
    # Bögen: 0: 0->1, 1: 0->2, 2: 1->3, 3: 2->3. Wege: rechts-unten = {0, 2}, unten-rechts = {1, 3}.
    assert grid2.edges == [(0, 1), (0, 2), (1, 3), (2, 3)]
    W, z = grid2.solve(np.array([[1.0, 5.0, 1.0, 1.0]]))      # rechts-unten kostet 2, unten-rechts 6
    assert W.tolist() == [[1, 0, 1, 0]] and z.tolist() == [2.0]
    W, z = grid2.solve(np.array([[4.0, 1.0, 5.0, 1.0]]))      # 9 gegen 2
    assert W.tolist() == [[0, 1, 0, 1]] and z.tolist() == [2.0]


def test_three_by_three_hand_instance(grid3):
    # Bögen: 0: 0->1 R, 1: 0->3 D, 2: 1->2 R, 3: 1->4 D, 4: 2->5 D, 5: 3->4 R, 6: 3->6 D, 7: 4->5 R, 8: 4->7 D, 9: 5->8 D, 10: 6->7 R, 11: 7->8 R
    assert grid3.edges == [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (4, 5), (4, 7), (5, 8), (6, 7), (7, 8)]
    c = np.ones(12)
    c[3] = 0.0   # 1 -> 4
    c[8] = 0.0   # 4 -> 7
    W, z = grid3.solve(c[None, :])
    assert z.tolist() == [2.0]                    # R, D, D, R: 1 + 0 + 0 + 1; jeder andere Weg nutzt höchstens einen der beiden Nullbögen und kostet mindestens 3
    assert np.flatnonzero(W[0]).tolist() == [0, 3, 8, 11]
    assert grid3.paths().shape == (6, 12)


def test_five_by_five_hand_instance_with_twenty_optimal_paths(grid5):
    """Erste Zeile und letzte Spalte kosten 3, alle anderen Bögen 1. Optimal (Kosten 8) ist jeder Weg, der zuerst nach unten und zuletzt nach rechts geht:
    von (1, 0) nach (4, 3) mit 3 Schritten rechts und 3 unten, also binomial(6, 3) = 20 Wege. Die Gleichstandsregel nimmt den lexikographisch kleinsten
    (R vor D): D R R R D D D R."""
    c = np.ones(40)
    for k, (a, b) in enumerate(grid5.edges):
        if a < 5 and b == a + 1:          # waagerechter Bogen der ersten Zeile
            c[k] = 3.0
        if a % 5 == 4 and b == a + 5:     # senkrechter Bogen der letzten Spalte
            c[k] = 3.0
    assert (c == 3.0).sum() == 8
    P = grid5.paths()
    costs = P @ c
    assert costs.min() == 8.0 and int((costs == 8.0).sum()) == math.comb(6, 3) == 20
    W, z = grid5.solve(c[None, :])
    assert z.tolist() == [8.0]
    moves = ["R" if grid5.edges[k][1] == grid5.edges[k][0] + 1 else "D" for k in _ordered_arcs(grid5, W[0])]
    assert "".join(moves) == "DRRRDDDR"


def _ordered_arcs(grid, w):
    arcs = {grid.edges[k][0]: k for k in np.flatnonzero(w)}
    node, out = 0, []
    while node != grid.nodes - 1:
        k = arcs[node]
        out.append(k)
        node = grid.edges[k][1]
    return out


def test_tie_rule_all_equal_costs_goes_right_first(grid5):
    W, z = grid5.solve(np.ones((1, 40)))
    assert z.tolist() == [8.0]
    assert "".join("R" if grid5.edges[k][1] == grid5.edges[k][0] + 1 else "D" for k in _ordered_arcs(grid5, W[0])) == "RRRRDDDD"


def test_every_chosen_path_is_a_connected_path(grid5):
    rng = np.random.default_rng(0)
    W, _ = grid5.solve(rng.uniform(0, 3, size=(50, 40)))
    assert all(_follow(grid5, w) for w in W)


def test_adding_a_constant_to_all_arcs_does_not_change_the_path(grid5):
    """Alle Wege haben genau 8 Bögen: Verschieben aller Kosten um eine Konstante verschiebt jeden Weg um 8 Konstanten und ändert die Wahl nicht."""
    rng = np.random.default_rng(1)
    C = rng.uniform(0.5, 4, size=(100, 40))
    W1, z1 = grid5.solve(C)
    W2, z2 = grid5.solve(C + 2.5)
    assert np.array_equal(W1, W2)
    assert np.allclose(z2, z1 + 8 * 2.5)


def test_scaling_costs_by_a_positive_factor_does_not_change_the_path(grid5):
    rng = np.random.default_rng(2)
    C = rng.uniform(0.5, 4, size=(100, 40))
    W1, z1 = grid5.solve(C)
    W2, z2 = grid5.solve(3.0 * C)
    assert np.array_equal(W1, W2) and np.allclose(z2, 3.0 * z1)


def test_batch_equals_row_by_row(grid5):
    rng = np.random.default_rng(3)
    C = rng.uniform(0, 5, size=(20, 40))
    W, z = grid5.solve(C)
    for r in range(20):
        Wr, zr = grid5.solve(C[r])
        assert np.array_equal(Wr[0], W[r]) and zr[0] == z[r]


def test_wrong_width_raises(grid5):
    with pytest.raises(ValueError):
        grid5.solve(np.ones((2, 39)))
    with pytest.raises(ValueError):
        Grid(1)


def test_pathset_two_paths_picks_the_first_one_in_a_tie(two):
    W, z = two.solve(np.array([[3.0, 5.0], [5.0, 3.0], [4.0, 4.0]]))
    assert W.tolist() == [[1, 0], [0, 1], [1, 0]] and z.tolist() == [3.0, 3.0, 4.0]
    assert two.d == 2


def test_pathset_on_all_grid_paths_equals_the_grid_solver_with_integer_costs(grid5):
    """Mit ganzzahligen Kosten (exakte Gleichstände) stimmen Rückwärtsrechnung und Aufzählung samt Gleichstandsregel überein."""
    rng = np.random.default_rng(4)
    C = rng.integers(0, 3, size=(200, 40)).astype(float)    # viele Gleichstände
    Wg, zg = grid5.solve(C)
    Wp, zp = PathSet(grid5.paths()).solve(C)
    assert np.array_equal(Wg, Wp) and np.array_equal(zg, zp)


def test_paths_are_in_lexicographic_order_right_before_down(grid5):
    P = grid5.paths()
    first = "".join("R" if grid5.edges[k][1] == grid5.edges[k][0] + 1 else "D" for k in _ordered_arcs(grid5, P[0]))
    last = "".join("R" if grid5.edges[k][1] == grid5.edges[k][0] + 1 else "D" for k in _ordered_arcs(grid5, P[-1]))
    assert first == "RRRRDDDD" and last == "DDDDRRRR"
    moves = ["".join("R" if grid5.edges[k][1] == grid5.edges[k][0] + 1 else "D" for k in _ordered_arcs(grid5, w)) for w in P]
    assert moves == sorted(moves, key=lambda m: m.replace("R", "0").replace("D", "1"))      # R = 0 vor D = 1


def test_node_xy_places_node_in_row_and_column(grid5):
    assert grid5.node_xy(0) == (0.0, 0.0) and grid5.node_xy(24) == (4.0, 4.0) and grid5.node_xy(7) == (2.0, 1.0)


def test_enumeration_count_for_other_sizes():
    for h in (2, 3, 4, 6):
        assert Grid(h).paths().shape[0] == math.comb(2 * (h - 1), h - 1)
    assert list(itertools.islice(Grid(3).paths().sum(axis=1), 6)) == [4.0] * 6
