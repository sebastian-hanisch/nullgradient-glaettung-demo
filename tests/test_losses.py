"""Verluste (ngl_losses): Handbeispiel, Eigenschaften von SPO, SPO+ und Teilgradient, Randfälle."""

import numpy as np
import pytest

import ngl_losses as L
from ngl_grid import PathSet, two_paths


def test_hand_example_two_paths(two, hand_c, hand_chat):
    """Handrechnung: c = (3, 5), ch = (4, 3). Unter ch gewinnt Weg 2 (3 < 4), er kostet wirklich 5, der beste Weg kostet 3: SPO-Verlust 2.
    SPO+ = max(c - 2 ch . w) + 2 ch . w*(c) - z*(c) = max(-5, -1) + 8 - 3 = 4. Teilgradient 2 (w*(c) - w(2 ch - c)) mit 2 ch - c = (5, 1) -> Weg 2: 2 ((1,0) - (0,1)) = (2, -2).
    Quadratfehler (3-4)^2 + (5-3)^2 = 5."""
    assert L.spo_loss(two, hand_c, hand_chat).tolist() == [2.0]
    assert L.spo_plus_loss(two, hand_c, hand_chat).tolist() == [4.0]
    assert L.spo_plus_subgradient(two, hand_c, hand_chat).tolist() == [[2.0, -2.0]]
    assert L.squared_error(hand_c, hand_chat).tolist() == [5.0]
    assert L.mse_per_arc(hand_c, hand_chat) == 2.5
    assert L.normalized_regret(two, hand_c, hand_chat) == pytest.approx(2.0 / 3.0)


def test_hand_example_along_the_line_through_c(two, hand_c):
    """Entlang ch(t) = c + t (4 - 3, 3 - 5) = (3 + t, 5 - 2 t): Weg 2 gewinnt für t > 2/3 (5 - 2t < 3 + t), SPO = 2 dann, sonst 0;
    SPO+(t) = max(0, 6 t - 2) (von Hand: max(-3 - 2t, -5 + 4t) + 2 (3 + t) - 3); Quadratfehler 5 t^2."""
    for t in (-1.0, 0.0, 0.3, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.7, 1.0, 2.5):
        chat = hand_c + t * np.array([[1.0, -2.0]])
        assert L.spo_plus_loss(two, hand_c, chat)[0] == pytest.approx(max(0.0, 6 * t - 2)), t
        assert L.spo_loss(two, hand_c, chat)[0] == (2.0 if t > 2.0 / 3.0 else 0.0), t
        assert L.squared_error(hand_c, chat)[0] == pytest.approx(5 * t * t), t


def test_right_decision_with_a_bad_forecast(two, hand_c):
    """ch = (6, 7): falsche Kosten (Quadratfehler 9 + 4 = 13), aber Weg 1 gewinnt wie unter c: Regret 0 (SPO), und SPO+ = max(-9, -9) + 12 - 3 = 0."""
    chat = np.array([[6.0, 7.0]])
    assert L.squared_error(hand_c, chat)[0] == 13.0
    assert L.spo_loss(two, hand_c, chat)[0] == 0.0
    assert L.spo_plus_loss(two, hand_c, chat)[0] == 0.0


def test_perfect_prediction_has_no_regret_and_zero_spo_plus_on_the_grid(grid5):
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 4, size=(50, 40))
    assert np.abs(L.spo_loss(grid5, C, C)).max() < 1e-12        # Rundungsreste von einigen 1e-15 (Rückwärtssumme gegen Skalarprodukt), nie eine echte Abweichung
    assert abs(L.normalized_regret(grid5, C, C)) < 1e-12
    # SPO+ bei ch = c: max_w (-c) w + 2 c w* - z* = -z + 2 z* - z* = 0
    assert np.allclose(L.spo_plus_loss(grid5, C, C), 0.0, atol=1e-12)


def test_single_path_has_regret_zero_for_every_forecast():
    one = PathSet(np.array([[1.0, 1.0, 0.0]]))
    rng = np.random.default_rng(1)
    C, Chat = rng.uniform(0, 5, size=(20, 3)), rng.uniform(-3, 9, size=(20, 3))
    assert np.all(L.spo_loss(one, C, Chat) == 0)
    assert np.allclose(L.spo_plus_loss(one, C, Chat), 0.0, atol=1e-12)   # w = w* immer: (c - 2ch) w + 2 ch w - c w = 0


def test_spo_plus_is_an_upper_bound_of_spo(grid5):
    rng = np.random.default_rng(2)
    for _ in range(20):
        C = rng.uniform(0.5, 4, size=(100, 40))
        Chat = C + rng.normal(0, rng.uniform(0.1, 3), size=C.shape)
        assert np.all(L.spo_plus_loss(grid5, C, Chat) >= L.spo_loss(grid5, C, Chat) - 1e-9)


def test_spo_is_zero_exactly_when_the_chosen_path_is_optimal(grid5):
    rng = np.random.default_rng(3)
    C = rng.uniform(0.5, 4, size=(300, 40))
    Chat = C + rng.normal(0, 0.7, size=C.shape)
    spo = L.spo_loss(grid5, C, Chat)
    W, _ = grid5.solve(Chat)
    Wstar, zstar = grid5.solve(C)
    optimal = np.isclose(np.sum(C * W, axis=1), zstar, atol=1e-12)
    assert np.array_equal(spo <= 1e-12, optimal)
    assert 0 < optimal.sum() < len(optimal)        # beide Fälle kommen vor


def test_scaling_the_forecast_keeps_the_path_but_changes_spo_plus(grid5):
    rng = np.random.default_rng(4)
    C = rng.uniform(0.5, 4, size=(100, 40))
    Chat = C + rng.normal(0, 0.8, size=C.shape)
    W1, _ = grid5.solve(Chat)
    W2, _ = grid5.solve(5.0 * Chat)
    assert np.array_equal(W1, W2)
    assert np.array_equal(L.spo_loss(grid5, C, Chat), L.spo_loss(grid5, C, 5.0 * Chat))
    assert not np.allclose(L.spo_plus_loss(grid5, C, Chat), L.spo_plus_loss(grid5, C, 5.0 * Chat))


def test_shifting_all_arcs_by_a_constant_keeps_the_decision_and_the_regret(grid5):
    rng = np.random.default_rng(5)
    C = rng.uniform(0.5, 4, size=(100, 40))
    Chat = C + rng.normal(0, 0.8, size=C.shape)
    assert np.allclose(L.spo_loss(grid5, C + 1.5, Chat + 7.0), L.spo_loss(grid5, C, Chat))


def test_spo_plus_is_convex_along_random_segments(grid5):
    """Konvex in ch: der Wert in der Mitte liegt nicht über dem Mittel der Endwerte."""
    rng = np.random.default_rng(6)
    C = rng.uniform(0.5, 4, size=(100, 40))
    A, B = rng.normal(2, 1.5, size=(100, 40)), rng.normal(2, 1.5, size=(100, 40))
    mid = L.spo_plus_loss(grid5, C, 0.5 * (A + B))
    assert np.all(mid <= 0.5 * (L.spo_plus_loss(grid5, C, A) + L.spo_plus_loss(grid5, C, B)) + 1e-9)


def test_subgradient_inequality_on_random_points(grid5):
    """Teilgradienten-Ungleichung: f(u) >= f(a) + s(a) . (u - a) für alle u."""
    rng = np.random.default_rng(7)
    C = rng.uniform(0.5, 4, size=(100, 40))
    A = rng.normal(2, 1.5, size=(100, 40))
    fa, S = L.spo_plus_parts(grid5, C, A)
    for _ in range(10):
        U = A + rng.normal(0, 1.0, size=A.shape)
        assert np.all(L.spo_plus_loss(grid5, C, U) >= fa + np.sum(S * (U - A), axis=1) - 1e-9)


def test_parts_equal_the_separate_functions(grid5):
    rng = np.random.default_rng(8)
    C = rng.uniform(0.5, 4, size=(10, 40))
    Chat = rng.normal(2, 1, size=(10, 40))
    loss, grad = L.spo_plus_parts(grid5, C, Chat)
    assert np.array_equal(loss, L.spo_plus_loss(grid5, C, Chat))
    assert np.array_equal(grad, L.spo_plus_subgradient(grid5, C, Chat))


def test_zero_forecast_follows_the_tie_rule(grid5):
    """ch = 0 (alle Bögen gleich): der gewählte Weg ist der erste der Gleichstandsregel (rechts vor unten), nicht eine zufällige Wahl."""
    C = np.random.default_rng(9).uniform(1, 2, size=(5, 40))
    W, _ = grid5.solve(np.zeros((5, 40)))
    first = grid5.paths()[0]
    assert all(np.array_equal(w, first) for w in W)
    assert np.all(L.spo_loss(grid5, C, np.zeros((5, 40))) >= 0)


def test_two_path_toy_is_the_same_solver_interface():
    assert two_paths().solve(np.array([[1.0, 2.0]]))[1].tolist() == [1.0]
