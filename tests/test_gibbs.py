"""Gibbs-Glättung (ngl_gibbs): Handbeispiel von Hand, Grenzfälle, Eigenschaften (Normierung, Skalierung, Verschiebung, Monotonie in T), Gradient gegen Differenzenquotient."""

import math

import numpy as np
import pytest

import ngl_gibbs as G


def _sigma(u):
    return 1.0 / (1.0 + math.exp(-u))


# --- Handbeispiel: zwei Wege, wahre Kosten c = (3, 5), Vorhersage (4, 3) -----------------------------------------------------------------------

@pytest.mark.parametrize("T,w1,w2,ecost", [
    (0.1, 0.0000, 1.0000, 4.9999),      # sigma(-10) = 4,5e-5 auf Weg 1
    (1.0, 0.2689, 0.7311, 4.4621),      # 1 / (1 + e) = 0,2689
    (10.0, 0.4750, 0.5250, 4.0500),     # sigma(0,1) = 0,52498
])
def test_hand_example_weights_and_expected_cost_by_hand(two, hand_c, hand_chat, T, w1, w2, ecost):
    """Gewicht auf Weg 2 = 1 / (1 + exp(-(c1hat - c2hat)/T)) = 1 / (1 + exp(-1/T)); erwartete Kosten 3 w1 + 5 w2 (Handrechnung im Docstring, Zahlen wie im Plan, auf vier Stellen)."""
    pi = G.gibbs_weights(hand_chat, two.W_all, T)[0]
    assert pi[0] == pytest.approx(w1, abs=5e-5) and pi[1] == pytest.approx(w2, abs=5e-5)
    exact_w2 = 1.0 / (1.0 + math.exp(-1.0 / T))
    assert pi[1] == pytest.approx(exact_w2, rel=1e-12)
    assert G.expected_cost(hand_c, hand_chat, two.W_all, T)[0] == pytest.approx(ecost, abs=5e-5)
    assert G.expected_cost(hand_c, hand_chat, two.W_all, T)[0] == pytest.approx(3 * (1 - exact_w2) + 5 * exact_w2, rel=1e-12)


def test_hand_example_gradient_by_hand(two, hand_c, hand_chat):
    """d E[c^T w] / d c1hat = 2 sigma'(Delta/T)/T mit sigma' = sigma (1 - sigma); bei Delta = 1, T = 1: 2 * 0,19661 = 0,39322 und die Gegenrichtung für c2hat."""
    _, g = G.gibbs_parts(hand_c, hand_chat, two.W_all, 1.0)
    s = _sigma(1.0)
    assert g[0, 0] == pytest.approx(2 * s * (1 - s), rel=1e-12)
    assert g[0, 1] == pytest.approx(-g[0, 0], rel=1e-12)
    assert g[0, 0] == pytest.approx(0.39322, abs=5e-6)


def test_hand_example_smoothed_regret_is_two_times_weight_of_the_wrong_path(two, hand_c):
    for delta in (-3.0, -0.5, 0.0, 0.5, 2.0):
        chat = np.array([[delta, 0.0]])
        pi = G.gibbs_weights(chat, two.W_all, 1.0)[0]
        assert G.smoothed_regret(hand_c, chat, two.W_all, 1.0)[0] == pytest.approx(2.0 * pi[1], rel=1e-12)


def test_tie_gives_equal_weights(two, hand_c):
    pi = G.gibbs_weights(np.array([[2.0, 2.0]]), two.W_all, 0.7)[0]
    assert pi.tolist() == [0.5, 0.5]


# --- Grenzfälle ---------------------------------------------------------------------------------------------------------------------------------

def test_t_to_zero_gives_the_hard_choice(two, hand_chat):
    pi = G.gibbs_weights(hand_chat, two.W_all, 1e-9)[0]
    assert pi.tolist() == [0.0, 1.0]       # (4, 3): Weg 2 hat die kleineren vorhergesagten Kosten


def test_t_to_infinity_gives_uniform_weights_and_the_mean_path_cost(grid5):
    rng = np.random.default_rng(1)
    Chat, C = rng.uniform(1, 3, (5, 40)), rng.uniform(1, 3, (5, 40))
    P = grid5.paths()
    pi = G.gibbs_weights(Chat, P, 1e9)
    assert np.abs(pi - 1.0 / 70).max() < 1e-7
    assert np.allclose(G.expected_cost(C, Chat, P, 1e9), (C @ P.T).mean(axis=1), atol=1e-6)


def test_hand_example_expected_cost_falls_from_five_to_four_with_t(two, hand_c, hand_chat):
    """Handbeispiel (kein allgemeines Gesetz): erwartete Kosten bei T = 1e-9 gleich 5 (der gewählte Weg), bei T = 1e9 gleich dem Mittel (3 + 5)/2 = 4."""
    assert G.expected_cost(hand_c, hand_chat, two.W_all, 1e-9)[0] == pytest.approx(5.0, abs=1e-12)
    assert G.expected_cost(hand_c, hand_chat, two.W_all, 1e9)[0] == pytest.approx(4.0, abs=1e-6)


def test_very_small_temperature_does_not_overflow(grid5):
    rng = np.random.default_rng(2)
    Chat = rng.uniform(1, 3, (30, 40))
    P = grid5.paths()
    for T in (1e-3, 1e-6, 1e-12):
        pi = G.gibbs_weights(Chat, P, T)
        assert np.all(np.isfinite(pi)) and np.allclose(pi.sum(axis=1), 1.0)
        logw, logz = G.log_weights(Chat, P, T)
        assert np.all(np.isfinite(logz)) and np.all(logw <= 1e-12)
    # bei eindeutigem Minimum ist das Gewicht des harten Weges 1
    W, _ = grid5.solve(Chat)
    pi = G.gibbs_weights(Chat, P, 1e-12)
    assert np.array_equal((pi == 1.0).astype(float) @ P, W)


def test_nonpositive_or_nonfinite_temperature_is_rejected(two, hand_chat, hand_c):
    for T in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            G.gibbs_weights(hand_chat, two.W_all, T)
        with pytest.raises(ValueError):
            G.gibbs_parts(hand_c, hand_chat, two.W_all, T)


# --- Eigenschaften ------------------------------------------------------------------------------------------------------------------------------

def test_weights_are_nonnegative_and_sum_to_one(grid5):
    rng = np.random.default_rng(3)
    P = grid5.paths()
    for T in (0.05, 0.5, 5.0):
        pi = G.gibbs_weights(rng.uniform(0, 4, (20, 40)), P, T)
        assert np.all(pi >= 0) and np.allclose(pi.sum(axis=1), 1.0, atol=1e-14)


def test_scaling_forecast_and_temperature_by_the_same_factor_changes_nothing(grid5):
    rng = np.random.default_rng(4)
    Chat, C = rng.uniform(1, 3, (10, 40)), rng.uniform(1, 3, (10, 40))
    P = grid5.paths()
    for lam in (0.1, 7.0, 100.0):
        assert np.allclose(G.gibbs_weights(lam * Chat, P, 0.5 * lam), G.gibbs_weights(Chat, P, 0.5), atol=1e-12)
        # Kosten und Gradient: erwartete Kosten skalieren mit dem Faktor der wahren Kosten, hier bleiben die wahren Kosten gleich
        e1, g1 = G.gibbs_parts(C, lam * Chat, P, 0.5 * lam)
        e0, g0 = G.gibbs_parts(C, Chat, P, 0.5)
        assert np.allclose(e1, e0, atol=1e-10) and np.allclose(g1 * lam, g0, atol=1e-8)


def test_adding_a_constant_to_every_arc_does_not_change_the_weights(grid5):
    """Alle Wege haben 8 Bögen: eine Konstante auf allen Bögen verschiebt jeden Wegwert um 8 mal die Konstante, die Gewichte bleiben."""
    rng = np.random.default_rng(5)
    Chat = rng.uniform(1, 3, (10, 40))
    P = grid5.paths()
    assert np.array_equal(P.sum(axis=1), np.full(70, 8.0))
    for shift in (-4.0, 2.5, 50.0):
        assert np.allclose(G.gibbs_weights(Chat + shift, P, 0.7), G.gibbs_weights(Chat, P, 0.7), atol=1e-12)


def test_identical_columns_give_the_same_weight_to_swapped_paths(grid5):
    """Ist ĉ auf allen Bögen gleich, haben alle 70 Wege denselben Wert: Gleichverteilung für jedes T."""
    P = grid5.paths()
    pi = G.gibbs_weights(np.full((2, 40), 1.7), P, 0.3)
    assert np.allclose(pi, 1.0 / 70, atol=1e-15)


def test_expected_cost_lies_between_best_and_worst_path_and_smoothed_regret_is_nonnegative(grid5):
    rng = np.random.default_rng(6)
    P = grid5.paths()
    C, Chat = rng.uniform(1, 3, (30, 40)), rng.uniform(1, 3, (30, 40))
    cp = C @ P.T
    for T in (0.01, 0.3, 3.0, 100.0):
        e = G.expected_cost(C, Chat, P, T)
        assert np.all(e >= cp.min(axis=1) - 1e-12) and np.all(e <= cp.max(axis=1) + 1e-12)
        assert np.all(G.smoothed_regret(C, Chat, P, T) >= -1e-12)


def test_with_exact_forecast_the_expected_regret_grows_with_temperature(grid5):
    """ĉ = c: d/dT E[c^T w] = Var(c^T w) / T^2 >= 0, der erwartete Regret steigt mit T (von 0 bei T -> 0 zum Mittel über alle Wege)."""
    rng = np.random.default_rng(7)
    P = grid5.paths()
    C = rng.uniform(1, 3, (15, 40))
    temps = (1e-3, 0.03, 0.1, 0.3, 1.0, 3.0, 30.0)
    r = np.array([G.smoothed_regret(C, C, P, T) for T in temps])
    assert np.all(np.diff(r, axis=0) >= -1e-12)
    assert r[0].max() < 1e-6


def test_smoothed_regret_approaches_the_hard_regret_for_small_t(grid5):
    from ngl_losses import spo_loss

    rng = np.random.default_rng(8)
    C, Chat = rng.uniform(1, 3, (40, 40)), rng.uniform(1, 3, (40, 40))
    assert np.allclose(G.smoothed_regret(C, Chat, grid5.paths(), 1e-9), spo_loss(grid5, C, Chat), atol=1e-9)


# --- Gradient gegen Differenzenquotient ---------------------------------------------------------------------------------------------------------

def test_gradient_matches_central_differences_of_the_enumeration(grid5):
    """Zentrale Differenzen (h = 1e-6) der erwarteten Kosten über die Aufzählung; Fehler unter 1e-6."""
    rng = np.random.default_rng(9)
    P = grid5.paths()
    C, Chat = rng.uniform(1, 3, (3, 40)), rng.uniform(1, 3, (3, 40))
    for T in (0.1, 1.0, 3.0):
        _, g = G.gibbs_parts(C, Chat, P, T)
        h = 1e-6
        for k in (0, 7, 19, 39):
            plus, minus = Chat.copy(), Chat.copy()
            plus[:, k] += h
            minus[:, k] -= h
            fd = (G.expected_cost(C, plus, P, T) - G.expected_cost(C, minus, P, T)) / (2 * h)
            assert np.abs(fd - g[:, k]).max() < 1e-6, (T, k)


def test_objective_gradient_matches_central_differences_in_the_coefficients(grid5):
    """Die Zielfunktion des Trainings (mittlerer erwarteter Regret) gegen zentrale Differenzen in einigen Koeffizienten."""
    rng = np.random.default_rng(10)
    X, C = rng.normal(size=(30, 5)), rng.uniform(1, 3, (30, 40))
    coef = rng.uniform(0.0, 1.0, (6, 40))
    f = G.gibbs_objective(grid5.paths(), X, C, 0.5)
    loss, grad = f(coef)
    assert loss >= 0
    h = 1e-6
    for (i, j) in ((0, 0), (1, 5), (3, 20), (5, 39)):
        plus, minus = coef.copy(), coef.copy()
        plus[i, j] += h
        minus[i, j] -= h
        fd = (f(plus)[0] - f(minus)[0]) / (2 * h)
        assert abs(fd - grad[i, j]) < 1e-6, (i, j)
