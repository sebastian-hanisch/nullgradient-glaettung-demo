"""Training (ngl_training): Adam von Hand, KQ auf linearer Wahrheit, SPO+-Trainingsverlust gegen KQ. Trainierte Zahlen nur als Ungleichungen und Bänder."""

import numpy as np
import pytest

import ngl_training as T
from ngl_runs import RunSpec, run_world
from ngl_world import WorldSpec, make_dataset


def test_adam_first_step_has_length_lr_in_every_coordinate():
    """Erster Adam-Schritt von Hand: m = 0,1 g, v = 0,001 g^2, nach der Verzerrungskorrektur m^ = g, v^ = g^2: Schritt = lr g / (|g| + 1e-8), also fast lr * Vorzeichen(g)."""
    g = np.array([2.0, -0.5, 100.0])
    res = T.adam_minimize(lambda c: (0.0, g), np.zeros(3), steps=1, lr=0.05, snapshot_steps=(1,))
    expected = -0.05 * g / (np.abs(g) + 1e-8)
    assert np.allclose(res.snapshots[1], expected, rtol=0, atol=1e-15)
    assert np.abs(res.snapshots[1] + 0.05 * np.sign(g)).max() < 1e-7


def test_adam_second_step_by_hand():
    """Zweiter Schritt mit konstantem Gradienten g = 1: m2 = 0,9 * 0,1 + 0,1 = 0,19, v2 = 0,999 * 0,001 + 0,001 = 0,001999; m^ = 0,19 / (1 - 0,81) = 1, v^ = 0,001999 / (1 - 0,998001) = 1: wieder Schritt lr."""
    res = T.adam_minimize(lambda c: (0.0, np.ones(1)), np.zeros(1), steps=2, lr=0.1, snapshot_steps=(2,))
    assert res.snapshots[2][0] == pytest.approx(-0.2, abs=1e-6)


def test_adam_minimizes_a_quadratic_and_returns_the_best_visited_point():
    target = np.array([3.0, -1.0])
    res = T.adam_minimize(lambda c: (float(np.sum((c - target) ** 2)), 2 * (c - target)), np.zeros(2), steps=400, lr=0.1)
    assert np.abs(res.coef - target).max() < 0.1
    assert res.losses.shape == (400,) and res.losses[0] == pytest.approx(10.0)
    assert res.losses[res.best_step] == pytest.approx(res.losses.min())      # Parameter und Verlust gehören zusammen
    assert res.best_step > 0


def test_adam_snapshot_zero_is_the_start_and_snapshots_are_copies():
    start = np.array([1.0, 2.0])
    res = T.adam_minimize(lambda c: (0.0, np.ones(2)), start, steps=3, lr=0.1, snapshot_steps=(0, 3))
    assert np.array_equal(res.snapshots[0], start)
    start[0] = 99.0
    assert res.snapshots[0][0] == 1.0


def test_least_squares_recovers_a_linear_truth_exactly(grid5):
    """Grad 1 ohne Rauschen: m(x) = (B x / sqrt(p) + 3) / 3,5 + 1 ist linear in x. KQ findet Achsenabschnitt 1 + 3/3,5 und Steigungen B / (3,5 sqrt(p)) bis auf Rundung."""
    spec = WorldSpec(p=5, deg=1, noise=0.0)
    data = make_dataset(spec, grid5.d, 200, 50, seed=3)
    coef = T.fit_least_squares(data.X_train, data.C_train)
    assert np.abs(coef[0] - (1 + 3 / 3.5)).max() < 1e-9
    assert np.abs(coef[1:] - data.B.T / (3.5 * np.sqrt(5))).max() < 1e-9
    assert np.abs(T.predict(coef, data.X_test) - data.C_test).max() < 1e-9


def test_predict_applies_intercept_and_slope():
    coef = np.array([[1.0, 2.0], [10.0, 0.0], [0.0, 100.0]])        # Achsenabschnitt (1, 2), Steigungen
    assert T.predict(coef, np.array([[1.0, 2.0]])).tolist() == [[11.0, 202.0]]


def test_spo_plus_training_loss_is_never_above_the_least_squares_start(grid5):
    """Der Start (KQ) gehört zu den besuchten Parametern: der Trainingsverlust des Ergebnisses ist höchstens der von KQ (jede Welt)."""
    for deg, noise, n, seed in [(6, 0.0, 100, 1), (6, 0.5, 100, 2), (4, 0.0, 300, 3), (1, 0.0, 100, 4), (2, 0.5, 300, 5)]:
        run = run_world(RunSpec(deg=deg, noise=noise, n=n, seed=seed, steps=60), snapshot_steps=(), keep_cases=False)
        assert run.train_loss_spo <= run.train_loss_kq + 1e-12, (deg, noise, n)


def test_train_loss_history_starts_at_the_start_value(grid5):
    run = run_world(RunSpec(deg=6, noise=0.0, n=100, seed=1, steps=20), snapshot_steps=(0, 20), keep_cases=False)
    assert run.train_losses[0] == pytest.approx(run.train_loss_kq)
    assert run.train_losses.shape == (20,)
    assert set(run.snapshots) == {0, 20} and np.array_equal(run.snapshots[0], run.coef_kq)


def test_zero_steps_returns_the_start(grid5):
    data = make_dataset(WorldSpec(deg=4), grid5.d, 100, 10, seed=1)
    kq = T.fit_least_squares(data.X_train, data.C_train)
    res = T.train_spo_plus(grid5, data.X_train, data.C_train, steps=0, init=kq)
    assert np.array_equal(res.coef, kq) and res.losses.size == 0 and res.best_step == 0


def test_spo_plus_lowers_the_regret_of_a_wrong_linear_model_on_one_world_band(grid5):
    """Welt Grad 6 ohne Rauschen, n = 1000: SPO+ liegt im Regret unter KQ (in der Messung dieser Demo in 20 von 20 Seeds; hier zwei Seeds als Probe, kein Beleg für die Zahl)."""
    for seed in (0, 1):
        run = run_world(RunSpec(deg=6, noise=0.0, n=1000, seed=seed), snapshot_steps=(), keep_cases=False)
        assert run.regret["spo"] < run.regret["kq"], seed
        assert run.regret["orakel"] == pytest.approx(0.0, abs=1e-12)


def test_with_a_linear_truth_there_is_nothing_to_win(grid5):
    run = run_world(RunSpec(deg=1, noise=0.0, n=100, seed=0), snapshot_steps=(), keep_cases=False)
    assert run.regret["kq"] == pytest.approx(0.0, abs=1e-12)
    assert run.regret["spo"] < 0.002              # SPO+ startet bei KQ und bleibt dort (Band, nicht exakt: Adam kann den Parameter verlassen)


def test_l2_penalty_changes_the_gradient_not_the_reported_loss(grid5):
    rng = np.random.default_rng(0)
    X, C = rng.normal(size=(40, 5)), rng.uniform(1, 4, size=(40, 40))
    coef = rng.normal(size=(6, 40))
    l0, g0 = T.spo_plus_objective(grid5, X, C, 0.0)(coef)
    l1, g1 = T.spo_plus_objective(grid5, X, C, 0.5)(coef)
    assert l0 == l1
    assert np.allclose(g1 - g0, 0.5 * coef)
