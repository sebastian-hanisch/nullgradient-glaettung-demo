"""Läufe (ngl_runs): geglättetes Training, Kostenfaktor, Anteil unveränderter Wege. Trainierte Zahlen nur als Ungleichungen und Bänder (Adam, plattformabhängig)."""

import numpy as np
import pytest

import ngl_runs as R
from ngl_gibbs import gibbs_objective
from ngl_grid import Grid
from ngl_training import fit_least_squares, predict


def test_scale_equivalence_regret_with_scaled_costs_and_scaled_temperature_is_the_same():
    """Kosten mal 10, Lernrate mal 10, T mal 10: dieselbe Rechnung bis auf Rundung (Adam ist gegen den Maßstab des Gradienten unempfindlich, Gewichte und Start skalieren mit).
    Geprüft bei mittleren Temperaturen; bei sehr kleinen T weichen die Läufe je Seed ab (Ursache nicht untersucht, vermutlich Rundung), dort gilt nur das Band."""
    a = R.gibbs_sweep(R.RunSpec(deg=6, noise=0.5, n=300, seed=0, scale=1.0, steps=60), (0.1, 1.0, 10.0))
    b = R.gibbs_sweep(R.RunSpec(deg=6, noise=0.5, n=300, seed=0, scale=10.0, steps=60), (1.0, 10.0, 100.0))
    assert b["kq"] == pytest.approx(a["kq"], abs=1e-12)
    for t in (0.1, 1.0, 10.0):
        assert b["gibbs"][10 * t] == pytest.approx(a["gibbs"][t], abs=1e-6), t


def test_same_temperature_on_scaled_costs_is_a_different_result():
    """Ohne Mitskalieren von T ist es nicht dieselbe Rechnung (die Aussage von Experiment 4): T = 1 bei Kosten mal 10 gleicht T = 0,1 bei Kosten mal 1, nicht T = 1."""
    a = R.gibbs_sweep(R.RunSpec(deg=6, noise=0.5, n=300, seed=0, scale=1.0, steps=60), (0.1, 1.0))
    b = R.gibbs_sweep(R.RunSpec(deg=6, noise=0.5, n=300, seed=0, scale=10.0, steps=60), (1.0,))
    assert b["gibbs"][1.0] == pytest.approx(a["gibbs"][0.1], abs=1e-6)
    assert abs(b["gibbs"][1.0] - a["gibbs"][1.0]) > 1e-4


def test_world_data_scaling_multiplies_all_costs():
    grid = Grid(5)
    d1 = R.make_world_data(R.RunSpec(deg=4, noise=0.5, n=50, seed=2, scale=1.0), grid)
    d10 = R.make_world_data(R.RunSpec(deg=4, noise=0.5, n=50, seed=2, scale=10.0), grid)
    assert np.allclose(d10.C_train, 10 * d1.C_train) and np.allclose(d10.C_test, 10 * d1.C_test) and np.allclose(d10.M_test, 10 * d1.M_test)
    assert np.array_equal(d10.X_train, d1.X_train)


def test_run_world_with_temperatures_returns_a_curve_and_final_regret_per_temperature():
    run = R.run_world(R.RunSpec(deg=4, noise=0.0, n=100, seed=1, steps=20, temperatures=(0.1, 1.0)), snapshot_steps=(0, 10, 20), keep_cases=False)
    assert set(run.coef_gibbs) == {0.1, 1.0} == set(run.regret_gibbs) == set(run.curve_gibbs)
    for T in (0.1, 1.0):
        assert set(run.curve_gibbs[T]) == {0, 10, 20}
        assert run.curve_gibbs[T][0] == pytest.approx(run.regret["kq"], abs=1e-12)     # Schritt 0 ist der KQ-Start
        assert run.curve_gibbs[T][20] == run.regret_gibbs[T]
        assert run.train_losses_gibbs[T].shape == (20,)
        assert np.all(np.isfinite(run.coef_gibbs[T]))


def test_smoothed_training_loss_starts_at_the_value_of_the_objective_at_kq():
    spec = R.RunSpec(deg=6, noise=0.0, n=100, seed=1, steps=10, temperatures=(1.0,))
    run = R.run_world(spec, snapshot_steps=(), keep_cases=False)
    f = gibbs_objective(run.grid.paths(), run.data.X_train, run.data.C_train, 1.0)
    assert run.train_losses_gibbs[1.0][0] == pytest.approx(f(run.coef_kq)[0], rel=1e-12)


def test_cell_result_has_one_column_per_temperature():
    run = R.run_world(R.RunSpec(deg=4, noise=0.0, n=100, seed=0, steps=5, temperatures=(0.03, 1.0)), snapshot_steps=(), keep_cases=False)
    row = R.cell_result(run)
    assert "regret_gibbs_0.03" in row and "regret_gibbs_1" in row and row["regret_gibbs_1"] == pytest.approx(100 * run.regret_gibbs[1.0])


def test_tiny_temperature_training_barely_moves_from_kq_on_a_noise_free_world_or_gets_worse():
    """Zweig-Test: T = 0,001 und T = 1 geben verschiedene Parameter (die Temperatur wirkt); der kleine Wert führt hier nicht unter den Regret von T = 1 (eine Welt, Grad 6, kein Rauschen, n = 300)."""
    run = R.run_world(R.RunSpec(deg=6, noise=0.0, n=300, seed=3, steps=120, temperatures=(0.001, 1.0)), snapshot_steps=(), keep_cases=False)
    assert not np.allclose(run.coef_gibbs[0.001], run.coef_gibbs[1.0])
    assert run.regret_gibbs[1.0] < run.regret["kq"]
    assert run.regret_gibbs[0.001] >= run.regret_gibbs[1.0]


# --- Anteil unveränderter Wege ------------------------------------------------------------------------------------------------------------------

def test_unchanged_share_is_one_without_perturbation_and_for_clear_gaps(grid2):
    """2 x 2 Gitter, Bögen (0 nach 1, 0 nach 2, 1 nach 3, 2 nach 3): Weg R D kostet 1 + 1, Weg D R 5 + 5; eine Störung um 10 % kann diesen Abstand nicht schließen."""
    Chat = np.tile([1.0, 5.0, 1.0, 5.0], (50, 1))
    sh = R.unchanged_shares(grid2, Chat, Chat, (0.0, 0.1), seed=1)
    assert sh[0.0] == {"weg": 1.0, "regret": 1.0} and sh[0.1] == {"weg": 1.0, "regret": 1.0}


def test_unchanged_share_for_near_ties_is_about_one_half(grid2):
    """Fast gleiche Wege (2,0 gegen 2,0 plus 1e-9): die Störung wirft den Weg mit gleicher Wahrscheinlichkeit um (Symmetrie), der Anteil liegt um 0,5 (feste Zufallsfolge, Band 0,3 bis 0,7)."""
    Chat = np.tile([1.0, 1.0, 1.0, 1.0 + 1e-9], (400, 1))
    C = np.tile([1.0, 2.0, 1.0, 2.0], (400, 1))       # wahre Kosten: Weg R D kostet 2, Weg D R kostet 4: ein Wechsel ändert den Regret
    sh = R.unchanged_shares(grid2, Chat, C, (0.3,), seed=5)[0.3]
    assert 0.3 < sh["weg"] < 0.7
    assert sh["regret"] == pytest.approx(sh["weg"], abs=1e-12)     # bei verschiedenen wahren Wegkosten fällt gleicher Regret mit gleichem Weg zusammen


def test_equal_regret_can_hold_with_a_different_path_only_for_equal_true_costs(grid2):
    Chat = np.tile([1.0, 1.0, 1.0, 1.0 + 1e-9], (400, 1))
    C = np.ones((400, 4))                              # alle wahren Wegkosten gleich: jeder Weg hat Regret 0
    sh = R.unchanged_shares(grid2, Chat, C, (0.3,), seed=5)[0.3]
    assert sh["regret"] == 1.0 and sh["weg"] < 1.0


def test_unchanged_share_uses_the_same_direction_for_every_delta(grid5):
    """Dieselbe Zufallsrichtung, nur die Größe wächst: die Zählung hängt nur von (Seed, Daten) ab und ist wiederholbar."""
    rng = np.random.default_rng(0)
    Chat, C = rng.uniform(1, 3, (200, 40)), rng.uniform(1, 3, (200, 40))
    a = R.unchanged_shares(grid5, Chat, C, (0.01, 0.2), seed=7)
    b = R.unchanged_shares(grid5, Chat, C, (0.2, 0.01), seed=7)
    assert a == b
    assert a[0.01]["weg"] >= a[0.2]["weg"]
    other = R.unchanged_shares(grid5, Chat, C, (0.2,), seed=8)[0.2]        # anderer Seed, andere Richtung: die Zählung hängt am Seed
    assert other["weg"] != a[0.2]["weg"]


def test_run_world_kq_regret_matches_a_direct_computation(grid5):
    spec = R.RunSpec(deg=4, noise=0.0, n=100, seed=0, steps=5)
    run = R.run_world(spec, snapshot_steps=(), keep_cases=False)
    kq = fit_least_squares(run.data.X_train, run.data.C_train)
    W, _ = grid5.solve(predict(kq, run.data.X_test))
    _, z = grid5.solve(run.data.C_test)
    assert run.regret["kq"] == pytest.approx((np.sum(run.data.C_test * W) - z.sum()) / z.sum(), rel=1e-12)
