"""Kostenwelt (ngl_world) und Lauf (ngl_runs): Formeln von Hand, Reproduzierbarkeit, Rauschbereich, Zusammenfassungen."""

import numpy as np
import pytest

import ngl_runs as R
from ngl_world import TAG_TEST, TAG_TRAIN, WorldSpec, draw_data, draw_weights, make_dataset, mean_costs


def test_mean_costs_by_hand():
    """p = 1, ein Bogen mit Gewicht 1: m = (x / 1 + 3)^deg / 3,5^deg + 1. Bei x = 0.5 und deg 1: 3,5 / 3,5 + 1 = 2; deg 2: (3,5/3,5)^2 + 1 = 2.
    Bei x = -0.5 und deg 2: (2,5 / 3,5)^2 + 1 = 1 + 25/49. Gewicht 0 (zweiter Bogen): m = (3/3,5)^deg + 1."""
    B = np.array([[1.0], [0.0]])
    X = np.array([[0.5], [-0.5]])
    m1 = mean_costs(B, X, WorldSpec(p=1, deg=1, noise=0.0))
    assert m1[0, 0] == pytest.approx(2.0) and m1[1, 0] == pytest.approx(2.5 / 3.5 + 1)
    m2 = mean_costs(B, X, WorldSpec(p=1, deg=2, noise=0.0))
    assert m2[0, 0] == pytest.approx(2.0) and m2[1, 0] == pytest.approx(1 + 25 / 49)
    assert m2[0, 1] == pytest.approx((3 / 3.5) ** 2 + 1)


def test_weights_are_zero_or_one_with_about_half_ones():
    B = draw_weights(40, WorldSpec(p=5), 11)
    assert B.shape == (40, 5) and set(np.unique(B)) <= {0.0, 1.0}
    assert 0.3 < B.mean() < 0.7


def test_same_seed_same_data_and_train_differs_from_test(grid5):
    spec = WorldSpec(deg=6, noise=0.5)
    a, b = make_dataset(spec, 40, 50, 30, 5), make_dataset(spec, 40, 50, 30, 5)
    assert np.array_equal(a.C_train, b.C_train) and np.array_equal(a.X_test, b.X_test) and np.array_equal(a.B, b.B)
    assert not np.array_equal(a.X_train[:30], a.X_test)
    assert not np.array_equal(make_dataset(spec, 40, 50, 30, 6).C_train, a.C_train)


def test_without_noise_observed_equals_mean_and_with_noise_ratio_is_inside_the_band():
    B = draw_weights(40, WorldSpec(), 3)
    _, C0, M0 = draw_data(B, WorldSpec(noise=0.0), 100, 3, TAG_TRAIN)
    assert np.array_equal(C0, M0)
    _, C1, M1 = draw_data(B, WorldSpec(noise=0.5), 500, 3, TAG_TRAIN)
    ratio = C1 / M1
    assert ratio.min() >= 0.5 and ratio.max() <= 1.5 and abs(ratio.mean() - 1.0) < 0.01
    _, C2, M2 = draw_data(B, WorldSpec(noise=1.0), 500, 3, TAG_TEST)
    assert (C2 / M2).min() >= 0.0


def test_costs_are_positive_in_the_standard_worlds(grid5):
    for deg in (1, 2, 4, 6):
        data = make_dataset(WorldSpec(deg=deg, noise=0.5), grid5.d, 300, 300, 2)
        assert data.C_train.min() > 0 and data.C_test.min() > 0


def test_oracle_has_zero_regret_without_noise_and_positive_regret_with_noise():
    r0 = R.run_world(R.RunSpec(deg=4, noise=0.0, n=100, seed=0, steps=5), snapshot_steps=(), keep_cases=False)
    r1 = R.run_world(R.RunSpec(deg=4, noise=0.5, n=100, seed=0, steps=5), snapshot_steps=(), keep_cases=False)
    assert abs(r0.regret["orakel"]) < 1e-12
    assert r1.regret["orakel"] > 0.02


def test_cell_result_has_percent_units_and_run_seeds_is_reproducible():
    spec = R.RunSpec(deg=4, noise=0.0, n=100, steps=10)
    rows = R.run_seeds(spec, [0, 1])
    assert [r["seed"] for r in rows] == [0, 1]
    assert rows == R.run_seeds(spec, [0, 1])
    run = R.run_world(R.RunSpec(deg=4, noise=0.0, n=100, seed=0, steps=10), snapshot_steps=(), keep_cases=False)
    assert rows[0]["regret_kq"] == pytest.approx(100 * run.regret["kq"])


def test_summarize_and_count_better():
    s = R.summarize([3.0, 1.0, 2.0, 10.0])
    assert s == {"median": 2.5, "min": 1.0, "max": 10.0}
    rows = [{"regret_spo": 1.0, "regret_kq": 2.0}, {"regret_spo": 2.0, "regret_kq": 2.0}, {"regret_spo": 3.0, "regret_kq": 2.0}]
    assert R.count_better(rows) == 1              # streng kleiner: ein Gleichstand zählt nicht


def test_case_regret_matches_the_total_normalized_regret(grid5):
    run = R.run_world(R.RunSpec(deg=4, noise=0.5, n=100, seed=1, steps=10))
    _, zstar = grid5.solve(run.data.C_test)
    total = np.sum(run.case_regret["kq"] * zstar) / np.sum(zstar)
    assert total == pytest.approx(run.regret["kq"])
    assert len(run.case_regret["spo"]) == run.spec.n_test
