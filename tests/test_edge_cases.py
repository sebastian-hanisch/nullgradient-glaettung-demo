"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, Nullen, kleinstes Gitter, Grenzen der Regler."""

import numpy as np
import pytest

import ngl_evaluation as ev
import ngl_gibbs as G
import ngl_runs as R
from ngl_world import WorldSpec, make_dataset


def test_exactly_two_nodes_per_side_is_the_smallest_grid(grid2):
    assert grid2.d == 4 and grid2.paths().shape == (2, 4)
    marg, logz = G.edge_marginals(grid2, np.zeros((1, 4)), 1.0)
    assert logz[0] == pytest.approx(np.log(2.0)) and np.allclose(marg, 0.5)


def test_world_run_on_the_smallest_grid_does_not_break():
    run = R.run_world(R.RunSpec(h=2, deg=2, noise=0.0, n=20, seed=1, steps=10, n_test=30, temperatures=(0.1, 1.0)), snapshot_steps=(0, 10))
    assert run.grid.d == 4 and run.data.C_test.shape == (30, 4) and set(run.regret_gibbs) == {0.1, 1.0}


@pytest.mark.parametrize("n", [1, 2])
def test_smoothed_training_with_one_or_two_cases_runs(grid5, n):
    data = make_dataset(WorldSpec(deg=4), grid5.d, n, 5, seed=3)
    res = R.train_gibbs(grid5.paths(), data.X_train, data.C_train, 1.0, steps=10, snapshot_steps=(10,))
    assert res.snapshots[10].shape == (6, 40) and np.all(np.isfinite(res.snapshots[10]))


def test_all_zero_costs_give_zero_regret_and_zero_gradient(grid5):
    C = np.zeros((3, 40))
    P = grid5.paths()
    e, g = G.gibbs_parts(C, C, P, 1.0)
    assert np.all(e == 0) and np.all(g == 0) and np.all(G.smoothed_regret(C, C, P, 1.0) == 0)


def test_negative_forecasts_are_allowed(grid5):
    rng = np.random.default_rng(0)
    C, Chat = rng.uniform(1, 3, (10, 40)), rng.uniform(-5, -1, (10, 40))
    pi = G.gibbs_weights(Chat, grid5.paths(), 0.5)
    assert np.allclose(pi.sum(axis=1), 1.0) and np.all(G.smoothed_regret(C, Chat, grid5.paths(), 0.5) >= -1e-12)


def test_huge_forecast_values_do_not_overflow(grid5):
    Chat = np.random.default_rng(1).uniform(1e5, 2e5, (5, 40))
    pi = G.gibbs_weights(Chat, grid5.paths(), 1.0)
    assert np.all(np.isfinite(pi)) and np.allclose(pi.sum(axis=1), 1.0)


def test_hand_state_at_the_slider_corners():
    for delta in (-4.0, 0.0, 4.0):
        for T in (0.1, 10.0):
            h = ev.hand_state(delta, T)
            assert h["chosen"] in (1, 2) and h["regret"] in (0.0, 2.0) and abs(h["w1"] + h["w2"] - 1) < 1e-12 and h["ecost"] >= 3.0 - 1e-12


def test_perturbation_of_zero_size_changes_nothing(grid5):
    rng = np.random.default_rng(2)
    Chat, C = rng.uniform(1, 3, (30, 40)), rng.uniform(1, 3, (30, 40))
    assert R.unchanged_shares(grid5, Chat, C, (0.0,), seed=1)[0.0] == {"weg": 1.0, "regret": 1.0}


def test_noise_free_linear_world_has_zero_regret_for_kq_and_a_small_one_for_tiny_temperature():
    run = R.run_world(R.RunSpec(deg=1, noise=0.0, n=100, seed=0, steps=60, temperatures=(0.001, 1.0)), snapshot_steps=(), keep_cases=False)
    assert run.regret["kq"] == pytest.approx(0.0, abs=1e-12)
    assert run.regret_gibbs[0.001] >= 0.0
