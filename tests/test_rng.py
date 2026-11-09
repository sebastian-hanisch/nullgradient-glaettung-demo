"""Zufallsstrom (ngl_rng): SplitMix64 gegen eine reine Python-Ganzzahlrechnung, Teilströme, Verteilungen."""

import numpy as np

from ngl_rng import SplitMix64, derive_seed

MASK = (1 << 64) - 1


def reference_splitmix(seed: int, count: int) -> list[int]:
    """Die Vorschrift von Steele, Lea und Flood in reiner Python-Ganzzahlrechnung (kein numpy)."""
    state = seed & MASK
    out = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


def test_reference_has_the_published_first_value_for_seed_zero():
    assert reference_splitmix(0, 1)[0] == 0xE220A8397B1DCDAF  # bekannter erster Ausgabewert von SplitMix64 mit Seed 0


def test_numpy_version_equals_the_integer_reference():
    for seed in (0, 1, 7, 123456789, 2**63 + 5, 2**64 - 1):
        got = [int(v) for v in SplitMix64(seed).u64(25)]
        assert got == reference_splitmix(seed, 25), seed


def test_stream_continues_where_it_stopped():
    a = SplitMix64(42)
    first, second = a.u64(10), a.u64(15)
    assert [int(v) for v in np.concatenate([first, second])] == reference_splitmix(42, 25)


def test_uniform_is_in_the_half_open_unit_interval_with_mean_one_half():
    u = SplitMix64(5).uniform(20000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01


def test_normal_has_mean_zero_and_unit_variance():
    z = SplitMix64(9).normal(40000)
    assert abs(z.mean()) < 0.03 and abs(z.var() - 1.0) < 0.05


def test_derive_seed_is_deterministic_and_separates_tags_and_seeds():
    assert derive_seed(7, 1) == derive_seed(7, 1)
    values = {derive_seed(s, t) for s in range(5) for t in range(1, 4)}
    assert len(values) == 15
