"""Presets und Permalink: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellung, Einrasten der Adresszeile, vorgerechnete Zellen zu jedem Preset vorhanden."""

import pytest

import ngl_bands as B
import ngl_constants as C
import ngl_evaluation as ev
import ngl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_choices_and_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert p["deg"] in C.DEG_CHOICES and p["noise"] in C.NOISE_CHOICES and p["n"] in C.N_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    d = ev.Settings()
    assert C.PRESETS[first] == {"deg": d.deg, "noise": d.noise, "n": d.n, "seed": d.seed}
    assert {k: spec.default for k, spec in P.SETTING_SPECS.items()} == {
        "deg_slider": d.deg, "noise_slider": d.noise, "n_slider": d.n, "seed_input": d.seed,
        "delta_slider": d.delta, "hand_t_slider": d.hand_t, "pert_slider": d.pert, "scale_slider": d.scale,
    }


def test_setting_specs_cover_every_preset_key_and_defaults_are_allowed_values():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_every_preset_has_a_precomputed_cell():
    for name, p in C.PRESETS.items():
        assert B.has_cell(p["deg"], p["n"], p["noise"]), name
        assert len(B.cell(p["deg"], p["n"], p["noise"])["regret_kq"]) == len(C.BAND_SEEDS)


def test_snap_rounds_to_the_nearest_allowed_value_and_limits_the_rest():
    spec = P.SETTING_SPECS
    assert P.snap(spec["deg_slider"], 3) == 2          # genau in der Mitte von 2 und 4: der kleinere
    assert P.snap(spec["deg_slider"], 5) == 4 and P.snap(spec["deg_slider"], 99) == 6 and P.snap(spec["deg_slider"], -3) == 1
    assert P.snap(spec["noise_slider"], 0.3) == 0.5 and P.snap(spec["noise_slider"], 0.2) == 0.0 and P.snap(spec["noise_slider"], 7.0) == 0.5
    assert P.snap(spec["n_slider"], 10) == 100 and P.snap(spec["n_slider"], 2000) == 1000 and P.snap(spec["n_slider"], 250) == 300
    assert P.snap(spec["seed_input"], -5) == 0 and P.snap(spec["seed_input"], 10**9) == C.SEED_MAX
    assert P.snap(spec["delta_slider"], 9.9) == C.DELTA_MAX and P.snap(spec["delta_slider"], -9.9) == C.DELTA_MIN
    assert P.snap(spec["delta_slider"], 0.3) == 0.25 and P.snap(spec["delta_slider"], 0.4) == 0.5
    assert P.snap(spec["hand_t_slider"], 2.0) == 1.0 and P.snap(spec["hand_t_slider"], 50.0) == 10.0       # 2 liegt in der Mitte von 1 und 3: der kleinere
    assert P.snap(spec["pert_slider"], 3.0) == 2.0 and P.snap(spec["scale_slider"], 40) == 10 and P.snap(spec["scale_slider"], 70) == 100


@pytest.mark.parametrize("deg,noise,n", [(d, e, n) for d in C.DEG_CHOICES for e in C.NOISE_CHOICES for n in C.N_CHOICES])
def test_settings_run_spec_matches_the_constants(deg, noise, n):
    spec = ev.Settings(deg=deg, noise=noise, n=n).run_spec()
    assert (spec.p, spec.h, spec.steps, spec.lr, spec.n_test) == (C.N_FEATURES, C.GRID_H, C.STEPS, C.LR, C.N_TEST)
    assert (spec.deg, spec.noise, spec.n) == (deg, noise, n)
    assert spec.temperatures == C.TRAIN_TEMPS
