"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart und Kostenwelt in der Seitenleiste, experimentbezogene Regler (Handbeispiel, Testfall) im Hauptbereich,
aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler
dort steht: sonst bliebe das Preset trotz Änderung markiert. Ein Preset setzt alle Regler (die übrigen auf ihren Standardwert)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ngl_constants as C
import ngl_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        for bereich in (at.sidebar, at.main):
            try:
                return getattr(bereich, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} steht weder in der Seitenleiste noch im Hauptbereich")


WELT_REGLER = {"deg_slider", "noise_slider", "n_slider", "seed_input"}
EXPERIMENT_REGLER = {"delta_slider", "hand_t_slider", "pert_slider", "scale_slider"}


def test_presets_und_kostenwelt_stehen_in_der_seitenleiste_experimentregler_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]
    in_sidebar = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    in_main = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert in_sidebar == WELT_REGLER and in_main == EXPERIMENT_REGLER


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == C.PRESETS[name][key]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for bereich in (at.sidebar, at.main) for art in REGLER_ARTEN for w in getattr(bereich, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, "jeder Eintrag in SETTING_SPECS gehört zu einem gezeichneten Regler"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(spec.hi if regler.value != spec.hi else spec.lo).run()
    assert not at.exception, [str(e) for e in at.exception]
    # das Standard-Preset ist nicht mehr aktiv; aktiv ist höchstens ein anderes Preset, dessen Werte genau den neuen Reglerständen entsprechen
    first = next(iter(C.PRESETS))
    assert _typen(at)[first] != "primary"
    aktiv = [n for n, typ in _typen(at).items() if typ == "primary"]
    for n in aktiv:
        assert all(at.session_state[sk] == C.PRESETS[n][k] for k, sk in P.PRESET_KEYS.items()), n
    if not aktiv:
        assert all(_typen(at)[n] == "secondary" for n in C.PRESETS)
    # erneutes Anklicken markiert wieder (das Preset setzt auch die übrigen Regler zurück)
    at.sidebar.button(key=f"preset_{first}").click().run()
    assert _typen(at)[first] == "primary"
