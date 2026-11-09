"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusatztests dieser Demo (unter der Vorlage) ---------------------------------------------------------------------------------------------
# Die Vorlage prüft nur `st.slider`; hier zusätzlich Auswahlregler (`select_slider`) und Zahlenfeld, Permalink, Verdikte je Preset, größte Welt.



# --- Zusatztests dieser Demo (unter der Vorlage) ---------------------------------------------------------------------------------------------
# Die Vorlage prüft nur `st.slider`; hier zusätzlich Auswahlregler (`select_slider`) und Zahlenfeld, Permalink, Verdikte je Preset, Experimente.

import pytest  # noqa: E402

import ngl_constants as C  # noqa: E402
import ngl_presets as P  # noqa: E402


def test_every_slider_select_slider_and_number_input_at_min_and_max_does_not_raise():
    for state_key, spec in P.SETTING_SPECS.items():
        for edge, value in (("min", spec.lo), ("max", spec.hi)):
            at = _fresh_app()
            found = False
            for art in ("slider", "select_slider", "number_input"):
                try:
                    widget = getattr(at, art)(key=state_key)
                except KeyError:
                    continue
                widget.set_value(value).run(timeout=TIMEOUT)
                found = True
                break
            assert found, state_key
            assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


def test_largest_world_and_largest_cost_factor_finish_without_error():
    """Größte Welt (n = 1000, Grad 6, Rauschstufe 0,5) und Kostenfaktor 100: Timeout statt Zeit-Assertion."""
    at = _fresh_app()
    at.select_slider(key="noise_slider").set_value(0.5)
    at.select_slider(key="scale_slider").set_value(100)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_snaps_values_to_the_allowed_choices():
    at = AppTest.from_file(APP_PATH)
    at.query_params["deg"] = "3"
    at.query_params["noise"] = "0.3"
    at.query_params["n"] = "10"
    at.query_params["seed"] = "-4"
    at.query_params["delta"] = "9.9"
    at.query_params["ht"] = "2"
    at.query_params["pert"] = "abc"
    at.query_params["scale"] = "50"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    assert (s["deg_slider"], s["noise_slider"], s["n_slider"], s["seed_input"], s["delta_slider"], s["hand_t_slider"], s["pert_slider"], s["scale_slider"]) == (
        2, 0.5, 100, 0, C.DELTA_MAX, 1.0, C.DEFAULT_PERT, 10)


def test_permalink_roundtrip_writes_the_settings_to_the_address():
    at = _fresh_app()
    at.select_slider(key="deg_slider").set_value(4).run(timeout=TIMEOUT)
    assert dict(at.query_params) == {"deg": "4", "noise": "0.0", "n": "1000", "seed": "0", "delta": "1.0", "ht": "1.0", "pert": "1.0", "scale": "10"}


@pytest.mark.parametrize("name,expected_kind,fragment", [
    ("Nichtlineare Wahrheit, ohne Rauschen (Voreinstellung)", "success", "keinen Gewinn gegen KQ, die mittlere schon"),
    ("Nichtlineare Wahrheit mit Rauschen", "success", "keinen Gewinn gegen KQ, die mittlere schon"),
    ("Lineare Wahrheit: nichts zu glätten", "warning", "keinen Gewinn gegen KQ"),
    ("Wenig Daten mit Rauschen", "warning", "keinen Gewinn gegen KQ"),
])
def test_every_preset_shows_the_expected_core_question_verdict(name, expected_kind, fragment):
    at = _fresh_app()
    at.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    shown = {"success": at.success, "info": at.info, "warning": at.warning}[expected_kind]
    assert any(fragment in e.value for e in shown), (name, [e.value for e in list(at.success) + list(at.info) + list(at.warning)])


def test_all_four_experiments_the_bridge_the_boundaries_and_the_formulas_are_there():
    at = _fresh_app()
    headings = [s.value for s in at.subheader]
    for i in range(1, 5):
        assert any(h.startswith(f"🔬 Experiment {i}:") for h in headings), i
    assert not any(h.startswith("🔬 Experiment 5:") for h in headings)
    assert "🚧 Wo die Annahmen enden" in headings and "🔗 Brücke: die Ableitung am Ergebnis" in headings
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any(e.label.startswith("So funktioniert") for e in at.expander)


def test_hand_example_default_shows_the_numbers_of_the_plan():
    """Voreinstellung Δ = 1, T = 1: Gewichte 0,2689 / 0,7311 und erwartete Kosten 4,4621 (Handrechnung in tests/test_gibbs.py)."""
    at = _fresh_app()
    values = {m.label: m.value for m in at.metric}
    assert values["Gewichte Weg 1 / Weg 2"] == "0,2689 / 0,7311"
    assert values["Erwartete Kosten"] == "4,4621"
    assert values["Harter Löser wählt"] == "Weg 2" and values["Steigung des harten Regrets"] == "0"


def test_hand_example_tie_and_correct_path():
    at = _fresh_app()
    at.slider(key="delta_slider").set_value(0.0).run(timeout=TIMEOUT)
    assert {m.label: m.value for m in at.metric}["Steigung des harten Regrets"] == "springt"
    assert any("Gleichstand" in e.value for e in at.info)
    at.slider(key="delta_slider").set_value(-2.0).run(timeout=TIMEOUT)
    assert any("richtigen: Regret 0,0" in e.value for e in at.success)
