"""PDF-Export: erzeugt ein gültiges PDF für jedes Preset (Umlaute und Sonderzeichen brechen die Kernschrift nicht)."""

import pytest

import ngl_constants as C
import ngl_evaluation as ev
from ngl_pdf_export import _s, build_pdf
from ngl_runs import run_world


@pytest.mark.parametrize("name", ["Lineare Wahrheit: nichts zu glätten", "Wenig Daten mit Rauschen"])
def test_pdf_is_built_for_presets(name):
    p = C.PRESETS[name]
    s = ev.Settings(deg=p["deg"], noise=p["noise"], n=p["n"], seed=p["seed"])
    run = run_world(s.run_spec(), snapshot_steps=(), keep_cases=False)
    pdf = build_pdf(s, run, ev.world_shares(run))
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


def test_special_characters_are_replaced_by_latin1_safe_text():
    assert _s("ĉ₁ − ĉ₂ → 3 – Δ ±1") == "c^1 - c^2 -> 3 - Delta +-1"
    assert _s("Löser für Gitter ß") == "Löser für Gitter ß"
