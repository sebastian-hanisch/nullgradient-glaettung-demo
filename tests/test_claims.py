"""Claims: jede Zahl der README steht hier als wörtliche Tabellenzeile, Band oder Ungleichung (DEMO-PLAYBOOK §11a).

Zahlen-Tabelle der README -> Test:

| README | Test |
|---|---|
| Tabellen je Zelle, Anteil gleicher Wege, Kostenmaßstab | `test_readme_contains_every_generated_table_line` (wörtlich aus `ngl_bands.json` erzeugt) |
| Treppe: 99,7 / 97,6 / 88,1 / 60,9 % bei ±0,1 / 1 / 5 / 20 % | `test_staircase_sentence` |
| Regret des gleichen Anteils mindestens so groß wie der des gleichen Weges | `test_equal_regret_share_is_never_below_equal_path_share` |
| T = 0,001 bringt im Median keinen Gewinn (9 von 20, 0 von 20, 2,0 gegen 2,0 %, 15,3 gegen 13,1 %, n = 100) | `test_smallest_temperature_learns_nothing` |
| T = 0,03 bis 3: 19 oder 20 von 20 unter KQ, Mediane 0,5 bis 0,6 / 11,6 bis 12,5, Bänder überlappen | `test_middle_temperatures_beat_kq_without_a_ranking` |
| Gleichauf mit SPO+ (0,5 %; 11,9 / 11,7 / 11,6 gegen 12,3) | `test_middle_temperatures_against_spo_plus` |
| Lineare Wahrheit (0 von 20), wenig Daten (kein T unter KQ, T = 1: 5 von 20, SPO+ 10 von 20) | `test_linear_truth_and_small_data` |
| Kostenmaßstab: gute Zone, Verschiebung um den Faktor, Einzel-Seeds | `test_good_zone_moves_with_the_cost_scale`, `test_scaled_runs_equal_unscaled_runs_with_scaled_temperature` |
| Zu großes T verliert (100 und 1000 über KQ) | `test_too_large_temperature_loses` |
| Anfang des Trainings (1,3 / 69,9 / 100 / 100 %) | `test_reach_share_at_the_start_of_the_default_instance` |
| Zahl der Wege 70 / 252 / 924 | `test_number_of_paths` |
| Handbeispiel (0,2689 / 0,7311 / 4,4621 / 4,9999 / 4,0500) | `test_gibbs.py`, `test_verdicts.py` (exakt) |
| Orakel-Abweichungen (1,4e-14 usw.) | nicht im Repo: Scratch-Lauf mit 300 Instanzen; im Repo die kleine Fassung `test_oracle_nullgradient_glaettung.py` |
| Messung dieser Demo (Seeds, Schritte, Lernrate) | `test_band_file_describes_the_documented_setup`, `test_live_run_reproduces_the_bands` |
| Plan-Zahlen in „Befunde und Korrekturen“ (Vormessung) | Zahlen der Vormessung `out_eo_v6_glaettung.txt`, hier nicht nachrechenbar; die neuen Werte stehen in den Tabellen |

Trainierte Zahlen hängen von der Plattform ab: gegen die Bänder wird in Ungleichungen und mit Toleranz geprüft, exakt nur Handwerte und die Tabellenzeilen aus der Messdatei.
"""

import math
from pathlib import Path

import numpy as np
import pytest

import ngl_bands as B
import ngl_constants as C
import readme_tables as T
from ngl_evaluation import de, reach_share
from ngl_grid import Grid
from ngl_runs import RunSpec, run_world

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
MAIN0, MAIN5, LIN, SMALL = B.cell(6, 1000, 0.0), B.cell(6, 1000, 0.5), B.cell(1, 1000, 0.0), B.cell(6, 100, 0.5)
MID = ("0.03", "0.1", "0.3", "1", "3")


def _med(c, key):
    return c["summary"][key]["median"]


def test_readme_contains_every_generated_table_line():
    blocks = [T.cell_table(*cell) for cell in T.CELLS] + [T.share_table(), T.scale_table(0.0), T.scale_table(0.5)]
    missing = [line for lines in blocks for line in lines if line not in README]
    assert not missing, missing[:3]


def test_band_file_describes_the_documented_setup():
    meta = B.own()["meta"]
    assert meta["seeds"] == list(C.BAND_SEEDS) and meta["steps"] == C.STEPS and meta["lr"] == C.LR and meta["n_test"] == C.N_TEST
    assert meta["temperaturen"] == list(C.BAND_TEMPS) and meta["skala_temperaturen"] == list(C.SCALE_TEMPS)
    assert len(B.own()["cells"]) == 4 and all(len(c["regret_kq"]) == 20 for c in B.own()["cells"])
    assert len(B.own()["skala"]) == 4
    assert "Zufalls-Seeds 0 bis 19" in README and "300 Adam-Schritte" in README and "Lernrate 0,05" in README


def test_staircase_sentence():
    def row(c):
        return " / ".join(de(100 * float(np.median(c["anteil_weg"][f"{p / 100.0:g}"]))) for p in (0.1, 1.0, 5.0, 20.0))

    assert f"im Median in {row(MAIN0)} % der 1000 Testfälle derselbe" in README
    assert row(MAIN0) == "99,7 / 97,6 / 88,1 / 60,9" and row(MAIN5) == "99,8 / 97,6 / 88,4 / 60,1"
    assert "Grad 1" in README and min(LIN["anteil_weg"]["0.001"]) == pytest.approx(0.774, abs=0.0005)       # Minimum 77,4 %


def test_equal_regret_share_is_never_below_equal_path_share():
    for c in B.own()["cells"]:
        for key, ws in c["anteil_weg"].items():
            assert all(r >= w - 1e-12 for r, w in zip(c["anteil_regret"][key], ws)), (c["deg"], c["noise"], key)


def test_smallest_temperature_learns_nothing():
    assert MAIN0["gibbs_besser_als_kq"]["0.001"] == 9 and MAIN5["gibbs_besser_als_kq"]["0.001"] == 0 and SMALL["gibbs_besser_als_kq"]["0.001"] == 0
    assert de(_med(MAIN0, "regret_gibbs_0.001")) == de(_med(MAIN0, "regret_kq")) == "2,0"
    assert de(_med(MAIN5, "regret_gibbs_0.001")) == "15,3" and de(_med(MAIN5, "regret_kq")) == "13,1"
    assert de(_med(SMALL, "regret_gibbs_0.001")) == "21,4" and de(_med(SMALL, "regret_kq")) == "14,0"
    assert all(g > k for g, k in zip(MAIN5["regret_gibbs_0.001"], MAIN5["regret_kq"]))                  # in allen 20 Seeds schlechter
    assert all(g > k for g, k in zip(SMALL["regret_gibbs_0.001"], SMALL["regret_kq"]))
    assert MAIN0["summary"]["regret_gibbs_0.001"]["min"] < MAIN0["summary"]["regret_kq"]["min"]       # Spanne 0,9 bis 3,0 gegen 1,4 bis 3,0
    assert (de(MAIN0["summary"]["regret_gibbs_0.001"]["min"]), de(MAIN0["summary"]["regret_gibbs_0.001"]["max"]), de(MAIN0["summary"]["regret_kq"]["min"]),
            de(MAIN0["summary"]["regret_kq"]["max"])) == ("0,9", "3,0", "1,4", "3,0")


def _intervals_overlap(c, keys):
    s = c["summary"]
    return all(s[f"regret_gibbs_{a}"]["min"] <= s[f"regret_gibbs_{b}"]["max"] and s[f"regret_gibbs_{b}"]["min"] <= s[f"regret_gibbs_{a}"]["max"] for a in keys for b in keys)


def test_middle_temperatures_beat_kq_without_a_ranking():
    for c in (MAIN0, MAIN5):
        assert all(c["gibbs_besser_als_kq"][t] >= 19 for t in MID)
        assert _intervals_overlap(c, MID)
    m0 = [_med(MAIN0, f"regret_gibbs_{t}") for t in MID]
    m5 = [_med(MAIN5, f"regret_gibbs_{t}") for t in MID]
    assert (de(min(m0)), de(max(m0))) == ("0,5", "0,6") and (de(min(m5)), de(max(m5))) == ("11,6", "12,5")
    assert [MAIN0["gibbs_besser_als_kq"][t] for t in MID] == [20] * 5 and [MAIN5["gibbs_besser_als_kq"][t] for t in MID] == [19, 19, 20, 20, 20]


def test_middle_temperatures_against_spo_plus():
    assert de(_med(MAIN0, "regret_gibbs_0.3")) == de(_med(MAIN0, "regret_gibbs_1")) == de(_med(MAIN0, "regret_spo")) == "0,5"
    spo5 = _med(MAIN5, "regret_spo")
    assert de(spo5) == "12,3"
    for t, shown in (("0.3", "11,9"), ("1", "11,7"), ("3", "11,6")):
        assert _med(MAIN5, f"regret_gibbs_{t}") < spo5 and de(_med(MAIN5, f"regret_gibbs_{t}")) == shown
    assert _intervals_overlap({"summary": {**{f"regret_gibbs_{t}": MAIN5["summary"][f"regret_gibbs_{t}"] for t in ("0.3", "1", "3")}, "regret_gibbs_spo": MAIN5["summary"]["regret_spo"]}},
                              ("0.3", "1", "3", "spo"))


def test_linear_truth_and_small_data():
    assert all(v == 0 for v in LIN["gibbs_besser_als_kq"].values()) and LIN["spo_besser_als_kq"] == 0
    for t in (0.001,) + tuple(float(x) for x in MID):
        assert _med(SMALL, f"regret_gibbs_{t:g}") > _med(SMALL, "regret_kq")                          # kein T im Median unter KQ
    assert SMALL["gibbs_besser_als_kq"]["1"] == 5 and SMALL["spo_besser_als_kq"] == 10
    assert de(_med(SMALL, "regret_gibbs_1")) == "14,7" and de(_med(SMALL, "regret_spo")) == "14,4" and de(_med(SMALL, "regret_kq")) == "14,0"
    best = min(MID, key=lambda t: _med(SMALL, f"regret_gibbs_{t}"))
    assert best == "1"                                                                                 # „bestes T = 1“ der README (Median 14,66 gegen 14,76 bei T = 3)


def _scale(noise, factor):
    return B.scale_cell(noise, factor)


def test_good_zone_moves_with_the_cost_scale():
    for noise in (0.0, 0.5):
        for factor in (1, 10):
            s = _scale(noise, factor)
            below = [T for T in C.SCALE_TEMPS if s["summary"][f"{T:g}"]["median"] < s["summary_kq"]["median"]]
            expected = {(0.0, 1): [0.01, 0.1, 1.0, 10.0], (0.5, 1): [0.01, 0.1, 1.0, 10.0], (0.0, 10): [0.01, 0.1, 1.0, 10.0, 100.0], (0.5, 10): [0.1, 1.0, 10.0, 100.0]}[(noise, factor)]
            assert below == expected, (noise, factor, below)
    # ohne Rauschen, Faktor 10, T = 0,01: 1,9 gegen 2,0 (im Median praktisch gleich)
    s = _scale(0.0, 10)
    assert de(s["summary"]["0.01"]["median"]) == "1,9" and de(s["summary_kq"]["median"]) == "2,0"
    # bei Faktor 1 ohne Rauschen ist T = 0,001 nicht unter KQ (2,02 gegen 1,96)
    assert _scale(0.0, 1)["summary"]["0.001"]["median"] > _scale(0.0, 1)["summary_kq"]["median"]
    # Kleinstes T bei Kosten mal 10 gegen T/10 bei Kosten mal 1: 0,7 gegen 0,7 und 12,1 gegen 12,2
    assert (de(_scale(0.0, 10)["summary"]["0.1"]["median"]), de(_scale(0.0, 1)["summary"]["0.01"]["median"])) == ("0,7", "0,7")
    assert (de(_scale(0.5, 10)["summary"]["0.1"]["median"]), de(_scale(0.5, 1)["summary"]["0.01"]["median"])) == ("12,1", "12,2")


def test_scaled_runs_equal_unscaled_runs_with_scaled_temperature():
    """Je Seed: Kosten mal 10 bei T gleich Kosten mal 1 bei T/10, für T = 1, 10, 100, 1000 (README: höchstens 1e-4 Prozentpunkte Unterschied)."""
    for noise in (0.0, 0.5):
        a, b = _scale(noise, 1), _scale(noise, 10)
        for t10, t1 in ((1.0, 0.1), (10.0, 1.0), (100.0, 10.0), (1000.0, 100.0)):
            x, y = np.array(b["regret_gibbs"][f"{t10:g}"]), np.array(a["regret_gibbs"][f"{t1:g}"])
            assert np.abs(x - y).max() < 1e-4, (noise, t10, np.abs(x - y).max())


def test_too_large_temperature_loses():
    for noise in (0.0, 0.5):
        s = _scale(noise, 1)
        assert s["summary"]["100"]["median"] > s["summary_kq"]["median"] and s["summary"]["1000"]["median"] > s["summary_kq"]["median"]
    s5 = _scale(0.5, 1)
    assert (de(s5["summary"]["100"]["median"]), de(s5["summary"]["1000"]["median"]), de(s5["summary_kq"]["median"])) == ("16,5", "17,9", "13,1")


def test_number_of_paths():
    assert [math.comb(2 * (h - 1), h - 1) for h in (5, 6, 7)] == [70, 252, 924]
    assert Grid(5).paths().shape[0] == 70


def test_reach_share_at_the_start_of_the_default_instance():
    """Voreinstellung (Seed 0): Anteil der Trainingsfälle am KQ-Start, in denen das größte Gibbs-Gewicht unter 99 % liegt (ohne Training: nur KQ und Gewichte; Bänder wegen Rundung)."""
    run = run_world(RunSpec(deg=6, noise=0.0, n=1000, seed=0, steps=1), snapshot_steps=(), keep_cases=False)
    P = run.grid.paths()
    shares = [100 * reach_share(P, run.coef_kq, run.data.X_train, t) for t in C.TRAIN_TEMPS]
    assert 0.5 < shares[0] < 3.0 and 60.0 < shares[1] < 80.0 and shares[2] >= 99.0 and shares[3] >= 99.0
    assert (de(shares[0]), de(shares[1])) == ("1,3", "69,9")                                         # README, mit Rundungsspielraum geprüft über das Band oben


def test_live_run_reproduces_the_bands():
    """Zwei Seeds von Hand nachgerechnet gegen die Bänder: Voreinstellung (Seed 0 und 1) liegt für KQ, SPO+ und T = 1 innerhalb von Minimum und Maximum der 20 Seeds der Messung (Seed 0 und 1 gehören dazu: gleiche Werte bis auf Rundung)."""
    for seed in (0, 1):
        run = run_world(RunSpec(deg=6, noise=0.0, n=1000, seed=seed, temperatures=(1.0,)), snapshot_steps=(), keep_cases=False)
        assert 100 * run.regret["kq"] == pytest.approx(MAIN0["regret_kq"][seed], abs=0.05)
        assert 100 * run.regret["spo"] == pytest.approx(MAIN0["regret_spo"][seed], abs=0.05)
        assert 100 * run.regret_gibbs[1.0] == pytest.approx(MAIN0["regret_gibbs_1"][seed], abs=0.05)


def test_hand_example_text_names_exact_versus_rounded_weights():
    """README und App rechnen 4,4621 nur mit den exakten Gewichten; die gerundeten ergäben 4,4622 (siehe test_gibbs.py)."""
    app = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "4,4622" in README and "exakten Gewichten" in README
    assert "4{,}4622" in app and "exakten Gewichten" in app
    assert r"3 \cdot 0{,}2689 + 5 \cdot 0{,}7311 = 4{,}4621" not in app
