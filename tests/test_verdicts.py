"""Verdikte (reine Funktionen in ngl_evaluation): Schwellen und jede Zahl im Text an handgesetzten Werten."""

import pytest

import ngl_evaluation as ev


def test_de_uses_decimal_comma_and_no_negative_zero():
    assert ev.de(1.234) == "1,2" and ev.de(-0.04) == "0,0" and ev.de(0.26894, 4) == "0,2689" and ev.de(12.0, 0) == "12"
    assert ev.t_label(0.001) == "0,001" and ev.t_label(1.0) == "1" and ev.t_label(10.0) == "10" and ev.t_label(0.1) == "0,1"


# --- Handbeispiel -------------------------------------------------------------------------------------------------------------------------------

def test_hand_state_matches_the_hand_calculation():
    h = ev.hand_state(1.0, 1.0)            # ĉ = (4, 3): Δ = 1
    assert h["chosen"] == 2 and h["regret"] == 2.0 and not h["tie"]
    assert h["w1"] == pytest.approx(0.2689, abs=5e-5) and h["w2"] == pytest.approx(0.7311, abs=5e-5)
    assert h["ecost"] == pytest.approx(4.4621, abs=5e-5) and h["smooth_regret"] == pytest.approx(h["ecost"] - 3.0)
    assert h["grad_soft"] == pytest.approx(0.39322, abs=5e-6)


def test_hand_state_at_tie_negative_delta_and_extremes():
    t = ev.hand_state(0.0, 1.0)
    assert t["tie"] and t["chosen"] == 1 and t["regret"] == 0.0 and t["w1"] == 0.5 and t["ecost"] == 4.0
    n = ev.hand_state(-4.0, 0.1)
    assert n["chosen"] == 1 and n["regret"] == 0.0 and n["w1"] == pytest.approx(1.0, abs=1e-12) and 0.0 <= abs(n["grad_soft"]) < 1e-12     # mathematisch positiv, in Gleitkommazahlen hier nicht mehr von null zu unterscheiden
    p = ev.hand_state(4.0, 10.0)
    assert p["chosen"] == 2 and p["w1"] == pytest.approx(1 - 1 / (1 + 2.718281828459045 ** -0.4), abs=1e-9)


def test_hand_curves_step_and_smooth_relation():
    c = ev.hand_curves(1.0)
    assert set(c["hard"].tolist()) == {0.0, 2.0}
    assert (c["hard"][c["x"] <= 0] == 0).all() and (c["hard"][c["x"] > 0] == 2).all()
    assert (c["slope"] >= 0).all() and c["slope"][c["x"] == 0][0] == pytest.approx(0.5)     # nie negativ; bei T = 1 und Δ = 0: 2 σ(0)(1 − σ(0)) = 0,5
    assert c["smooth"][c["x"] == 0][0] == pytest.approx(1.0)           # beim Gleichstand: halbes Gewicht auf dem falschen Weg, Regret 2 * 0,5


def test_verdict_hand_cases_and_numbers():
    kind, text = ev.verdict_hand(ev.hand_state(1.0, 1.0))
    assert kind == "warning" and "Weg 2" in text and "Regret 2,0" in text and "0,2689" in text and "0,7311" in text and "4,4621" in text and "T = 1" in text
    kind, text = ev.verdict_hand(ev.hand_state(-1.0, 3.0))
    assert kind == "success" and "Weg 1, den richtigen: Regret 0,0" in text and "T = 3" in text
    kind, text = ev.verdict_hand(ev.hand_state(0.0, 0.1))
    assert kind == "info" and "Gleichstand" in text and "0,5000" in text and "4,0000" in text


# --- Störung ------------------------------------------------------------------------------------------------------------------------------------

def test_verdict_perturbation_threshold_at_95_percent_of_the_shown_value():
    kind, text = ev.verdict_perturbation(1.0, 0.975, 0.975)
    assert kind == "warning" and "±1,0 %" in text and "97,5 %" in text and "1000 Testfälle" in text and "Differenzenquotient" in text
    assert ev.verdict_perturbation(5.0, 0.9496, 0.9496)[0] == "warning"       # angezeigt 95,0 %
    assert ev.verdict_perturbation(20.0, 0.9494, 0.9494)[0] == "info"         # angezeigt 94,9 %
    kind, text = ev.verdict_perturbation(20.0, 0.62, 0.64)
    assert kind == "info" and "62,0 %" in text and "64,0 %" in text


# --- Kernfrage und Lernen ------------------------------------------------------------------------------------------------------------------------

def test_verdict_core_three_cases():
    kind, text = ev.verdict_core(1.0, 0.976, 1.9, 0.001, 2.1, 1.0, 0.5)
    assert kind == "success" and "97,6 %" in text and "T = 0,001" in text and "2,1 %" in text and "T = 1" in text and "0,5 %" in text and "KQ liegt bei 1,9 %" in text
    assert ev.verdict_core(1.0, 0.976, 1.9, 0.001, 1.0, 1.0, 0.5)[0] == "info"
    kind, text = ev.verdict_core(1.0, 0.976, 0.0, 0.001, 0.15, 1.0, 0.01)
    assert kind == "warning" and "keinen Gewinn" in text


def test_verdict_learning_cases_and_numbers():
    gibbs = {0.001: 15.0, 0.1: 12.0, 1.0: 11.5, 3.0: 11.6}
    kind, text = ev.verdict_learning(13.0, 12.4, gibbs)
    assert kind == "success" and "T = 0,001 endet bei 15,0 % und damit über KQ (13,0 %)" in text and "T = 1 mit 11,5 %" in text and "SPO+: 12,4 %" in text
    kind, text = ev.verdict_learning(13.0, 12.4, {0.001: 13.0, 1.0: 13.2})
    assert kind == "warning" and "so hoch wie KQ" in text and "nicht unter KQ" in text
    kind, text = ev.verdict_learning(13.0, 12.4, {0.001: 12.0})
    assert kind == "info" and "unter KQ" in text
    assert ev.verdict_learning(13.0, 12.4, {0.001: 12.96, 1.0: 12.0})[1].startswith("T = 0,001 endet bei 13,0 %, so hoch wie KQ")     # angezeigte Werte entscheiden


def test_reach_share_counts_cases_with_more_than_one_path_over_one_percent(grid5):
    import numpy as np

    paths = grid5.paths()
    X = np.zeros((4, 5))
    coef = np.zeros((6, 40))
    coef[0, :] = 1.0                                           # alle Bögen gleich: alle 70 Wege gleich gewichtet
    assert ev.reach_share(paths, coef, X, 1.0) == 1.0
    coef[0, :] = np.arange(40) * 10.0                          # sehr verschiedene Bögen: ein Weg hat fast das ganze Gewicht
    assert ev.reach_share(paths, coef, X, 0.5) == 0.0


# --- Kostenmaßstab -------------------------------------------------------------------------------------------------------------------------------

def test_best_temperature_prefers_the_smaller_on_equal_shown_values():
    assert ev.best_temperature({0.1: 12.04, 1.0: 11.84, 10.0: 12.4}) == (1.0, 11.84)
    assert ev.best_temperature({0.1: 12.04, 1.0: 11.96}) == (0.1, 12.04)           # beide zeigen 12,0
    assert ev.best_temperature({0.1: 12.0, 1.0: 12.01}) == (0.1, 12.0)         # beide zeigen 12,0


def test_verdict_scale_ratio_equals_factor_or_not():
    base = {0.1: 12.0, 1.0: 11.5, 10.0: 12.0}
    kind, text = ev.verdict_scale(10, base, {1.0: 12.0, 10.0: 11.5, 100.0: 12.0}, 13.0)
    assert kind == "success" and "T = 1 (11,5 %)" in text and "T = 10 (11,5 %)" in text and "Faktor 10" in text and "KQ liegt in beiden Fällen bei 13,0 %" in text
    kind, text = ev.verdict_scale(10, base, {1.0: 11.4, 10.0: 11.5, 100.0: 12.0}, 13.0)
    assert kind == "info" and "T = 1 (11,4 %)" in text and "weicht vom Kostenfaktor 10 ab" in text


# --- Bänder -------------------------------------------------------------------------------------------------------------------------------------

def test_band_rows_and_share_band_read_the_measurement():
    rows = ev.band_rows(6, 1000, 0.0)
    assert [r["Verfahren"] for r in rows][:3] == ["Orakel", "KQ", "SPO+ (300 Schritte)"] and len(rows) == 3 + 6
    assert rows[-1]["Verfahren"] == "geglättet, T = 3" and rows[0]["seeds"] == 20
    sb = ev.share_band(6, 1000, 0.0, 1.0)
    assert 0.9 < sb["weg"]["median"] <= 1.0 and sb["seeds"] == 20 and sb["weg"]["min"] <= sb["weg"]["median"] <= sb["weg"]["max"]
    assert ev.band_str({"median": 1.96, "min": 1.38, "max": 3.0}) == "2,0 [1,4; 3,0]"
