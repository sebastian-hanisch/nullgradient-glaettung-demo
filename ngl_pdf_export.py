"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from ngl_evaluation import Settings, de, hand_state, learning_rows, t_label, verdict_hand, verdict_learning, verdict_perturbation

_REPLACE = {"→": "->", "–": "-", "—": "-", "ĉ": "c^", "₁": "1", "₂": "2", "Δ": "Delta", "−": "-", "≥": ">=", "≤": "<=", "·": "*", "×": "x", "σ": "sigma", "±": "+-", "²": "^2", "ˆ": "^"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, run, shares: dict) -> bytes:
    """`run`: das `WorldRun` zu den Einstellungen (Live-Lauf einer Instanz); `shares`: {Störung in Prozent: {"weg", "regret"}} dieser Instanz."""
    h = hand_state(settings.delta, settings.hand_t)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Warum der Gradient fehlt: der Löser ist eine Treppe"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Welt: 5 x 5 Gitter, 5 Merkmale, Grad {settings.deg}, Rauschstufe {de(settings.noise)}, {settings.n} Trainingsfälle, Zufalls-Seed {settings.seed}, "
        f"{run.spec.n_test} Testfälle. Alles simuliert; eine Instanz, kein Mittel über Zufalls-Seeds."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Regret auf den Testfällen (normiert, in Prozent der besten Wegkosten)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in learning_rows(run):
        pdf.multi_cell(0, 5, _s(f"{r['Verfahren']}: {de(r['regret'])} %"), new_x="LMARGIN", new_y="NEXT")
    kind, text = verdict_learning(100 * run.regret["kq"], 100 * run.regret["spo"], {T: 100 * r for T, r in run.regret_gibbs.items()})
    pdf.ln(1)
    pdf.multi_cell(0, 5, _s(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Anteil der Testfälle mit gleichem Weg nach zufälliger Störung der KQ-Vorhersage"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for pct in sorted(shares):
        pdf.multi_cell(0, 5, _s(f"Störung bis +-{t_label(pct)} %: Weg gleich in {de(100 * shares[pct]['weg'])} % der Fälle, Regret gleich in {de(100 * shares[pct]['regret'])} %."), new_x="LMARGIN", new_y="NEXT")
    sel = settings.pert
    if sel in shares:
        pdf.ln(1)
        pdf.multi_cell(0, 5, _s(verdict_perturbation(sel, shares[sel]["weg"], shares[sel]["regret"])[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Handbeispiel: zwei Wege"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Wahre Kosten c = (3, 5), Vorhersage-Abstand Delta = {de(settings.delta, 2)}, Temperatur T = {t_label(settings.hand_t)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, _s(verdict_hand(h)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Warum der Gradient fehlt: Gibbs-Glättung des Lösers im Gitter. Quelle: nullgradient-glaettung-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
