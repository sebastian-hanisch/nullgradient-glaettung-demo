"""Erzeugt die Zahlentabellen der README aus den vorgerechneten Bändern (`ngl_bands.json`).

`tests/test_claims.py` prüft, dass jede dieser Zeilen wörtlich in der README steht: die Zahlen der README sind damit nie von Hand abgeschrieben, und ändert sich eine Messung,
schlägt der Test an. Ausgabe zum Einfügen: `python tests/readme_tables.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ngl_bands as B  # noqa: E402
import ngl_constants as C  # noqa: E402
from ngl_evaluation import band_rows, band_str, de, t_label  # noqa: E402

CELLS = ((6, 1000, 0.0), (6, 1000, 0.5), (1, 1000, 0.0), (6, 100, 0.5))


def cell_table(deg: int, n: int, noise: float) -> list[str]:
    """Regret (Median [Minimum; Maximum], Prozent) von Orakel, KQ, SPO+ und dem geglätteten Verfahren je Temperatur, dazu in wie vielen Seeds besser als KQ."""
    lines = ["| Verfahren | Regret in % (Median [Minimum; Maximum]) | besser als KQ |", "|---|---|---|"]
    for r in band_rows(deg, n, noise):
        better = "–" if r["besser"] is None else f"{r['besser']} von {r['seeds']}"
        lines.append(f"| {r['Verfahren']} | {band_str(r['band'])} | {better} |")
    return lines


def share_table() -> list[str]:
    """Anteil der Testfälle mit gleichem Weg nach zufälliger relativer Störung der KQ-Vorhersage (Median [Minimum; Maximum] über 20 Seeds, Prozent)."""
    head = "| Welt | " + " | ".join(f"±{t_label(p)} %" for p in C.BAND_PERTS) + " |"
    lines = [head, "|---|" + "---|" * len(C.BAND_PERTS)]
    for deg, n, noise in CELLS:
        c = B.cell(deg, n, noise)
        cols = []
        for p in C.BAND_PERTS:
            v = sorted(c["anteil_weg"][f"{p / 100.0:g}"])
            import numpy as np

            cols.append(f"{de(100 * float(np.median(v)))} [{de(100 * v[0])}; {de(100 * v[-1])}]")
        lines.append(f"| Grad {deg}, n = {n}, Rauschstufe {de(noise)} | " + " | ".join(cols) + " |")
    return lines


def scale_table(noise: float) -> list[str]:
    """Regret des geglätteten Verfahrens (Median über 20 Seeds, Prozent) je Temperatur bei Kosten mal 1 und mal 10, dazu KQ."""
    b1, b10 = B.scale_cell(noise, 1), B.scale_cell(noise, 10)
    lines = ["| Temperatur T | Kosten mal 1 | Kosten mal 10 |", "|---|---|---|"]
    for T in C.SCALE_TEMPS:
        k = f"{T:g}"
        lines.append(f"| {t_label(T)} | {band_str(b1['summary'][k])} | {band_str(b10['summary'][k])} |")
    lines.append(f"| KQ (beide gleich) | {band_str(b1['summary_kq'])} | {band_str(b10['summary_kq'])} |")
    return lines


if __name__ == "__main__":
    for deg, n, noise in CELLS:
        print(f"### Grad {deg}, n = {n}, Rauschstufe {de(noise)}")
        print("\n".join(cell_table(deg, n, noise)))
        print()
    print("### Anteil gleicher Wege")
    print("\n".join(share_table()))
    print()
    for noise in (0.0, 0.5):
        print(f"### Kostenmaßstab, Rauschstufe {de(noise)}")
        print("\n".join(scale_table(noise)))
        print()
