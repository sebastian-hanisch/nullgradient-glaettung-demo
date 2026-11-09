"""Zufallsstrom für die Gitter-Welt: SplitMix64 mit ganzzahliger Kernrechnung (keine Streamlit-Abhängigkeit, nur numpy).

Herkunft: nach `inverse_generator.SplitMix64` aus `entrollte-netze-demo` kopiert und gekürzt (SplitMix64: Steele, Lea, Flood 2014). Die Ausgabe Nummer k
hängt nur von (Seed, k) ab und damit weder von der Plattform noch von der numpy-Version. Plattformabhängig ist nur der Schritt von Gleichverteilten zu
Normalverteilten (`log` und `cos` der C-Bibliothek, letzte Stelle); die Instanzen des Gitters sind deshalb stetig verteilt und kennen praktisch keine Gleichstände,
trainierte Zahlen werden trotzdem nur als Bänder und Ungleichungen behauptet.

Nie Pythons `hash()` als Seed benutzen (pro Prozess gesalzen, nicht reproduzierbar).

Kopie aus `spo-kuerzester-weg-demo/skw_rng.py` (dort Modul `skw_*`), nur umbenannt (`skw_` zu `ngl_`); die Rechnung ist unverändert, die Tests in `tests/test_rng.py` prüfen sie hier erneut.
"""

import numpy as np

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom: `u64` liefert 64-Bit-Ganzzahlen, `uniform` Gleichverteilte in [0, 1), `normal` Standardnormalverteilte."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)


def derive_seed(seed: int, tag: int) -> int:
    """Unabhängiger Teilstrom: der erste Ausgabewert eines SplitMix64, dessen Zustand aus (Seed, Etikett) gemischt ist.

    Damit bekommen Gewichtsmatrix, Trainingsdaten und Testdaten einer Welt getrennte Ströme aus einem einzigen Zufalls-Seed."""
    mixed = (int(seed) * _GAMMA + int(tag) * _M1 + 0x1234567) & _MASK
    return int(SplitMix64(mixed).u64(1)[0])
