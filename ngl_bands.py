"""Vorgerechnete Bänder (keine Streamlit-Abhängigkeit): `ngl_bands.json` (eigene Messung, `tools/measure_bands.py`)."""

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def own() -> dict:
    return json.loads((HERE / "ngl_bands.json").read_text(encoding="utf-8"))


def has_cell(deg: int, n: int, noise: float) -> bool:
    return any((c["deg"], c["n"], c["noise"]) == (deg, n, noise) for c in own()["cells"])


def cell(deg: int, n: int, noise: float) -> dict:
    """Eigene Messung einer Zelle (Grad, n, Rauschen): Listen je Seed, Zusammenfassungen, Anteile unveränderter Wege."""
    for c in own()["cells"]:
        if (c["deg"], c["n"], c["noise"]) == (deg, n, noise):
            return c
    raise KeyError((deg, n, noise))


def scale_cell(noise: float, factor: int) -> dict:
    """Messung zum Kostenmaßstab (Grad 6, n = 1000): Gibbs-Regret je Temperatur für Kosten mal `factor`."""
    for c in own()["skala"]:
        if (c["noise"], c["faktor"]) == (noise, factor):
            return c
    raise KeyError((noise, factor))


def has_scale_cell(deg: int, n: int, noise: float) -> bool:
    return (deg, n) == (6, 1000) and any(c["noise"] == noise for c in own()["skala"])
