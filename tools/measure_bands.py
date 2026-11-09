"""Misst die Bänder, die die App als "vorgerechnet" zeigt, und schreibt `ngl_bands.json` (feste Zufalls-Seeds 0 bis 19, Adam 300 Schritte).

    python tools/measure_bands.py [--jobs 2] [--out ngl_bands.json]

Inhalt: (1) vier Zellen (Grad, n, Rauschen): normierter Regret von Orakel, KQ, SPO+ und dem geglätteten Verfahren (Gibbs-Verteilung über alle 70 Wege) für T = 0,001 / 0,03 / 0,1 / 0,3 / 1 / 3,
dazu der Anteil der Testfälle, in denen der Weg nach relativer Störung der KQ-Vorhersage um höchstens delta (0,1 bis 20 %) gleich bleibt; (2) der Kostenmaßstab: Welt Grad 6, n = 1000,
Rauschen 0 und 0,5, Kosten mal 1 und mal 10, Gibbs-Regret für T = 0,001 bis 1000 in Zehnerpotenzen (Lernrate mit dem Faktor skaliert).
Die Zahlen sind trainiert und plattformabhängig; Tests prüfen sie nur als Bänder und Ungleichungen.
Dauer: einige Minuten bis zu einer Stunde je nach Last. Nie mehr als zwei Prozesse (`--jobs`)."""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # ein Thread je Prozess: kleine Matrizen, mehrere Prozesse
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import ngl_constants as C  # noqa: E402
from ngl_runs import RunSpec, cell_result, gibbs_sweep, run_world, unchanged_shares  # noqa: E402
from ngl_training import predict  # noqa: E402

SEEDS = C.BAND_SEEDS
CELLS = ((6, 1000, 0.0), (6, 1000, 0.5), (1, 1000, 0.0), (6, 100, 0.5))
DELTAS = tuple(p / 100.0 for p in C.BAND_PERTS)
SCALE_NOISES = (0.0, 0.5)


def _cell(args):
    deg, n, noise, seed = args
    run = run_world(RunSpec(deg=deg, noise=noise, n=n, seed=seed, temperatures=C.BAND_TEMPS), snapshot_steps=(), keep_cases=False)
    shares = unchanged_shares(run.grid, predict(run.coef_kq, run.data.X_test), run.data.C_test, DELTAS, seed)
    return (deg, n, noise, seed, cell_result(run), shares)


def _scale(args):
    noise, scale, seed = args
    out = gibbs_sweep(RunSpec(deg=6, noise=noise, n=1000, seed=seed, scale=float(scale)), C.SCALE_TEMPS)
    return (noise, scale, seed, out)


def _summary(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "ngl_bands.json"))
    args = ap.parse_args()
    t0 = time.time()
    with Pool(max(1, min(args.jobs, 2))) as pool:
        scale_res = pool.map(_scale, [(e, s, seed) for e in SCALE_NOISES for s in C.BAND_SCALES for seed in SEEDS], chunksize=1)
        cell_res = pool.map(_cell, [(d, n, e, seed) for (d, n, e) in CELLS for seed in SEEDS], chunksize=1)
    gkeys = [f"regret_gibbs_{T:g}" for T in C.BAND_TEMPS]
    out = {
        "meta": {
            "beschreibung": "Eigene Messung dieser Demo (tools/measure_bands.py): lineares Modell, KQ, SPO+ und geglättetes Verfahren (Gibbs über 70 Wege, Adam, Lernrate 0,05, Start bei KQ), 5 x 5 Gitter, 5 Merkmale, 1000 Testfälle, Zufalls-Seeds 0 bis 19.",
            "seeds": list(SEEDS), "steps": C.STEPS, "lr": C.LR, "n_test": C.N_TEST, "temperaturen": list(C.BAND_TEMPS), "skala_temperaturen": list(C.SCALE_TEMPS),
            "stoerungen_prozent": list(C.BAND_PERTS),
            "einheiten": "Regret in Prozent der besten Wegkosten (Summe über die Testfälle); Anteile als Bruchteil 0 bis 1",
            "laufzeit_s": round(time.time() - t0),
        },
        "cells": [],
        "skala": [],
    }
    for (deg, n, e) in CELLS:
        rows = sorted([(r["seed"], r, sh) for (d, nn, ee, s, r, sh) in cell_res if (d, nn, ee) == (deg, n, e)], key=lambda t: t[0])
        assert len(rows) == len(SEEDS)
        keys = ("regret_orakel", "regret_kq", "regret_spo") + tuple(gkeys)
        cell = {"deg": deg, "n": n, "noise": e, **{k: [r[k] for _, r, _ in rows] for k in keys}}
        cell["summary"] = {k: _summary(cell[k]) for k in keys}
        cell["spo_besser_als_kq"] = int(sum(r["regret_spo"] < r["regret_kq"] for _, r, _ in rows))
        cell["gibbs_besser_als_kq"] = {f"{T:g}": int(sum(r[f"regret_gibbs_{T:g}"] < r["regret_kq"] for _, r, _ in rows)) for T in C.BAND_TEMPS}
        cell["anteil_weg"] = {f"{d:g}": [sh[d]["weg"] for _, _, sh in rows] for d in DELTAS}
        cell["anteil_regret"] = {f"{d:g}": [sh[d]["regret"] for _, _, sh in rows] for d in DELTAS}
        out["cells"].append(cell)
    for noise in SCALE_NOISES:
        for scale in C.BAND_SCALES:
            rows = sorted([(s, r) for (e, sc, s, r) in scale_res if e == noise and sc == scale], key=lambda t: t[0])
            assert len(rows) == len(SEEDS)
            gib = {f"{T:g}": [100.0 * r["gibbs"][float(T)] for _, r in rows] for T in C.SCALE_TEMPS}
            out["skala"].append({
                "deg": 6, "n": 1000, "noise": noise, "faktor": scale,
                "regret_kq": [100.0 * r["kq"] for _, r in rows], "regret_gibbs": gib,
                "summary": {k: _summary(v) for k, v in gib.items()}, "summary_kq": _summary([100.0 * r["kq"] for _, r in rows]),
            })
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
