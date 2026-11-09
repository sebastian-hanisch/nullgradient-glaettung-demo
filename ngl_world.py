"""Kostenwelt für die Gitter-Welt: Merkmale, wahre Kosten und Rauschen (keine Streamlit-Abhängigkeit, nur numpy).

Schema (aus dem Gedächtnis nach dem Aufbau der SPO-Literatur nachgebaut, NICHT gegen den Primärtext geprüft; die Zahlen dieser Demo sind Messungen an
diesem Aufbau, keine Literaturwerte):

    Merkmale x ~ N(0, I_p),  Gewichtsmatrix B* (d x p) mit unabhängigen Einträgen aus {0, 1} (je 1/2),
    mittlere Kosten  m_j(x) = ((B* x)_j / sqrt(p) + 3)^deg / 3,5^deg + 1,
    beobachtete Kosten  c_j = m_j(x) * eps_j,  eps_j ~ U[1 - e, 1 + e] unabhängig.

deg = 1 ist eine lineare Wahrheit (ein lineares Modell kann sie exakt darstellen); je größer deg, desto stärker weicht die Wahrheit von jedem linearen Modell ab.
e ist die Rauschstufe. Bei e = 0 sind beobachtete und mittlere Kosten gleich; das "Orakel" (kennt m(x)) hat dann Regret 0.

Zufall nur über `ngl_rng.SplitMix64`; Gewichtsmatrix, Trainingsdaten und Testdaten bekommen getrennte Teilströme aus einem Zufalls-Seed.

Kopie aus `spo-kuerzester-weg-demo/skw_world.py` (dort Modul `skw_*`), nur umbenannt (`skw_` zu `ngl_`); die Rechnung ist unverändert, die Tests in `tests/test_world.py` prüfen sie hier erneut.
"""

from dataclasses import dataclass

import numpy as np

from ngl_rng import SplitMix64, derive_seed

TAG_WEIGHTS, TAG_TRAIN, TAG_TEST = 1, 2, 3


@dataclass(frozen=True)
class WorldSpec:
    """Eine Kostenwelt: Zahl der Merkmale p, Grad der Wahrheit deg, Rauschstufe e (0 bis 1)."""

    p: int = 5
    deg: int = 6
    noise: float = 0.0


def draw_weights(d: int, spec: WorldSpec, seed: int) -> np.ndarray:
    """Gewichtsmatrix B* (d x p) aus {0, 1}, je Eintrag mit Wahrscheinlichkeit 1/2 eine 1."""
    u = SplitMix64(derive_seed(seed, TAG_WEIGHTS)).uniform(d * spec.p).reshape(d, spec.p)
    return (u < 0.5).astype(float)


def mean_costs(B: np.ndarray, X: np.ndarray, spec: WorldSpec) -> np.ndarray:
    """Mittlere Kosten m(x) (n x d): ((B x) / sqrt(p) + 3)^deg / 3,5^deg + 1."""
    base = (X @ B.T) / np.sqrt(spec.p)
    return (base + 3.0) ** spec.deg / 3.5 ** spec.deg + 1.0


def draw_data(B: np.ndarray, spec: WorldSpec, n: int, seed: int, tag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n Fälle aus der Welt: Merkmale X (n x p), beobachtete Kosten C (n x d), mittlere Kosten M (n x d). `tag` wählt den Teilstrom (Training oder Test)."""
    d = B.shape[0]
    rng = SplitMix64(derive_seed(seed, tag))
    X = rng.normal(n * spec.p).reshape(n, spec.p)
    M = mean_costs(B, X, spec)
    if spec.noise > 0:
        eps = 1.0 - spec.noise + 2.0 * spec.noise * rng.uniform(n * d).reshape(n, d)
        C = M * eps
    else:
        C = M.copy()
    return X, C, M


@dataclass(frozen=True)
class Dataset:
    """Trainings- und Testdaten einer Welt zu einem Zufalls-Seed."""

    spec: WorldSpec
    seed: int
    B: np.ndarray
    X_train: np.ndarray
    C_train: np.ndarray
    X_test: np.ndarray
    C_test: np.ndarray
    M_test: np.ndarray  # mittlere Kosten der Testfälle: das Orakel kennt sie


def make_dataset(spec: WorldSpec, d: int, n_train: int, n_test: int, seed: int) -> Dataset:
    B = draw_weights(d, spec, seed)
    X_tr, C_tr, _ = draw_data(B, spec, n_train, seed, TAG_TRAIN)
    X_te, C_te, M_te = draw_data(B, spec, n_test, seed, TAG_TEST)
    return Dataset(spec, seed, B, X_tr, C_tr, X_te, C_te, M_te)
