"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button. Schnellstart und Kostenwelt stehen in der Seitenleiste, experimentbezogene Regler im Hauptbereich beim Experiment.

Herkunft: Aufbau aus `spo-kuerzester-weg-demo/skw_presets.py` übernommen; die Regler sind die dieser Demo."""

import random
from dataclasses import dataclass

import streamlit as st

import ngl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # bei Auswahlreglern: erlaubte Werte; ein Wert aus der Adresszeile rastet auf den nächsten ein
    step: float | None = None      # bei Schiebereglern mit Dezimalwerten: Rasterweite


SETTING_SPECS = {
    "deg_slider": SettingSpec("deg", int, C.DEFAULT_DEG, min(C.DEG_CHOICES), max(C.DEG_CHOICES), C.DEG_CHOICES),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, min(C.NOISE_CHOICES), max(C.NOISE_CHOICES), C.NOISE_CHOICES),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, min(C.N_CHOICES), max(C.N_CHOICES), C.N_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "delta_slider": SettingSpec("delta", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX, step=C.DELTA_STEP),
    "hand_t_slider": SettingSpec("ht", float, C.DEFAULT_HAND_T, min(C.HAND_T_CHOICES), max(C.HAND_T_CHOICES), C.HAND_T_CHOICES),
    "pert_slider": SettingSpec("pert", float, C.DEFAULT_PERT, min(C.PERT_CHOICES), max(C.PERT_CHOICES), C.PERT_CHOICES),
    "scale_slider": SettingSpec("scale", int, C.DEFAULT_SCALE, min(C.SCALE_CHOICES), max(C.SCALE_CHOICES), C.SCALE_CHOICES),
}
PRESET_KEYS = {"deg": "deg_slider", "noise": "noise_slider", "n": "n_slider", "seed": "seed_input"}


def snap(spec: SettingSpec, value):
    """Wert auf die Reglergrenzen begrenzen und bei Auswahlreglern auf den nächsten erlaubten Wert einrasten (bei Gleichabstand der kleinere)."""
    if spec.choices:
        return min(spec.choices, key=lambda c: (abs(c - value), c))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step:
        value = round(round(value / spec.step) * spec.step, 6)
    return spec.caster(value)


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if value != value:   # NaN
                continue
            st.session_state[state_key] = snap(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt ALLE Regler: die Werte des Presets, die übrigen (Experimentregler) auf ihre Standardwerte; so ist das Preset danach genau dann aktiv."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(preset_key_of.get(state_key), spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Die Experimentregler gehören dazu: ändert jemand sie, ist kein Preset mehr aktiv."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
