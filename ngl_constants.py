"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Das Gitter und die Welt (aus spo-kuerzester-weg-demo übernommen) -----------------------------------------------------------------------
GRID_H = 5                       # 5 x 5 Knoten: 40 Bögen, 70 Wege, je 8 Bögen
N_FEATURES = 5                   # Merkmale je Fall
N_TEST = 1000                    # Testfälle je Welt
STEPS, LR = 300, 0.05            # Adam: Schritte und Lernrate (Start bei der KQ-Lösung)
BAND_SEEDS = tuple(range(20))    # Zufalls-Seeds der vorgerechneten Bänder (ngl_bands.json)

DEG_CHOICES = (1, 2, 4, 6)       # Grad der Wahrheit: 1 = linear
NOISE_CHOICES = (0.0, 0.5)       # Rauschstufe e
N_CHOICES = (100, 300, 1000)     # Trainingsfälle
DEFAULT_DEG, DEFAULT_NOISE, DEFAULT_N, DEFAULT_SEED = 6, 0.0, 1000, 0
SEED_MAX = 999999

# --- Experiment 1: das Handbeispiel (zwei Wege mit je einem Bogen, wahre Kosten (3, 5)) ---------------------------------------------------------
HAND_C = (3.0, 5.0)
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = -4.0, 4.0, 0.25, 1.0    # Vorhersage-Abstand ĉ₁ − ĉ₂ (ĉ = (4, 3) ist Δ = 1)
HAND_T_CHOICES = (0.1, 0.3, 1.0, 3.0, 10.0)                                # Temperaturen des Handbeispiels
DEFAULT_HAND_T = 1.0
HAND_X_RANGE = (-6.0, 6.0)
HAND_POINTS = 241

# --- Experiment 2: Störung der Vorhersage ---------------------------------------------------------------------------------------------------
PERT_CHOICES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)   # relative Störung in Prozent
DEFAULT_PERT = 1.0

# --- Experiment 3: Lernen mit Glättung -----------------------------------------------------------------------------------------------------
TRAIN_TEMPS = (0.001, 0.1, 1.0, 3.0)          # live trainierte Temperaturen
BAND_TEMPS = (0.001, 0.03, 0.1, 0.3, 1.0, 3.0)  # Temperaturen der vorgerechneten Bänder (Tabelle)
CURVE_STEPS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300)

# --- Experiment 4: Temperatur und Kostenmaßstab ---------------------------------------------------------------------------------------------
SCALE_CHOICES = (10, 100)                      # Kostenfaktor gegenüber der Welt (Faktor 1 wird immer mitgezeigt)
DEFAULT_SCALE = 10
SCALE_TEMPS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)   # je eine Zehnerpotenz; Faktor f verschiebt die gute Zone um log10(f) Stufen
BAND_SCALES = (1, 10)                          # Faktoren der vorgerechneten Bänder

# --- Störgrößen für Experiment 2 (vorgerechnete Bänder) --------------------------------------------------------------------------------------
BAND_PERTS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

# --- Anzeige -------------------------------------------------------------------------------------------------------------------------------
COLOR_KQ = "#4c78a8"
COLOR_SPO = "#2F6B65"
COLOR_ORACLE = "#7f7f7f"
COLOR_STEP = "#d62728"
COLOR_SMOOTH = "#b8860b"
COLOR_BEST = "#222222"
EDGE_COLORSCALE = "Viridis"
# Farben je Temperatur (vier live trainierte und die Handbeispiel-Kurve)
TEMP_COLORS = {0.001: "#d62728", 0.1: "#b8860b", 1.0: "#2F6B65", 3.0: "#6a3d9a"}

PRESETS = {
    "Nichtlineare Wahrheit, ohne Rauschen (Voreinstellung)": {"deg": 6, "noise": 0.0, "n": 1000, "seed": 0},
    "Nichtlineare Wahrheit mit Rauschen": {"deg": 6, "noise": 0.5, "n": 1000, "seed": 0},
    "Lineare Wahrheit: nichts zu glätten": {"deg": 1, "noise": 0.0, "n": 1000, "seed": 0},
    "Wenig Daten mit Rauschen": {"deg": 6, "noise": 0.5, "n": 100, "seed": 0},
}
PRESET_HELP = {
    "Nichtlineare Wahrheit, ohne Rauschen (Voreinstellung)": "Grad 6, Rauschstufe 0, 1000 Trainingsfälle, Zufalls-Seed 0: ein lineares Modell passt nicht zur Wahrheit, die Kosten sind aber exakt bekannt; hier zeigt sich der Unterschied zwischen hartem und geglättetem Löser am deutlichsten.",
    "Nichtlineare Wahrheit mit Rauschen": "Grad 6, Rauschstufe 0,5, 1000 Trainingsfälle: ein Teil des Regrets ist unvermeidbar, auch das Orakel hat Regret; die Glättung gewinnt nur einen Teil der Lücke zwischen KQ und Orakel.",
    "Lineare Wahrheit: nichts zu glätten": "Grad 1, Rauschstufe 0: das lineare Modell kann die Wahrheit exakt darstellen, KQ trifft jede Entscheidung; es gibt nichts zu gewinnen.",
    "Wenig Daten mit Rauschen": "Grad 6, Rauschstufe 0,5, nur 100 Trainingsfälle bei 240 Parametern: die Messung dieser Demo zeigt, ob die Glättung hier hilft.",
}
