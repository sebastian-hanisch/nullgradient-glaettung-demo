# Warum der Gradient fehlt: der Löser ist eine Treppe – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-nullgradient-glaettung-demo.streamlit.app/)** -->

Stück der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es baut auf `spo-kuerzester-weg-demo` auf und benutzt dasselbe Vehikel: ein abstraktes **5 × 5 Gitter** (Wege von links oben nach rechts unten, 40 Bögen, 70 Wege), in dem ein lineares Modell aus Merkmalen die Bogenkosten vorhersagt und ein exakter Löser den kürzesten Weg wählt. Gitter, Löser, Kostenwelt, Zufallsstrom (SplitMix64) und das SPO+-Training sind von dort **kopiert und umbenannt** (Herkunft im Kopf jedes Moduls), nicht importiert. Neu ist die **Glättung des Lösers**: statt des einen billigsten Weges eine Gibbs-Verteilung über alle 70 Wege.

## Warum dieses Problem

Wer ein Modell auf die **Entscheidung** trainieren will, braucht einen Gradienten des Regrets nach der Vorhersage. Der Regret des vom Löser gewählten Weges ist aber eine **Treppe**: er ändert sich nur, wenn sich der gewählte Weg ändert, und ist dazwischen konstant. Die Kernfrage der Demo: **Wie oft liegt man auf einer Stufe, und was ersetzt den fehlenden Gradienten?**

- **Die Treppe ist bei Grad 6 breit, bei Grad 1 schmaler:** Bei zufälliger relativer Störung der KQ-Vorhersage um höchstens ±0,1 / 1 / 5 / 20 % bleibt der gewählte Weg im Median in 99,7 / 97,6 / 88,1 / 60,9 % der 1000 Testfälle derselbe (Grad 6, n = 1000, ohne Rauschen, 20 Zufalls-Seeds; bei Grad 1 bleibt der Weg bei ±20 % nur in 11,2 % der Fälle derselbe). Dort ist der Regret konstant, ein Differenzenquotient ist null.
- **Glättung gibt dem Lernen eine Richtung:** Mit der Gibbs-Verteilung über alle Wege (Temperatur T) liegt der Regret nach 300 Adam-Schritten bei Grad 6, n = 1000 für T = 0,03 bis 3 in 19 von 20 oder 20 von 20 Seeds unter dem von Kleinsten Quadraten (KQ), ohne Rauschen und mit Rauschen. Bei T = 0,001 nicht: ohne Rauschen im Median so hoch wie KQ (2,0 gegen 2,0 %), mit Rauschen in keinem der 20 Seeds besser (15,3 gegen 13,1 %).
- **Auch zu viel Glättung verliert:** Bei Kosten mit 5. bis 95. Perzentil 1,0 bis 3,9 (mittlere Kosten der Testfälle, Grad 6) liegt der Regret für T = 100 und T = 1000 über dem von KQ (Tabellen unten). Die Aussage „T muss nur nicht null sein“ aus dem Plan gilt deshalb nicht.
- **T gehört zum Kostenmaßstab:** Mit allen Kosten mal 10 (und der Lernrate mal 10) liegt dieselbe Kurve um eine Zehnerpotenz nach rechts verschoben; bei Temperaturen ab 1 stimmen die Werte mit denen der Welt mal 1 bei T geteilt durch 10 überein.
- **Es gilt nicht überall:** Bei linearer Wahrheit gibt es nichts zu gewinnen (Grad 1), und bei wenig Daten (n = 100, Rauschstufe 0,5) liegt kein T im Median unter KQ.

## Modell

**Welt** (aus `spo-kuerzester-weg-demo` übernommen, dort aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft): Merkmale $x \sim \mathcal N(0, I_5)$; eine Gewichtsmatrix $B \in \{0,1\}^{40 \times 5}$ mit fairen Münzwürfen; mittlere Kosten $m_j(x) = ((Bx)_j/\sqrt{5} + 3)^{\deg} / 3{,}5^{\deg} + 1$; beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$. Grad 1 ist linear in $x$, Grad 2, 4 und 6 weichen zunehmend davon ab; $e$ ist die Rauschstufe (0 oder 0,5). Das **Orakel** kennt $m(x)$.

**Harter Löser.** Rückwärtsrechnung über die 25 Knoten des azyklischen Gitters; bei Gleichstand gewinnt am Knoten „rechts“ vor „unten“. Regret des gewählten Weges $w(\hat c)$: $c^\top w(\hat c) - z^*(c)$ mit $z^*(c) = \min_w c^\top w$. Er ist in $\hat c$ stückweise konstant.

**Gibbs-Glättung.** Jeder Weg $w$ bekommt das Gewicht $\pi_T(w) = \exp(-\hat c^\top w / T) / \sum_{w'} \exp(-\hat c^\top w' / T)$; der geglättete Regret ist $\sum_w \pi_T(w)\, c^\top w - z^*(c)$. Er ist in $\hat c$ überall glatt, sein Gradient ist die Kovarianz von Wegkosten und Weg unter $\pi_T$, geteilt durch $-T$. Für $T \to 0$ wird $\pi_T$ der harte Löser, für $T \to \infty$ die Gleichverteilung über die Wege. $T$ hat die Einheit der Kosten. Gerechnet wird über **Aufzählung der 70 Wege** mit log-sum-exp (stabil bei kleinem $T$).

**Zweite Rechenweise (Orakel).** Auf dem azyklischen Gitter lässt sich dieselbe Verteilung ohne Wegliste durch **Summe-Produkt** (Vorwärts-Rückwärts) bestimmen: Randverteilung je Bogen $\mathbb E[w_e]$ und $\log Z$. Die erwarteten Kosten sind $c^\top \mathbb E[w]$. Das ist nur der Wert; die hier benutzte Kovarianzformel des Gradienten braucht zusätzlich gemeinsame Randverteilungen zweier Bögen, die die Demo nicht rechnet (ohne sie ginge es über die Ableitung der Summe-Produkt-Rechnung, die die Demo ebenfalls nicht baut). Die Demo benutzt diese Rechnung nur in den Tests.

**Handbeispiel.** Zwei Wege mit je einem Bogen, $c = (3, 5)$, Vorhersage $\hat c = (4, 3)$: der Löser wählt Weg 2 (Regret 2), bei jeder Änderung von $\hat c$, die die Reihenfolge nicht umkehrt, bleibt der Regret 2. Geglättet ist das Gewicht von Weg 2 gleich $1/(1 + e^{-1/T})$: bei $T = 0{,}1$ ungefähr $1{,}0000$ (erwartete Kosten $4{,}9999$), bei $T = 1$ gleich $0{,}7311$ (Weg 1: $0{,}2689$, erwartete Kosten $4{,}4621$), bei $T = 10$ gleich $0{,}5250$ (erwartete Kosten $4{,}0500$). Der geglättete Regret ist $2\,\sigma(\Delta/T)$ mit $\Delta = \hat c_1 - \hat c_2$; seine Steigung nach $\hat c_1$ ist $\frac{2}{T}\sigma'(\Delta/T)$, überall positiv.

**Training.** Lineares Modell mit Achsenabschnitt (240 Parameter), Start bei der KQ-Lösung, Adam (Lernrate 0,05, Vollbatch, 300 Schritte). Bei SPO+ (Vergleichsspalte, unverändert aus `spo-kuerzester-weg-demo`) gilt der Parameter mit dem kleinsten Trainingsverlust unter den besuchten, beim geglätteten Training der Parameter **nach dem letzten Schritt**. Gemessen wird immer mit dem **harten** Löser auf 1000 Testfällen: normierter Regret = Summe der Regrets geteilt durch die Summe der besten Wegkosten.

## Methodik

Vier Experimente mit live berechnetem Verdikt: (1) **die Treppe von Hand** (zwei Wege, Regler für den Vorhersage-Abstand $\Delta$ und für $T$; Regret und Steigung, hart und geglättet); (2) **Anteil unveränderter Wege** (Regler für die Störungsgröße; Balken für sieben Größen, dazu die vorgerechneten Mediane); (3) **Lernen mit Glättung** (Regret auf den Testfällen nach t Schritten für KQ, SPO+ und $T$ = 0,001 / 0,1 / 1 / 3, dazu der Anteil der Trainingsfälle, in denen die Verteilung am Start mehr als einen Weg nennenswert gewichtet); (4) **Temperatur und Kostenmaßstab** (Regret gegen $T$ für Kosten mal 1 und mal 10 oder 100). Dazu eine Brücke zur Ableitung am Ergebnis. Die Seitenleiste enthält den Schnellstart (vier Szenarien) und die Kostenwelt (Grad, Rauschstufe, n, Zufalls-Seed); die Adresszeile spiegelt die Konfiguration. Live gerechnet wird je Welt: KQ, SPO+ und vier geglättete Trainings (ein paar Sekunden beim ersten Laden); die vorgerechneten Bänder (20 Zufalls-Seeds, `ngl_bands.json`) kommen aus `tools/measure_bands.py` und gelten nur für die vier Zellen der Schnellstart-Szenarien (und, in Experiment 4, für Grad 6, n = 1000).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `ngl_bands.json` (eigene Messung dieser Demo, Zufalls-Seeds 0 bis 19, 300 Adam-Schritte, Lernrate 0,05, 1000 Testfälle; die Tabellen sind wörtlich aus der Datei erzeugt und in `tests/test_claims.py` geprüft) oder aus dem Live-Lauf der Voreinstellung (Zufalls-Seed 0, Band im Test). Regret in Prozent der besten Wegkosten, Median [Minimum; Maximum] über die 20 Seeds. Die Zahlen sind trainiert und plattformabhängig; die Tests prüfen Bänder und Ungleichungen, die Handwerte exakt.

### Wie oft ändert eine kleine Störung den Weg? (Experiment 2)

Anteil der Testfälle mit gleichem gewählten Weg, wenn jeder Eintrag der KQ-Vorhersage um einen zufälligen Bruchteil bis zur genannten Größe verändert wird (Median [Minimum; Maximum] über 20 Seeds, Prozent):

| Welt | ±0,1 % | ±0,5 % | ±1 % | ±2 % | ±5 % | ±10 % | ±20 % |
|---|---|---|---|---|---|---|---|
| Grad 6, n = 1000, Rauschstufe 0,0 | 99,7 [99,5; 100,0] | 98,8 [96,9; 99,4] | 97,6 [95,1; 98,4] | 95,3 [90,8; 96,2] | 88,1 [78,6; 90,7] | 78,2 [65,3; 81,6] | 60,9 [49,3; 66,2] |
| Grad 6, n = 1000, Rauschstufe 0,5 | 99,8 [99,4; 100,0] | 98,8 [97,9; 99,5] | 97,6 [95,5; 98,6] | 95,3 [92,2; 96,6] | 88,4 [80,6; 90,5] | 78,2 [67,6; 81,9] | 60,1 [50,8; 65,4] |
| Grad 1, n = 1000, Rauschstufe 0,0 | 98,4 [77,4; 99,4] | 93,0 [73,7; 95,3] | 86,5 [68,9; 89,7] | 75,2 [60,6; 80,7] | 50,3 [40,7; 59,5] | 27,8 [22,9; 35,6] | 11,2 [9,5; 16,0] |
| Grad 6, n = 100, Rauschstufe 0,5 | 99,7 [99,5; 100,0] | 98,8 [98,1; 99,6] | 97,8 [95,8; 98,7] | 95,8 [91,6; 97,3] | 89,3 [84,1; 92,6] | 79,8 [72,3; 84,6] | 63,1 [51,1; 71,7] |

Die Störung ist je Eintrag eine eigene Zufallszahl (gleichverteilt in $[-1, 1]$, mal der Größe), dieselbe Richtung für alle Größen. Der Anteil mit gleichem **Regret** ist in jedem Seed und bei jeder Größe mindestens so groß wie der mit gleichem Weg (gleicher Weg heißt gleicher Regret; die Umkehrung gilt nur bei Wegen mit gleichen wahren Kosten; ein Beispiel dafür sind ohne Rauschen zwei Bögen mit gleicher Zeile in $B$, als Erklärung des Unterschieds bei Grad 1 nicht geprüft). Bei Grad 1 ist der Anteil deutlich kleiner und streut stark über die Seeds (Minimum 77,4 % bei ±0,1 %); eine Erklärung dafür ist nicht gemessen.

### Regret nach 300 Schritten, je Zelle (Experiment 3)

**Grad 6, n = 1000, Rauschstufe 0,0:**

| Verfahren | Regret in % (Median [Minimum; Maximum]) | besser als KQ |
|---|---|---|
| Orakel | 0,0 [0,0; 0,0] | – |
| KQ | 2,0 [1,4; 3,0] | – |
| SPO+ (300 Schritte) | 0,5 [0,4; 0,8] | 20 von 20 |
| geglättet, T = 0,001 | 2,0 [0,9; 3,0] | 9 von 20 |
| geglättet, T = 0,03 | 0,5 [0,4; 0,8] | 20 von 20 |
| geglättet, T = 0,1 | 0,5 [0,3; 0,8] | 20 von 20 |
| geglättet, T = 0,3 | 0,5 [0,3; 0,7] | 20 von 20 |
| geglättet, T = 1 | 0,5 [0,3; 0,7] | 20 von 20 |
| geglättet, T = 3 | 0,6 [0,4; 0,9] | 20 von 20 |

**Grad 6, n = 1000, Rauschstufe 0,5:**

| Verfahren | Regret in % (Median [Minimum; Maximum]) | besser als KQ |
|---|---|---|
| Orakel | 10,5 [9,3; 12,3] | – |
| KQ | 13,1 [11,3; 14,5] | – |
| SPO+ (300 Schritte) | 12,3 [10,7; 13,4] | 20 von 20 |
| geglättet, T = 0,001 | 15,3 [13,1; 17,1] | 0 von 20 |
| geglättet, T = 0,03 | 12,5 [10,6; 14,4] | 19 von 20 |
| geglättet, T = 0,1 | 12,1 [10,7; 13,7] | 19 von 20 |
| geglättet, T = 0,3 | 11,9 [10,5; 13,0] | 20 von 20 |
| geglättet, T = 1 | 11,7 [10,6; 13,5] | 20 von 20 |
| geglättet, T = 3 | 11,6 [10,5; 13,3] | 20 von 20 |

**Grad 1, n = 1000, Rauschstufe 0,0 (lineare Wahrheit):**

| Verfahren | Regret in % (Median [Minimum; Maximum]) | besser als KQ |
|---|---|---|
| Orakel | 0,0 [0,0; 0,0] | – |
| KQ | 0,0 [0,0; 0,0] | – |
| SPO+ (300 Schritte) | 0,0 [0,0; 0,0] | 0 von 20 |
| geglättet, T = 0,001 | 0,2 [0,1; 0,3] | 0 von 20 |
| geglättet, T = 0,03 | 0,0 [0,0; 0,0] | 0 von 20 |
| geglättet, T = 0,1 | 0,0 [0,0; 0,0] | 0 von 20 |
| geglättet, T = 0,3 | 0,0 [0,0; 0,0] | 0 von 20 |
| geglättet, T = 1 | 0,0 [0,0; 0,0] | 0 von 20 |
| geglättet, T = 3 | 0,0 [0,0; 0,1] | 0 von 20 |

**Grad 6, n = 100, Rauschstufe 0,5 (wenig Daten):**

| Verfahren | Regret in % (Median [Minimum; Maximum]) | besser als KQ |
|---|---|---|
| Orakel | 10,5 [9,3; 12,3] | – |
| KQ | 14,0 [11,5; 16,7] | – |
| SPO+ (300 Schritte) | 14,4 [12,5; 16,7] | 10 von 20 |
| geglättet, T = 0,001 | 21,4 [17,8; 29,8] | 0 von 20 |
| geglättet, T = 0,03 | 15,4 [14,0; 19,0] | 3 von 20 |
| geglättet, T = 0,1 | 15,2 [12,8; 18,4] | 3 von 20 |
| geglättet, T = 0,3 | 14,9 [12,7; 17,6] | 8 von 20 |
| geglättet, T = 1 | 14,7 [12,3; 17,9] | 5 von 20 |
| geglättet, T = 3 | 14,8 [12,7; 19,1] | 5 von 20 |

Was daraus folgt, jeweils nur für diese Zellen und das lineare Modell:

- **T = 0,001 bringt im Median keinen Gewinn:** Ohne Rauschen liegt der Median bei 2,0 % wie der von KQ (in 9 von 20 Seeds besser, Spanne 0,9 bis 3,0 gegen 1,4 bis 3,0); mit Rauschen ist T = 0,001 in allen 20 Seeds schlechter als KQ (15,3 gegen 13,1 %); bei n = 100 ebenfalls in allen 20 (21,4 gegen 14,0 %).
- **Mittlere Temperaturen tragen, eine Rangfolge unter ihnen folgt nicht:** Für T = 0,03 bis 3 liegen die Mediane bei Grad 6, n = 1000 zwischen 0,5 und 0,6 % (ohne Rauschen) und zwischen 11,6 und 12,5 % (mit Rauschen); die Bänder (Minimum bis Maximum über 20 Seeds) von je zwei dieser fünf Temperaturen überlappen sich in beiden Zellen.
- **Gleichauf mit SPO+, nicht eindeutig besser:** Ohne Rauschen ist der Median für T = 0,3 und T = 1 gleich dem von SPO+ (0,5 %); mit Rauschen liegen die Mediane für T = 0,3, 1 und 3 (11,9 / 11,7 / 11,6 %) unter dem von SPO+ (12,3 %), die Bänder überlappen. SPO+ ruft in jedem Schritt den Löser auf allen Trainingsfällen auf; die Glättung braucht keinen Löser in der Schleife, dafür die Aufzählung aller 70 Wege (eine Matrixmultiplikation mit der Wegmatrix).
- **Lineare Wahrheit, wenig Daten:** Bei Grad 1 ohne Rauschen ist keine Variante in einem Seed besser als KQ (0 von 20). Bei Grad 6, n = 100, Rauschstufe 0,5 liegt kein T im Median unter KQ (14,0 %; bestes T = 1 mit 14,7 %, in 5 von 20 Seeds besser; SPO+ in 10 von 20, in der Stabilitätsprobe der Schwesterdemo `spo-kuerzester-weg-demo` mit 100 weiteren Seeds nur in 29 von 100).

### Temperatur und Kostenmaßstab (Experiment 4)

Grad 6, n = 1000, Regret in % nach 300 Schritten, Median [Minimum; Maximum] über 20 Seeds; bei „Kosten mal 10“ sind alle Kosten (Training und Test) mit 10 multipliziert und die Lernrate ist mit 10 multipliziert, T bleibt in Kosteneinheiten.

**Rauschstufe 0,0:**

| Temperatur T | Kosten mal 1 | Kosten mal 10 |
|---|---|---|
| 0,001 | 2,0 [0,9; 3,0] | 5,7 [3,9; 11,4] |
| 0,01 | 0,7 [0,4; 1,0] | 1,9 [0,9; 3,7] |
| 0,1 | 0,5 [0,3; 0,8] | 0,7 [0,4; 1,2] |
| 1 | 0,5 [0,3; 0,7] | 0,5 [0,3; 0,8] |
| 10 | 1,0 [0,8; 1,5] | 0,5 [0,3; 0,7] |
| 100 | 4,8 [3,0; 8,2] | 1,0 [0,8; 1,5] |
| 1000 | 6,5 [4,3; 10,0] | 4,8 [3,0; 8,2] |
| KQ (beide gleich) | 2,0 [1,4; 3,0] | 2,0 [1,4; 3,0] |

**Rauschstufe 0,5:**

| Temperatur T | Kosten mal 1 | Kosten mal 10 |
|---|---|---|
| 0,001 | 15,3 [13,1; 17,1] | 21,0 [17,3; 25,1] |
| 0,01 | 12,2 [11,1; 13,8] | 15,2 [13,1; 20,6] |
| 0,1 | 12,1 [10,7; 13,7] | 12,1 [10,6; 13,4] |
| 1 | 11,7 [10,6; 13,5] | 12,1 [10,7; 13,7] |
| 10 | 11,8 [10,4; 13,5] | 11,7 [10,6; 13,5] |
| 100 | 16,5 [13,2; 20,0] | 11,8 [10,4; 13,5] |
| 1000 | 17,9 [16,0; 21,6] | 16,5 [13,2; 20,0] |
| KQ (beide gleich) | 13,1 [11,3; 14,5] | 13,1 [11,3; 14,5] |

- **Die gute Zone wandert mit dem Maßstab:** Bei Kosten mal 1 liegt der Median-Regret unter KQ für T von 0,01 bis 10 (beide Rauschstufen), bei Kosten mal 10 für T von 0,1 bis 100 (Rauschstufe 0,5), ohne Rauschen auch bei T = 0,01 (1,9 gegen 2,0 %, im Median praktisch gleich); T = 100 und 1000 bei Kosten mal 1 liegen über KQ.
- **Die Verschiebung ist genau der Faktor, wo T nicht sehr klein ist:** Für T = 1, 10, 100 und 1000 bei Kosten mal 10 stimmen die Regret-Werte jedes einzelnen Seeds mit denen bei Kosten mal 1 und T geteilt durch 10 überein (Unterschied höchstens 1e-4 Prozentpunkte). Bei sehr kleinem T weichen die Läufe je Seed voneinander ab (T = 0,1 bei mal 10 gegen T = 0,01 bei mal 1: Mediane 0,7 gegen 0,7 % ohne Rauschen, 12,1 gegen 12,2 % mit Rauschen), die Mediane bleiben nahe; die Ursache (vermutlich Rundung, von Adam verstärkt) ist nicht untersucht. Das ist keine neue Erkenntnis, sondern die Eigenschaft, dass $\hat c$ und $T$ gemeinsam skalieren (im Test `test_scale_equivalence…` auch an einem Seed gezeigt).

### Der Anfang des Trainings (Voreinstellung)

In der Voreinstellung (Grad 6, n = 1000, ohne Rauschen, Zufalls-Seed 0) hat am KQ-Start bei T = 0,001 etwa 1 % der Trainingsfälle (1,3 % im gemessenen Lauf) mehr als einen Weg mit über 1 % Gewicht, bei T = 0,1 etwa 70 % (69,9 %), bei T = 1 und T = 3 alle (100 %). Eine **Deutung** (nicht gemessen): Bei T = 0,001 geht in das Training nur ein Gradient aus den wenigen Fällen nahe an einem Gleichstand ein, und Adam normiert diese kleinen Gradienten auf volle Schrittlänge; das passt zum Befund, dass T = 0,001 mit Rauschen schlechter endet als KQ.

### Prüfungen des Lösers und der Gewichte (Scratch-Lauf, nicht im Repo)

Mit dem Orakel-Test als Vorlage über 300 Zufallsinstanzen (Kosten gleichverteilt in $[0{,}5; 4]$, $T$ aus {0,03; 0,1; 0,3; 1; 3; 10}): größte Abweichung der Gibbs-Gewichte (Aufzählung gegen eine rekursiv erzeugte Wegliste in Python-Zahlen) 1,4·10⁻¹⁴, der Randverteilungen und von $\log Z$ (Summe-Produkt gegen Aufzählung) 1,3·10⁻¹³, des Lösers gegen vollständige Aufzählung (Wert) 5·10⁻¹⁵ bei gleichem Weg in allen 300 Fällen, des Gradienten gegen zentrale Differenzen ($h = 10^{-6}$) der Summe-Produkt-Rechnung 4,4·10⁻⁷. Die genauen Ziffern hängen von den gezogenen Instanzen ab; belastbar ist die Größenordnung (10⁻¹⁴, 10⁻¹³, 10⁻¹⁵, 10⁻⁷). Im Repo läuft die kleine Fassung (`tests/test_oracle_nullgradient_glaettung.py`).

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vormessung reproduzieren sich nicht auf die Stelle.** Andere Zufallsquelle (SplitMix64 statt numpy) und 300 statt 600 Adam-Schritte in der Vorlage des Gitters. Beispiele (Median, Grad 6, n = 1000): ohne Rauschen KQ 1,87 → 2,0 %, SPO+ 0,55 → 0,5 %, T = 0,03 / 0,1 / 0,3 / 1 / 3: 0,61 / 0,59 / 0,53 / 0,51 / 0,59 → 0,5 / 0,5 / 0,5 / 0,5 / 0,6 %; mit Rauschen KQ 12,47 → 13,1 %, SPO+ 11,84 → 12,3 %, T = 0,03 bis 3: 12,05 / 11,61 / 11,52 / 11,55 / 11,24 → 12,5 / 12,1 / 11,9 / 11,7 / 11,6 %; T = 0,001: 1,95 → 2,0 % und 14,63 → 15,3 %. Anteil unveränderter Wege bei ±0,1 / 1 / 5 / 20 %: 99,8 / 97,5 / 88,5 / 62,1 → 99,7 / 97,6 / 88,1 / 60,9 % (ohne Rauschen), 99,8 / 97,7 / 88,9 / 61,9 → 99,8 / 97,6 / 88,4 / 60,1 % (mit Rauschen). Es gilt die neue Messung. Die Rangfolge „T = 0,001 wie KQ oder schlechter, T = 0,03 bis 3 darunter“ bleibt.
- **„Die Temperatur muss weder fein noch grob sein, nur nicht null“ (Plan) trägt nicht.** Bei zu großem T verliert die Glättung wieder: bei Kosten mit 5. bis 95. Perzentil 1,0 bis 3,9 liegen T = 100 und T = 1000 über KQ (Tabellen oben). Der Plan hatte nur T bis 3 gemessen; die Aussage ist darum eingeschränkt auf „T im Bereich 0,01 bis 10 (drei Zehnerpotenzen, vier Rasterwerte) unter KQ, T = 0,001 und T ab 100 nicht“, gemessen auf einem Zehnerpotenz-Raster und bezogen auf diese Kostenskala.
- **Die Grenze der Aufzählung im Plan („größere Netze brauchen Störung oder Regularisierung“) ist zu grob.** Die Aufzählung wächst mit der Zahl der Wege (binomial(2 (h − 1), h − 1): 70 bei h = 5, 252 bei h = 6, 924 bei h = 7). Auf dem azyklischen Gitter gibt es aber die Summe-Produkt-Rechnung ohne Wegliste; sie liefert Wert und Randverteilungen. Für den Gradienten ohne Aufzählung gibt es auf dem Gitter Wege (zum Beispiel die Ableitung der Summe-Produkt-Rechnung); die Demo baut sie nicht. Für allgemeine Polytope (Wege mit Nebenbedingungen) fehlt die Struktur; das ist die eigentliche Grenze.
- **Neu gegenüber dem Plan:** Bei n = 100 mit Rauschstufe 0,5 (Grad 6) hilft die Glättung nicht (kein T im Median unter KQ, bestes T in 5 von 20 Seeds besser); SPO+ liegt dort in 10 von 20 Seeds unter KQ (14,4 gegen 14,0 %), in Übereinstimmung mit der Messung von `spo-kuerzester-weg-demo` (Vormessung 2 von 20 trägt dort nur schwach). Bei Grad 1 liegt der Anteil unveränderter Wege deutlich niedriger als bei Grad 6 (Tabelle oben).
- **Dateiumfang:** n = 5000 aus dem Gitter-Gerüst ist nicht übernommen (Trainingsfälle 100 / 300 / 1000); die Messung deckt vier Zellen ab, nicht das ganze Raster.
- **Kein numpy-Wald, kein scikit-learn:** der Zufallswald der Vorlage kommt hier nicht vor.

## Ehrliche Grenzen

- **Eine Welt, ein Gitter, eine Modellklasse:** nur 5 × 5, nur diese Kostenformel, nur lineares Modell, Adam mit fester Lernrate und 300 Schritten, nur Gibbs-Glättung (keine quadratische Regularisierung des LP). Aussagen über andere Netze, Modelle, Löser oder Lernraten sind nicht gemessen.
- **Die Temperatur ist an die Kostenskala gebunden** (Kosten mit 5. bis 95. Perzentil 1,0 bis 3,9): eine Zahl wie „T = 1“ gilt nur für diese Skala; Experiment 4 zeigt das für den Faktor 10 (die Lernrate wird dabei mitskaliert; eine feste Lernrate würde die Kurven zusätzlich verändern, das ist nicht gemessen).
- **Die Mediane der mittleren Temperaturen unterscheiden sich innerhalb der Streuung:** keine Rangfolge unter T = 0,03 bis 3; kein Signifikanztest, die Spanne (Minimum bis Maximum) ist keine Konfidenzaussage.
- **Zahlen sind trainiert und plattformabhängig.** Gemessen unter Windows; unter Linux können einzelne Werte um Zehntelpunkte abweichen. Die Tests prüfen Bänder und Ungleichungen.
- **Die Deutung zu T = 0,001** (Adam normiert die kleinen Gradienten wenig verschiedener Fälle) ist nicht gemessen; gemessen ist nur, wie viele Fälle am Start überhaupt einen Gradienten liefern.
- **Das Gitter und die Kostenformel sind aus dem Gedächtnis nachgebaut**, nicht gegen den Primärtext geprüft. Elmachtoub und Grigas (Management Science 68(1), 2022; SPO+) habe ich nur auf der Ebene des Abstracts gelesen. Hu, Kallus und Mao (*Fast Rates for Contextual Linear Optimization*, Management Science 68(6), 2022, arXiv 2011.03030, Volltext gelesen, Nummern nach arXiv v3, Beweise und Anhänge nicht gelesen) nennen als Bedingung für schnelle Raten, wie oft die Lücke zwischen bestem und zweitbestem Extrempunkt klein ist; diese Lücke wird hier nicht gemessen, Experiment 2 misst nur, wie oft eine kleine Störung den Weg ändert. Für die Gibbs-Glättung selbst wurde keine Quelle gelesen; Formeln und Gradient sind selbst hergeleitet und in den Tests nachgerechnet.
- **Brücke zur Ableitung am Ergebnis:** Die Aussagen über `kegel-ableitung-demo` stützen sich auf deren README (gelesen am 2026-11-09, Stand 25f4b40): Sie setzt eine eindeutige, reguläre Lösung voraus, am Knick (Gleichstand) entscheidet das Rauschen des Lösers, bei nicht eindeutigen Lösungen ist `J` singulär. Dass die Lösung eines kürzesten-Weg-LP in den Kosten stückweise konstant ist (Ableitung null, wo sie existiert), ist hier eine eigene Überlegung, in Experiment 2 gemessen und nicht aus jener Demo übernommen; sie rechnet kein Gitter. `entrollte-netze-demo` (Ableitung durch den Algorithmus) ist nur verlinkt.

## Bewusst nicht umgesetzt

- **Störung statt Aufzählung** (Fenchel-Young-Verlust, gestörte Optimierer) und **Regularisierung des LP**: andere Wege zum Gradienten, hier nicht gebaut.
- **Gradient der Glättung ohne Aufzählung** (auf dem Gitter zum Beispiel über die Ableitung der Summe-Produkt-Rechnung): nicht gebaut; die Summe-Produkt-Rechnung dient nur als Test.
- **Größere Gitter, Wege mit Nebenbedingungen, andere Modellklassen** (Zufallswald, Kern-Ridge): nicht gemessen.
- **Die Kegelableitung** nachzubauen: ist die Aufgabe von `kegel-ableitung-demo`.
- **Rangfolge der Temperaturen, Abstimmung von T per Kreuzvalidierung, Lernraten-Suche:** nicht gemessen.

## Verifikation

- **Kern:** Gibbs-Gewichte, erwartete Kosten und Gradient gegen das Handbeispiel von Hand (`tests/test_gibbs.py`: Gewichte 0,2689 / 0,7311, erwartete Kosten 4,4621, Steigung 0,39322), Grenzfälle (T → 0 liefert den harten Weg, T → ∞ die Gleichverteilung und das Mittel der Wegkosten), Eigenschaften (Normierung, Skalieren von $\hat c$ und $T$ mit demselben Faktor, Verschieben aller Bögen, Monotonie des erwarteten Regrets in $T$ bei exakter Vorhersage, Gradient gegen zentrale Differenzen).
- **Orakel (`tests/test_oracle_nullgradient_glaettung.py`):** Gibbs-Gewichte über Aufzählung gegen eine rekursiv aufgebaute Wegliste in Python-Zahlen, gegen die Summe-Produkt-Rechnung (Randverteilungen, $\log Z$) und gegen `scipy.special.logsumexp`; der Löser gegen vollständige Aufzählung (300 Fälle); der Gradient der erwarteten Kosten gegen zentrale Differenzen der Summe-Produkt-Rechnung; 3 × 3 Gitter von Hand (6 Wege).
- **Läufe:** Kostenfaktor (Skalierungsgleichheit), Anteil unveränderter Wege (Handfälle im 2 × 2 Gitter), geglättetes Training (`tests/test_runs.py`); kopierte Module (Zufallsstrom, Gitter, Welt, Verluste, Training) mit ihren Tests erneut geprüft.
- **Zahlen der README:** `tests/test_claims.py` prüft jede Tabellenzeile wörtlich gegen `ngl_bands.json` und jede Aussage mit Zahl an der Messung. Nicht im Repo, sondern im Scratch-Lauf: die 300 Instanzen des Orakels (Abschnitt oben).
- **Presets, Layout, Oberfläche:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`); AppTest-Rauchtest (jeder Button, jeder Regler an den Rändern), Permalink, Verdikt je Preset, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (vier Experimente, Brücke, Grenzen, Formeln) |
| `ngl_rng.py`, `ngl_grid.py`, `ngl_world.py`, `ngl_losses.py`, `ngl_training.py` | Kopien aus `spo-kuerzester-weg-demo` (Zufallsstrom, Gitter und Löser, Kostenwelt, Regret und SPO+, KQ und Adam), nur umbenannt |
| `ngl_gibbs.py` | Gibbs-Glättung: Gewichte, erwartete Kosten, Gradient, Zielfunktion; Summe-Produkt als zweite Rechenweise |
| `ngl_runs.py` | ein Lauf je Welt (KQ, SPO+, geglättetes Training), Kostenfaktor, Anteil unveränderter Wege |
| `ngl_bands.py`, `ngl_bands.json` | vorgerechnete Bänder (20 Zufalls-Seeds) und ihr Zugriff |
| `ngl_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ngl_evaluation.py` | Handbeispiel, Kennzahlen, Verdikt-Logik, Tabellen |
| `ngl_layout.py` | Fester Kopf (Stylesheet) |
| `ngl_presets.py` | Permalink (Begrenzen und Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `ngl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ngl_pdf_export.py` | PDF der aktuellen Welt |
| `tools/measure_bands.py` | misst `ngl_bands.json` (höchstens zwei Prozesse) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Tests je Modul, Orakel, Claims (`readme_tables.py` erzeugt die Tabellen), Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
