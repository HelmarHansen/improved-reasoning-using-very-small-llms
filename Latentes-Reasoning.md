# Nicht-natürlichsprachiges, latentes & rekursives Reasoning — Wissensstand

> Stand: Juni 2026. Überblick über den Forschungsstand zu Reasoning, das **nicht** in natürlicher Sprache (Token für Token), sondern in kontinuierlichen/latenten Zuständen und/oder durch wiederholte (rekursive) Berechnung stattfindet. Fokus mit Blick auf das Projektziel „besseres Reasoning mit sehr kleinen LLMs". Quellen am Ende.

---

## 1. Worum geht es? Begriffsklärung

**Klassisches Chain-of-Thought (CoT)** lässt ein Modell laut „nachdenken": Es erzeugt eine Kette von Wort-Tokens, liest seine eigene Ausgabe wieder ein und rechnet so schrittweise weiter. Das Denken passiert also **im Sprachraum** (diskrete Tokens).

**Latentes / nicht-natürlichsprachiges Reasoning** verlagert dieses Nachdenken in den **kontinuierlichen verborgenen Zustandsraum** des Modells (die Hidden States / Aktivierungen). Statt jeden Zwischenschritt in Worte zu übersetzen, rechnet das Modell direkt mit hochdimensionalen Vektoren weiter. „Rekursiv" heißt dabei: dieselbe Berechnung (derselbe Block) wird mehrfach auf den Zustand angewendet, um „tiefer" zu denken.

Zwei oft genannte Motivationen:

1. **Bandbreite/Informationsverlust.** Ein Token trägt nur wenige Bit. Ein Hidden-State-Vektor (z. B. 1024–4096 Dimensionen in float) trägt um Größenordnungen mehr Information. Verbalisierung zwingt das Modell, seinen reichen, parallelen internen Zustand in einen schmalen, sequentiellen Tokenstrom zu pressen — ein „functional rift" zwischen paralleler interner Berechnung und sequenzieller Sprachausgabe.
2. **Manche Berechnung lässt sich schlecht versprachlichen.** Suche, Backtracking, Halten mehrerer Hypothesen gleichzeitig — das passt schlecht in eine festgelegte, lineare Wortkette.

**Zentrale Unterscheidung im Feld:**

| Achse | Pol A | Pol B |
|---|---|---|
| Repräsentation | diskrete Tokens (Sprache) | kontinuierliche Vektoren (latent) |
| Sichtbarkeit | explizit (lesbar) | implizit (intern, nicht lesbar) |
| Mehr-Rechenzeit durch | mehr Tokens (Breite/Länge) | mehr Iterationen/Tiefe (Rekursion) |

Nicht-natürlichsprachiges latentes rekursives Reasoning sitzt jeweils auf Pol B.

---

## 2. Warum das theoretisch mächtiger sein kann

Mehrere Arbeiten liefern formale Argumente, dass „im Latenten / durch Wiederholung denken" die Ausdrucksstärke echt erhöht:

- **Looped Transformers sind Turing-vollständig.** Indem man einen Transformer-Block mit geteilten Gewichten wiederholt anwendet, kann man einen universellen Rechner (z. B. SUBLEQ/OISC) emulieren — sogar bei konstanter Tiefe und Breite, bei geeigneter Eingabekodierung. Viele Reasoning-Probleme brauchen **mehr Tiefe, nicht mehr Parameter**.
- **Pause/Filler-Tokens erhöhen die Ausdrucksstärke nachweisbar.** Für Transformer konstanter Tiefe und logarithmischer Breite gilt: ohne Filler berechnen sie nur eine echte Teilmenge von AC⁰; mit polynomiell vielen Pause-Tokens lässt sich die ganze Klasse AC⁰ ausdrücken. Schon das Einschieben „inhaltsloser" Tokens (z. B. `......`) schafft also zusätzlichen Berechnungsspielraum, weil deren Hidden States Rechenarbeit für spätere Tokens leisten.
- **„Reasoning durch Superposition".** Theoretische Analyse von Chain-of-Continuous-Thought (Coconut) argumentiert, dass ein kontinuierlicher Gedanke **mehrere alternative nächste Schritte gleichzeitig** kodieren kann (eine Art Überlagerung). Dadurch wird statt eines vorzeitig festgelegten Pfads effektiv eine **Breitensuche (BFS)** möglich — vorteilhaft bei Planungs-/Suchproblemen.

Kurz: Tiefe-durch-Rekursion und Rechnen-im-Kontinuierlichen sind nicht nur Effizienztricks, sondern können Funktionsklassen erreichen, die reines „No-CoT" nicht erreicht.

---

## 3. Die wichtigsten Ansätze (geordnet nach Familie)

### 3.1 Kontinuierliche Gedanken als Eingabe-Embedding (Coconut & Verwandte)

**Coconut — „Chain of Continuous Thought"** (Hao et al., Meta, Dez. 2024) ist der Referenzpunkt. Idee: Statt den letzten Hidden State in ein Wort-Token zu dekodieren, wird er **direkt als Eingabe-Embedding des nächsten Schritts** zurückgeführt. Spezielle Marker `<bot>`/`<eot>` (begin/end of thought) schalten den Latent-Modus an und aus. Trainiert wird per Curriculum (schrittweises Ersetzen expliziter CoT-Schritte durch latente). Ergebnis: bei Logik-/Planungsaufgaben mit viel Suchbedarf schlägt Coconut explizites CoT und braucht weniger Tokens, weil der kontinuierliche Gedanke alternative Pfade überlagern kann (BFS-artig).

Verwandte / aufbauende Arbeiten:

- **SoftCoT / SoftCoT++** (Xu et al., 2025): Ein kleines Hilfsmodell erzeugt „soft thought tokens", die über ein Projektionsmodul ins LLM gespeist werden — ohne das Haupt-LLM zu verändern; vermeidet katastrophales Vergessen.
- **CODI / Compressed CoT (CCoT)** und ähnliche Distillations-Ansätze: komprimieren explizite Reasoning-Ketten in dichte, kontinuierliche Repräsentationen.
- **Token Assorted** (Su et al., 2025): mischt **latente diskrete Tokens** (per VQ-VAE erzeugt) mit normalen Text-Tokens, kürzt so die Reasoning-Spuren und verbessert Performance.
- **Hybrid Reasoning Policy Optimization (HRPO)** (Yue et al., 2025): integriert latentes Reasoning per Reinforcement Learning, mischt Hidden States und Token-Embeddings.
- **SwiReasoning** (2025): trainingsfrei; wechselt dynamisch zwischen latentem und explizitem Modus je nach Konfidenz/Entropie — Pareto-Verbesserung bei Genauigkeit vs. Tokenkosten.

### 3.2 Rekursive Tiefe / Looped Transformers (Rechnen durch Wiederholung)

Hier ist die zentrale Idee nicht ein „Gedanken-Embedding", sondern das **wiederholte Anwenden desselben Blocks** auf den Zustand, um Rechen-Tiefe zur Testzeit beliebig zu skalieren.

- **Recurrent Depth / „Huginn"** (Geiping et al., Feb. 2025, *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*): Ein rekurrenter Block wird zur Testzeit auf beliebige Tiefe „aufgerollt". Proof-of-Concept mit **3,5 Mrd. Parametern**, 800 Mrd. Tokens; die Reasoning-Leistung steigt mit mehr Iterationen — bis zu einer Rechenlast, die **~50 Mrd. Parametern** entspricht. Vorteile laut Autoren: **keine speziellen CoT-Trainingsdaten** nötig, funktioniert mit kleinem Kontextfenster, erfasst Reasoning-Arten, die sich schlecht in Worte fassen lassen. (Veröffentlicht auf NeurIPS/ICML 2025, Code + Modell „Huginn-0125" offen.)
- **Ouro — Looped Language Models (LoopLM)** (ByteDance, Okt. 2025, *Scaling Latent Reasoning via Looped Language Models*): baut Reasoning bereits in die **Vortraining**sphase ein — geteilte Gewichte, iterative Berechnung im Latenten, **entropie-regularisiertes Ziel** für gelernte Tiefen-Zuteilung, skaliert auf 7,7 Bio. Tokens. Bemerkenswert: **Ouro-1,4B und -2,6B erreichen die Leistung von bis zu 12B-SOTA-Modellen**; der Vorsprung kommt aus besserer *Wissens-Manipulation*, nicht mehr Wissens-*Kapazität*. LoopLM-Reasoning-Spuren sind außerdem **besser mit der finalen Antwort aligned** als explizites CoT.
- **Theorie & Mechanik:** Looped/recurrent-depth-Modelle erlauben **Tiefen-Extrapolation** (Generalisierung auf mehr Reasoning-Tiefe als im Training gesehen). Arbeiten wie *What Makes Looped Transformers Perform Better Than Non-Recursive Ones*, *Loop, Think & Generalize*, *Beyond Memorization* und Stabilisierungs-Arbeiten (2026) untersuchen, warum und wann das funktioniert und wie man die rekurrente Dynamik testzeit-stabil hält.

### 3.3 Internalisiertes / implizites CoT (Verbalisierung ganz weglassen)

- **Stepwise Internalization / ICoT-SI** (Deng et al., 2024, *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step*): Start mit einem explizit auf CoT trainierten Modell; dann werden die Zwischenschritte **schrittweise entfernt** und das Modell jeweils nachtrainiert, bis es die Antwort direkt (ohne sichtbare Schritte) erzeugt. Die „Scratchpad"-Funktion der Zwischenschritte übernehmen die internen Zustände. Trade-off: bei 9×9-Multiplikation vergleichbar genau wie explizites CoT, aber **~11× schneller**; bei GSM8K aber Genauigkeitsverlust (Mistral-7B: 0,68 explizit → 0,51 internalisiert).
- **Filler-/Pause-Tokens** (*Let's Think Dot by Dot*, Pfau & Merrill 2024; Goyal et al. „Think before you speak"): Das Modell erzeugt bedeutungslose Füll-Tokens (`...`) vor der Antwort; deren Hidden States liefern trotzdem nützliche Berechnung. Praktischer Beleg für „verstecktes Rechnen" ohne interpretierbare Schritte.

### 3.4 Latente Tokens mit Vokabular-Bezug (Brücke zwischen latent und diskret)

- **Latent-SFT / „Vocabulary-Space Superposition"** (Deng et al., Okt. 2025): latente Tokens werden als **Superposition über Vokabular-Wahrscheinlichkeiten** interpretiert; erreicht Leistung nahe explizitem CoT bei geringeren Kosten und „globaler Parallelität".
- **Geometric Latent Reasoning** (Kumar et al., Juni 2026): formuliert latentes Reasoning als **geometrisches Pfad-Approximationsproblem im vortrainierten Token-Embedding-Raum**; kürzere Generierungen bei gehaltener Genauigkeit (getestet u. a. mit Qwen3).
- **LatentSeek** (Li et al., Mai 2025): Test-Time-Anpassung pro Instanz via **Policy-Gradient im Latenten** (selbst erzeugte Reward-Signale), ohne Gewichte zu ändern.

### 3.5 Multimodal & Diffusion

- **Monet**, **MCOUT** (2025): latentes Reasoning im **visuellen** Raum für Vision-Language-Modelle — kontinuierliche „visuelle Gedanken" jenseits von Bild und Sprache.
- **Maskierte Diffusionsmodelle** werden in Surveys als Weg zu „infinite-depth" latentem Reasoning genannt (iterative Verfeinerung statt links-nach-rechts-Dekodierung).

---

## 4. Überblickstabelle der Schlüsselarbeiten

| Arbeit (Jahr) | Familie | Kernidee | Befund |
|---|---|---|---|
| Coconut (2024) | kontinuierl. Gedanke | Hidden State als nächstes Eingabe-Embedding | schlägt CoT bei Such-/Planungsaufgaben, weniger Tokens |
| ICoT-SI (2024) | internalisiert | explizite Schritte schrittweise entfernen | bis ~11× schneller; teils Genauigkeitsverlust |
| Let's Think Dot by Dot (2024) | Filler-Tokens | bedeutungslose `...`-Tokens | verstecktes Rechnen messbar nützlich |
| Recurrent Depth / Huginn (2025) | rekursive Tiefe | Block zur Testzeit beliebig oft aufrollen | 3,5B ≈ 50B-Rechenleistung; keine CoT-Daten nötig |
| Token Assorted (2025) | latente diskrete Tokens | VQ-VAE-Tokens + Text mischen | kürzere Spuren, bessere Leistung |
| SoftCoT (2025) | soft thoughts | Hilfsmodell speist weiche Gedanken ein | kein Vergessen, LLM unverändert |
| Pause-Token-Theorie (2025) | Theorie | Filler erhöhen Ausdrucksstärke | erschließt ganz AC⁰ |
| Survey on Latent Reasoning (2025) | Übersicht | Taxonomie des Felds | aktivierungs-basierte Rekurrenz, infinite-depth |
| Ouro / LoopLM (2025) | looped, pretrained | Rekursion ins Vortraining | 1,4–2,6B ≈ bis zu 12B; besser aligned |
| Vocabulary-Space Superposition (2025) | latente Tokens | Superposition über Vokabular | nahe explizitem CoT, billiger |
| Geometric Latent Reasoning (2026) | latente Tokens | Pfad im Embedding-Raum | kürzere Generierung, gleiche Genauigkeit |

---

## 5. Bezug zum Projekt: „Reasoning mit sehr kleinen LLMs"

Das Thema ist für sehr kleine Modelle besonders attraktiv, weil **Tiefe/Rekursion Parameter ersetzen kann**:

- **Mehr Rechnen statt mehr Parameter.** Recurrent-Depth (Huginn) und Ouro zeigen empirisch, dass ein kleines Modell durch mehr Iterationen die Effektivleistung eines viel größeren erreichen kann (3,5B→~50B-Rechenlast; 1,4–2,6B ≈ bis 12B). Für ein Sub-1B-Projekt ist das der direkteste Hebel.
- **Keine teuren CoT-Trainingsdaten nötig** (Recurrent Depth): relevant, wenn man kein großes annotiertes Reasoning-Korpus hat.
- **Token-Effizienz/Inferenzkosten.** Internalisiertes/latentes Reasoning kann deutlich schneller sein (ICoT-SI ~11×), weil keine lange Wortkette erzeugt wird — wichtig bei lokaler Ausführung kleiner Modelle.
- **Anknüpfung an Qwen3.5-0.8B** (siehe [Qwen3.5-0.8B.md](./Qwen3.5-0.8B.md)): Das Modell hat einen sichtbaren Thinking-Modus und das dokumentierte Failure-Mode der **„thinking loops"**. Latentes/rekursives Reasoning ist genau die Gegenrichtung — feste/gelernte Iterationszahl im Latenten statt unkontrollierter Token-Endlosschleifen. Ein naheliegendes Experiment: explizites Thinking vs. internalisiertes/latentes Reasoning auf demselben kleinen Modell vergleichen (Genauigkeit, Tokens, Laufzeit, Loop-Häufigkeit).
- **Reproduzierbarkeit.** Huginn-0125 und die Ouro-Modelle sind offen verfügbar; viele Coconut-/ICoT-Implementierungen sind klein genug für Standard-Hardware.

---

## 6. Offene Probleme & Kritik

- **Interpretierbarkeit/Aufsicht.** Latente Schritte sind per Definition nicht lesbar — keine direkte menschliche Kontrolle des „Gedankengangs". Surveys nennen das ein zentrales offenes Problem (unsupervisable processes).
- **Evaluationslücke.** Es ist schwer, **echtes tiefes latentes Reasoning** von einer reinen Input-Output-Abkürzung zu unterscheiden; es fehlen klare Metriken.
- **Faithfulness-Debatte.** Eine Forschungslinie (2026) argumentiert, dass Reasoning ohnehin primär *latent* ist und explizites CoT oft nur **post-hoc-Rationalisierung** — Verbalisierung wirke eher als „Gerüst/Alignment" denn als echte zusätzliche Rechen-Tiefe. Andere zeigen, dass implizites Reasoning **nicht zwingend wirklich Schritt für Schritt** abläuft (*Do LLMs Really Think Step-by-step in Implicit Reasoning?*).
- **Tiefe-Genauigkeits-Paradox / stille Fehler.** *When Shallow Wins* (2026) berichtet, dass mehr latente Tiefe nicht monoton hilft und „silent failures" auftreten können; Stabilisierung der rekurrenten Dynamik zur Testzeit ist ein aktives Thema.
- **Kausalität latenter Tokens.** *Do Latent Tokens Think?* (kausale/adversariale Analyse von Coconut) hinterfragt, wie viel die kontinuierlichen Gedanken tatsächlich zum Ergebnis beitragen.
- **Trainingsstabilität.** Curriculum (Coconut), entropie-regularisierte Tiefen-Zuteilung (Ouro) und kontrastive RL-Varianten existieren gerade *weil* naives Training latenter/rekursiver Schemata instabil ist (Feature-Collapse etc.).

---

## 7. Mentale Landkarte (Kurzfassung)

```
Reasoning
├── im Sprachraum (Tokens)
│   ├── explizit, sichtbar ............... klassisches CoT
│   └── Filler-/Pause-Tokens ............. "Dot by Dot" (Brücke ins Versteckte)
└── im latenten Raum (Vektoren)  ← Thema dieses Dokuments
    ├── kontinuierliche Gedanken ......... Coconut, SoftCoT, CODI, Token-Assorted
    ├── internalisiert (kein Output) ..... ICoT-SI (Stepwise Internalization)
    ├── rekursive Tiefe (Looping) ........ Huginn/Recurrent-Depth, Ouro/LoopLM
    ├── latente Tokens m. Vokabular-Bezug  Latent-SFT, Geometric Latent Reasoning
    └── multimodal / Diffusion ........... Monet, MCOUT, masked-diffusion
```

---

## 8. Quellen

**Kernarbeiten**
- Hao et al. — *Training LLMs to Reason in a Continuous Latent Space* (Coconut), 2024: https://arxiv.org/abs/2412.06769
- Geiping et al. — *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn), 2025: https://arxiv.org/abs/2502.05171
- *Scaling Latent Reasoning via Looped Language Models* (Ouro / LoopLM), 2025: https://arxiv.org/abs/2510.25741 · Projektseite: https://ouro-llm.github.io/
- Deng et al. — *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step*, 2024: https://arxiv.org/pdf/2405.14838
- Pfau & Merrill — *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models*, 2024: https://arxiv.org/abs/2404.15758

**Theorie**
- *Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers*, 2025: https://arxiv.org/abs/2505.21024
- *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought*, 2025: https://arxiv.org/pdf/2505.12514
- *What Makes Looped Transformers Perform Better Than Non-Recursive Ones*, 2025: https://arxiv.org/pdf/2510.10089

**Weitere Methoden**
- Su et al. — *Token Assorted: Mixing Latent and Text Tokens*, 2025: https://arxiv.org/abs/2502.03275
- Xu et al. — *SoftCoT: Soft Chain-of-Thought for Efficient Reasoning*, 2025: https://hf.co/papers/2502.12134
- Yue et al. — *Hybrid Latent Reasoning via Reinforcement Learning* (HRPO), 2025: https://hf.co/papers/2505.18454
- *SwiReasoning: Switch-Thinking in Latent and Explicit*, 2025: https://hf.co/papers/2510.05069
- Deng et al. — *Latent Reasoning in LLMs as a Vocabulary-Space Superposition* (Latent-SFT), 2025: https://hf.co/papers/2510.15522
- Li et al. — *Seek in the Dark: Test-Time Instance-Level Policy Gradient in Latent Space* (LatentSeek), 2025: https://hf.co/papers/2505.13308
- Kumar et al. — *Geometric Latent Reasoning Induces Shorter Generations*, 2026: https://hf.co/papers/2606.02248
- *Compressed Chain of Thought (CCoT)*, 2024: https://arxiv.org/html/2412.13171v1
- *Monet: Reasoning in Latent Visual Space*, 2025: https://hf.co/papers/2511.21395
- *MCOUT: Multimodal Chain of Continuous Thought*, 2025: https://hf.co/papers/2508.12587

**Surveys**
- Zhu et al. — *A Survey on Latent Reasoning*, 2025: https://arxiv.org/abs/2507.06203
- Chen et al. — *Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning*, 2025: https://arxiv.org/html/2505.16782v1
- *Implicit Reasoning in Large Language Models: A Comprehensive Survey*, 2025: https://arxiv.org/html/2509.02350

**Kritik & Grenzen**
- *LLM Reasoning Is Latent, Not the Chain of Thought*, 2026: https://arxiv.org/html/2604.15726
- *When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning*, 2026: https://arxiv.org/pdf/2603.03475
- *Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought*, 2025: https://arxiv.org/pdf/2512.21711
- *Do LLMs Really Think Step-by-step In Implicit Reasoning?*, 2024: https://arxiv.org/html/2411.15862v4
