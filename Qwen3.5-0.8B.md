# Qwen3.5-0.8B — Vollständige Übersicht

> Stand: Juni 2026. Quellen: offizielle Hugging-Face-Modellkarte, `config.json` und Qwen-Blog. Links am Ende.

## 1. Kurzüberblick

Qwen3.5-0.8B ist das kleinste Mitglied der **Qwen3.5-Familie** von Alibabas Qwen-Team, veröffentlicht Anfang **März 2026** (die Familie startete am 16. Februar 2026, die kleinen Modelle 0.8B–9B folgten am 2. März 2026). Trotz seiner geringen Größe ist es ein **nativ multimodales** Modell (Text + Bild + Video) mit einer neuartigen **hybriden Attention-Architektur** und unterstützt sowohl einen **Thinking- als auch einen Non-Thinking-Modus**.

| Eigenschaft | Wert |
|---|---|
| Voller Name | Qwen3.5-0.8B (post-trained) / Qwen3.5-0.8B-Base |
| Parameter | 873,4 Mio. (≈ 0,8 B) |
| Architektur-Typ | `qwen3_5` — Hybrid (Gated DeltaNet + Gated Attention) |
| Modalitäten | Text, Bild, Video → Text |
| Kontextlänge | 262.144 Tokens (256K) nativ |
| Vokabular | 248.320 Tokens (gepaddet; Familie nutzt ~250K) |
| Lizenz | Apache 2.0 (kommerziell frei nutzbar) |
| Format | Hugging Face Transformers / safetensors, bfloat16 |
| Min. Hardware | ~2 GB RAM/VRAM — läuft auf Laptop/Handy/im Browser (WebGPU) |
| Empfohlene Nutzung | Prototyping, aufgabenspezifisches Fine-Tuning, Forschung |

Das Modell ist kompatibel mit Hugging Face Transformers, vLLM, SGLang, KTransformers und Ollama (`ollama run qwen3.5:0.8b`).

## 2. Architektur

Qwen3.5 basiert auf dem Architektur-Ansatz, den Alibaba **„Qwen3-Next"** nennt. Die Kernidee ist ein **hybrides Attention-Design**, das lineare Attention (Gated DeltaNet) mit klassischer voller (Self-)Attention kombiniert. Bei den größeren Modellen der Familie kommt zusätzlich ein sparsames **Mixture-of-Experts (MoE)**-System hinzu; das 0.8B-Modell selbst ist ein **dichtes (dense) Modell** ohne MoE.

### 2.1 Sprachmodell-Kern (aus `config.json`)

| Parameter | Wert |
|---|---|
| Hidden Dimension | 1024 |
| Anzahl Layer | 24 |
| FFN Intermediate-Dim | 3584 |
| Aktivierungsfunktion | SiLU |
| RMSNorm epsilon | 1e-6 |
| Token-Embedding | mit LM-Head geteilt (tied weights) |
| Multi-Token-Prediction (MTP) | 1 zusätzliche Vorhersage-Ebene |

Das Layer-Layout folgt dem Muster:

```
6 × ( 3 × (Gated DeltaNet → FFN)  →  1 × (Gated Attention → FFN) )
```

Das heißt: Auf je **drei lineare-Attention-Blöcke** folgt **ein voller-Attention-Block** (Verhältnis **3:1**, `full_attention_interval = 4`). Über 24 Layer ergeben sich so **18 lineare** und **6 volle** Attention-Schichten.

### 2.2 Gated DeltaNet (lineare Attention)

Die linearen Schichten verwenden **Gated DeltaNet** — eine lineare-Attention-Variante, die den gated-decay-Mechanismus von Mamba2 mit der **Delta-Regel** zur Aktualisierung des versteckten Zustands verbindet.

- Lineare Key/Value-Heads: 16 (QK) und 16 (V)
- Head-Dimension: 128
- Convolution-Kernel-Dim: 4
- SSM-Berechnung in float32

Vorteil: **konstanter Speicherbedarf** unabhängig von der Sequenzlänge. Das macht die native Kontextlänge von 256K Tokens auch auf einem winzigen Modell möglich, weil der KV-Cache nicht linear mit der Länge wächst.

### 2.3 Gated Attention (volle Attention)

Die vollen Attention-Schichten liefern die **präzisionskritische** Rechenarbeit für komplexes Reasoning:

- Attention-Heads: 8 (Query), 2 (Key/Value) → Grouped-Query-Attention (GQA)
- Head-Dimension: 256
- Rotary Position Embedding (RoPE) über `partial_rotary_factor = 0.25` (→ 64 rotierte Dimensionen), `rope_theta = 10.000.000`
- `attn_output_gate = true` (gated output)
- M-RoPE (interleaved) zur Kodierung multimodaler Positionen

**Arbeitsteilung:** Die linearen Schichten übernehmen routinemäßige Berechnung bei konstantem Speicher, die vollen Schichten sorgen für die nötige Präzision bei anspruchsvollem Reasoning. Diese Aufteilung ist der zentrale Effizienz-Trick der Architektur.

### 2.4 Vision-Encoder

Das Modell ist **nativ multimodal** durch Early-Fusion-Training (Text und Vision werden von Anfang an gemeinsam trainiert, kein nachträglich angeschraubter Adapter):

- Vision-Transformer mit 12 Layern, Hidden-Size 768, 12 Heads
- Patch-Size 16, spatial-merge 2, temporal-patch 2
- Output-Projektion auf 1024 (= Sprachmodell-Hidden-Size)
- Eigene Bild-/Video-Token-IDs im Vokabular

## 3. Reasoning: Thinking- und Non-Thinking-Modus

Qwen3.5-0.8B unterstützt — wie die ganze Familie — **zwei Inferenz-Modi**, umschaltbar über den Parameter `enable_thinking` im Chat-Template:

- **Thinking-Modus:** Das Modell erzeugt zuerst eine sichtbare Chain-of-Thought (Gedankenkette), bevor es die finale Antwort gibt. Stärker bei Mathematik, Logik, mehrstufigen Aufgaben.
- **Non-Thinking-Modus:** Direkte Antwort ohne Zwischenschritte. Schneller, günstiger, gut für einfache Aufgaben.

Die größeren Familienmitglieder kennen zusätzlich Modi wie *Auto* (adaptives Denken mit Tool-Use), *Thinking* und *Fast*.

### Wichtige Einschränkung beim 0.8B

Die Modellkarte warnt ausdrücklich: Im Thinking-Modus neigt **gerade Qwen3.5-0.8B** stärker als die größeren Geschwister zu **„thinking loops"** — es kann in Endlosschleifen geraten und die Generierung nicht sauber beenden. Empfehlung des Qwen-Teams: Sampling-Parameter für den konkreten Anwendungsfall feintunen und Streaming nutzen, um solche Schleifen früh zu erkennen und abzubrechen.

### Empfohlene Sampling-Parameter

| Modus / Aufgabe | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Non-Thinking (Text) | 1.0 | 1.00 | 20 | 2.0 |
| Non-Thinking (Vision) | 0.7 | 0.80 | 20 | 1.5 |
| Thinking (Text) | 1.0 | 0.95 | 20 | 1.5 |
| Thinking (Vision / präzises Coding) | 0.6 | 0.95 | 20 | 0.0 |

Weitere Best Practices: Output-Länge ~32.768 Tokens (bis 81.920 bei Wettbewerbs-Mathematik/Coding); bei Mathe den Prompt „Please reason step by step, and put your final answer within \boxed{}." anhängen; in Mehrfach-Dialogen die Thinking-Inhalte **nicht** in die Historie übernehmen.

## 4. Benchmarks (0.8B)

Die folgenden Werte stammen aus der offiziellen Modellkarte. Format **Thinking / Non-Thinking**, wo beide angegeben sind. Zum Vergleich: Das 0.8B ist das kleinste Modell — die Werte sind im Verhältnis zur Winzgröße zu lesen, nicht gegen Frontier-Modelle.

### Sprache / Wissen / STEM

| Benchmark | Qwen3.5-0.8B | (Vgl.) Qwen3.5-2B |
|---|---|---|
| MMLU-Pro (Non-Think) | 29.7 | 55.3 |
| MMLU-Pro (Think) | 42.3 | 66.5 |
| MMLU-Redux (Think) | 59.5 | 79.6 |
| C-Eval (Think) | 50.5 | 73.2 |
| GPQA (Think) | 11.9 | 51.6 |
| SuperGPQA (Think) | 21.3 | 37.5 |

### Instruction Following / Agent

| Benchmark | Qwen3.5-0.8B |
|---|---|
| IFEval (Non-Think) | 52.1 |
| IFEval (Think) | 44.0 |
| IFBench (Think) | 21.0 |
| MultiChallenge (Think) | 18.9 |
| BFCL-V4 (Agent, Think) | 25.3 |

### Multimodal / Vision (Think / Non-Think)

| Benchmark | Qwen3.5-0.8B |
|---|---|
| MMMU | 49.0 / 47.4 |
| MathVista (mini) | 62.2 / 58.6 |
| DynaMath | 49.9 / 46.5 |
| RealWorldQA | 63.4 / 61.6 |
| MMStar | 58.3 / 55.9 |
| AI2D (Test) | 69.9 / 68.7 |
| OmniDocBench 1.5 | 61.0 / 70.6 |
| CC-OCR | 63.2 / 66.7 |

Einordnung: Für ein Sub-1-B-Modell sind besonders die **Vision-/Dokument-Werte** (MathVista 62, OCR ~63–67) bemerkenswert hoch — die multimodale Stärke ist über die ganze Familie ein Markenzeichen.

## 5. Familien-Kontext

Die Qwen3.5-Familie umfasst acht Modelle über vier Größenordnungen, alle mit gemeinsamer Hybrid-Architektur, nativer Multimodalität und 201 Sprachen:

| Serie | Modelle | Release |
|---|---|---|
| Flagship | Qwen3.5-397B-A17B (397B total, 17B aktiv, MoE) | 16. Feb 2026 |
| Medium | Qwen3.5-27B (dense), 35B-A3B, 122B-A10B | 24. Feb 2026 |
| Small | **Qwen3.5-0.8B**, 2B, 4B, 9B | 2. März 2026 |

Gemeinsame Neuerungen gegenüber Qwen3: Vokabular von 150K → ~250K Tokens (10–60 % effizienteres Encoding je Sprache), 201 statt 119 Sprachen, natives FP8-Training (bei den großen Modellen ~50 % weniger Aktivierungsspeicher), und RL über sehr viele Aufgaben/Umgebungen skaliert.

## 6. Relevanz für „Reasoning mit sehr kleinen LLMs"

Für ein Projekt zu verbessertem Reasoning in sehr kleinen Modellen ist Qwen3.5-0.8B aus mehreren Gründen interessant:

- **Sub-1-B mit explizitem Reasoning-Modus** und sichtbarer Chain-of-Thought — direkt vergleichbar mit/ohne Thinking.
- **Apache-2.0 + 2 GB Footprint** — lokal auf Standard-Hardware (sogar Browser/WebGPU) reproduzierbar, ideal für Experimente und Fine-Tuning.
- **Dokumentiertes Failure-Mode** (Thinking-Loops beim 0.8B) — ein konkreter, untersuchbarer Schwachpunkt kleiner Reasoning-Modelle.
- **Base- und post-trained Variante** verfügbar (`Qwen3.5-0.8B-Base`) — erlaubt eigenes Reasoning-Fine-Tuning (z. B. mit selbst erzeugten CoT-Daten oder RL).
- **Multi-Token-Prediction (MTP)** im Training — relevant, falls Inferenz-Beschleunigung/Decoding untersucht wird.

## 7. Quellen

- [Qwen/Qwen3.5-0.8B — Hugging Face Modellkarte](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen/Qwen3.5-0.8B-Base — Hugging Face](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [config.json — Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)
- [Qwen3.5 Blog (offiziell)](https://qwen.ai/blog?id=qwen3.5)
- [Qwen 3.5 Developer Guide — Lushbinary](https://lushbinary.com/blog/qwen-3-5-developer-guide-benchmarks-architecture-integration-2026/)
- [Qwen 3.5: A Complete Model Family from 0.8B to 397B — Enclave AI](https://enclaveai.app/blog/2026/03/08/qwen-3-5-complete-model-family-local-ai/)
