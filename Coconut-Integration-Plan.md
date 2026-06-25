# Coconut in Qwen3.5-0.8B integrieren & benchmarken — Arbeitsplan

> Stand: Juni 2026. Ziel: Metas **Coconut** (Chain of Continuous Thought) auf **Qwen3.5-0.8B** anwenden und auf **GSM8K + ProsQA** gegen Baselines testen. Rahmen: Entwicklung zuerst auf dem Mac, Training später auf gemieteter Cloud-GPU (Budget ~1000 €). Siehe auch [Latentes-Reasoning.md](./Latentes-Reasoning.md) und [Qwen3.5-0.8B.md](./Qwen3.5-0.8B.md).

---

## 0. Wie Coconut technisch funktioniert (Kurz, damit der Plan klar ist)

Aus dem Original-Code (`facebookresearch/coconut`, `coconut.py`) — der Mechanismus ist überraschend schlank:

1. Es gibt 3 Spezial-Tokens: `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`. Im Eingabe-String markieren `<|latent|>`-Platzhalter die Stellen, an denen „im Latenten gedacht" wird.
2. Beim Forward wird die Eingabe in **Embeddings** umgewandelt (`inputs_embeds`). Statt an einer `<|latent|>`-Position das normale Token-Embedding zu nutzen, wird dort der **letzte Hidden State des vorherigen Schritts** eingesetzt — der „continuous thought".
3. Das geschieht iterativ: pro Latent-Position ein zusätzlicher Forward-Pass, der den vorigen Hidden State zurückspeist. Danach ein finaler Pass für die Antwort.
4. **Trainings-Curriculum:** zuerst normales CoT-Finetuning (Stage 0), dann werden schrittweise explizite Reasoning-Schritte durch je `c_thought` Latent-Tokens ersetzt (Stage 1, 2, …). Initialisiert wird Coconut aus dem CoT-Checkpoint.

Repo-Struktur: `coconut.py` (Modell-Wrapper), `dataset.py` (Daten/Collator), `run.py` (Train/Eval-Loop, `torchrun`), `args/*.yaml` (Konfigs), `preprocessing/` (Daten), `data/` (u. a. **ProsQA liegt schon bei**).

Datenformat (JSON-Liste):
```json
[{ "question": "...", "answer": "...", "steps": ["...", "..."] }]
```

---

## ⚠️ 1. Die zentrale Hürde: Qwen3.5 ist hybrid

Coconut wurde laut Repo **mit GPT-2 und Llama-3 getestet** — beides reine (Standard-)Attention-Modelle. Der Code nutzt an einer Stelle, dass der **KV-Cache eine Liste von `(key, value)`-Tupeln** ist, die man entlang der Sequenz **zuschneiden** kann:

```python
past_key_values = [(k[:, :, :n, :], v[:, :, :n, :]) for k, v in kv_cache]
```

Qwen3.5-0.8B hat aber eine **hybride Architektur**: 18 lineare-Attention-Layer (**Gated DeltaNet**) + 6 volle-Attention-Layer (3:1). Die linearen Layer führen **keinen zuschneidbaren KV-Cache**, sondern einen rekurrenten Zustand (SSM-artig). Der obige Slicing-Trick **funktioniert dort nicht** → das ist der Haupt-Anpassungspunkt.

**Lösung (empfohlen): KV-Cache weglassen, jeden Pass voll neu rechnen.**
Statt den Cache zwischen Latent-Pässen wiederzuverwenden, rechnet man jeden Forward-Pass über die gesamte bisherige Sequenz (`past_key_values=None`). Das ist architektur-agnostisch und für ein 0.8B-Modell bei kurzen GSM8K/ProsQA-Sequenzen völlig vertretbar (etwas langsamer, aber korrekt). Das ist die sauberste Variante für ein hybrides Modell.

Zusätzliche Stolpersteine:
- **transformers-Cache-API:** Neuere transformers-Versionen geben `DynamicCache` statt Tupel zurück. Entweder Repo-Version aus `requirements.txt` pinnen oder die No-Cache-Variante nutzen (umgeht das Problem ganz).
- **Qwen3.5 ist neu (März 2026):** transformers-Support prüfen, ggf. aktuelle/`main`-Version nötig.
- `generate()` im Repo unterstützt **nur batch_size == 1**.
- **bf16 auf Mac/MPS** ist eingeschränkt → auf dem Mac fp32/fp16 nutzen, bf16 erst auf der A100.

---

## 2. Empfehlung: in zwei Stufen de-risken

Nicht sofort alles auf einmal. Reihenfolge, die Fehlerquellen trennt:

1. **Pipeline zuerst auf reinem Attention-Modell validieren.** Erst die adaptierte Codebasis mit einem kleinen Standard-Modell (z. B. `Llama-3.2-1B` oder `Qwen2.5-0.5B`) end-to-end durchlaufen lassen — dort funktioniert der Original-KV-Cache-Pfad. So trennst du „Coconut-Logik korrekt?" von „Qwen3.5-Hybrid korrekt?".
2. **Dann auf Qwen3.5-0.8B umstellen** mit der No-Cache-Variante.

---

## 3. Phasenplan

### Phase 0 — Mac: Setup & Sanity (kein Training)
- Repo klonen, venv/conda, `requirements.txt`. wandb-Account (Logging).
- Qwen3.5-0.8B laden, prüfen: `get_input_embeddings()`, Forward mit `inputs_embeds`, `output_hidden_states=True`, Form von `past_key_values`.
- Mini-Skript: 3 Spezial-Tokens hinzufügen, `model.resize_token_embeddings(len(tokenizer))`, IDs an den Coconut-Wrapper geben.
- **Ergebnis:** Klarheit, ob/welche Cache-Anpassung nötig ist (erwartet: ja, No-Cache).

### Phase 1 — Mac: Code-Adaption
- `coconut.py`: GPT-2-Spezialfall ist schon per `else`-Zweig abgedeckt. **No-Cache-Variante** einbauen (jeden Latent-Pass über die volle Präfix-Sequenz rechnen, kein Slicing). Optional Flag `use_kv_cache: false`.
- Cache-API-Kompatibilität / Versions-Pin klären.
- `dataset.py`/Tokenizer: Qwen-Chat-Template & EOS korrekt setzen.
- **Ergebnis:** Lauffähiger Coconut-Wrapper um Qwen3.5-0.8B.

### Phase 2 — Mac: Daten vorbereiten
- **GSM8K:** `bash preprocessing/gsm_icot.bash` (nutzt die augmentierten Train/Val-Sets aus dem Internalize-CoT-Repo).
- **ProsQA:** liegt bereits unter `data/prosqa_*.json` — nichts zu tun.
- Format prüfen: `{question, answer, steps[]}`.

### Phase 3 — Mac: Smoke-Test (Überanpassung)
- `debug: true`, ~50 Beispiele, batch_size 1, wenige Schritte, CPU/MPS.
- Ziel: Loss fällt, Latent-Forward läuft fehlerfrei, eine Eval-Vorhersage kommt raus. **Kein** Performance-Anspruch — nur „Pipeline grün".

### Phase 4 — Cloud-GPU: Volles Curriculum-Training
- **GPU mieten:** einzelne **A100 80GB** reicht (0.8B passt locker, sogar volles Fine-Tuning). Anbieter z. B. RunPod/Lambda/Vast, grob ~1,5–2 €/h.
- **Pro Datenset zwei Trainings:**
  1. **Stage 0 — CoT-Finetune** (`args/gsm_cot.yaml`): Checkpoint wählen (laut Repo Val-Accuracy ~40 %).
  2. **Coconut-Curriculum** (`args/gsm_coconut.yaml`): `load_model_path` = CoT-Checkpoint; Hyperparameter wie `c_thought`, `max_latent_stage`, `epochs_per_stage` aus der YAML übernehmen (Werte dort nachlesen, nicht raten). Bei 1 GPU `nproc_per_node 1` und `batch_size_training`/`gradient_accumulation_steps` anpassen.
- Analog für **ProsQA** (`args/prosqa_coconut.yaml`).
- **Budget-Check:** Bei ~2 €/h sind 1000 € ≈ 500 GPU-Stunden. Eine 0.8B-Curriculum-Trainingsrunde liegt typischerweise im Bereich weniger bis ~mehrerer Stunden → reichlich Puffer für Wiederholungen/Ablationen.

### Phase 5 — Benchmark & Vergleich
- **Eval-Konfigs:** `args/gsm_coconut_eval.yaml`, `args/prosqa_coconut_eval.yaml` (`only_eval: true`, `load_model_path` = bester Checkpoint).
- **Baselines fair mittrainieren/-testen** (das Repo kann alle Modi):
  - `no_cot` (direkte Antwort), `cot` (explizit), `no_thoughts` (Coconut ohne Gedanken), **`coconut`**.
  - Zusätzlich: Qwen3.5-0.8B **Thinking-Modus** als externe Referenz (siehe Qwen-Doku).
- **Metriken:** Genauigkeit (GSM8K exakte Zahl, ProsQA korrekte Auswahl), **# generierte Tokens**, **Latenz/Durchsatz**, und für den Kern des Projekts: **Rechenaufwand vs. Genauigkeit** (Tokens/Forward-Pässe als Proxy). Failure-Modes notieren (z. B. Thinking-Loops vs. stabile Latent-Pässe).

---

## 4. Erfolgskriterien / was die Arbeit zeigen soll
- **Funktionsnachweis:** Coconut läuft auf einem hybriden Sub-1B-Modell (Beitrag, da Original nur GPT-2/Llama).
- **Vergleich:** Coconut vs. CoT vs. No-CoT auf GSM8K & ProsQA — Genauigkeit **und** Effizienz (Tokens/Latenz).
- **Erwartung laut Paper:** Coconut glänzt v. a. bei **ProsQA** (Suche/BFS); bei GSM8K eher Effizienzgewinn als großer Genauigkeitssprung. Realistisch ist das die spannendere Story.

---

## 5. Sofort-To-dos (nächster Schritt)
1. Entscheiden: erst Pipeline auf `Llama-3.2-1B`/`Qwen2.5-0.5B` validieren? (empfohlen, ja)
2. Repo klonen + Mac-Env aufsetzen.
3. Ich kann als Nächstes die **angepasste `coconut.py` (No-Cache-Variante für Qwen3.5)** + ein **Mac-Smoke-Test-Skript** generieren.

## 6. Quellen
- Coconut-Repo: https://github.com/facebookresearch/coconut
- Coconut-Paper (2412.06769): https://arxiv.org/abs/2412.06769
- GSM8K-Vorverarbeitung (Internalize-CoT-Daten): https://github.com/da03/Internalize_CoT_Step_by_Step
- Community-CPU/Mini-Repro: https://github.com/LuciaLicakova/coconut-cpu-mini
- OpenCoconut (alternative Implementierung): https://github.com/casper-hansen/OpenCoconut
