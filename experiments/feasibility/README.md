# Attention-output-change feasibility experiment

The inference implementation lives in `src/kvrepair/repair_probe.py` and `src/kvrepair/feasibility.py`. It is separate from the original CacheEngine and demonstration CLI. The [protocol](protocol.md) defines the frozen selectors, costs and scoring; [Gate 1](../../docs/feasibility-gate1.md) records numerical assumptions and measurements.

Run from the repository root with the existing `.venv` and cached model. To reproduce without replacing the original pilot, use a fresh output directory:

```sh
PILOT_DIR=artifacts/feasibility-replication
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m kvrepair.feasibility gate1 --directory "$PILOT_DIR"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m kvrepair.feasibility development --directory "$PILOT_DIR"
.venv/bin/python -m kvrepair.feasibility freeze --directory "$PILOT_DIR"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m kvrepair.feasibility heldout --directory "$PILOT_DIR"
```

A failed Gate 1 prevents advancement. Development also checks all-position repair against native recomputation. Freeze records source, configuration and dataset fingerprints. Held-out execution refuses changed inputs or an existing pilot artifact. Complete rendered prompts are retokenized; unsupported lengths, seams and insufficient repair budgets are explicitly rejected.

The original pilot was paused at the user's request after two complete held-out questions, then explicitly resumed. `resume_pilot.py` verifies the frozen source/configuration/data and library versions, saves the partial artifact, and evaluates only unfinished case IDs using the unchanged evaluator. Its runtime substitutions affect case enumeration and checkpoint persistence only. The combined report retains completed records verbatim, records execution sessions, and writes checkpoints atomically. It does not tune or rerun completed cases. If the pilot is already complete, it performs no inference.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u experiments/feasibility/resume_pilot.py
.venv/bin/python experiments/feasibility/report_results.py
```

`report_results.py` reads existing artifacts and renders tables without inference. Its paths target the original pilot. Raw outputs include answers, token limits, original/full failures, actual token edits, selected repair positions, logit/distribution differences, answer preferences and costs. Manually audited factual annotations are kept separate from the frozen phrase scores.

Eight development questions share three document edits; 24 planned held-out questions share eight different edits. They are not independent per-question samples. The model is unquantized TinyLlama in CPU float32. No cloud compute, model downloads, training, quantization or serving integration is involved. Two interleaved timing repetitions are functional observations; metadata preparation, selection, copying, repair and generation are charged, without assuming amortization or GPU performance.
