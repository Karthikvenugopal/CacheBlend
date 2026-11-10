# KV repair after document edits

**Archived; method development stopped.** No further selector development or TinyLlama factual sweeps are planned. The code and all results are preserved in the [archive](archives/kvrepair-2026-11-09/README.md). The attention calculation is validated; the selector is rejected; factual repair efficacy remains inconclusive. This project has not demonstrated a successful repair method.

The [reproduction-target assessment](docs/reproduction-target.md) identifies a small-model published candidate but recommends shelving development under current conditions. No new model was downloaded or experiment implemented. MPS is available outside the execution sandbox, as confirmed by a basic tensor test; historical results remain CPU measurements.

The authorized [six-pair Qwen3-1.7B reproduction](docs/qwen-reproduction-results.md) downloaded the pinned checkpoint but stopped at its first native/cached numerical gate on MPS/BF16. Restricted agreement and generated-answer failures remain unestablished; generation was not run. All six case statuses and the bounded first-case investigation are retained. The author's model revision is unknown. Repair and selector development remain stopped.

The subsequent [three-configuration diagnosis](docs/endpoint-precision-diagnosis.md) found that the native MPS/BF16 path reproduces the discrepancy, while CPU and MPS float32 pass the unchanged float32 tolerance. MPS float32/eager is the recommended local candidate, with CPU float32 as reference. Only the first case was examined; no reproduction or repair experiments were resumed.

The later authorized [six-case float32 continuation](docs/qwen-float32-reproduction.md) is complete: 18/18 published restricted choices matched; unrestricted full answers were clearly correct on four cases and inconclusive on two. Four paired MPS reuse-induced wrong decisions were observed, with two cases additionally flagged for CPU–MPS logit differences. All within-backend full/cache checks passed unchanged tolerance. No selective repair was tested; method development remains stopped.

The subsequent [bounded headroom diagnostic](docs/qwen-headroom-results.md) completed 399 actual partial repairs on two fixed interventions at 16/32-token budgets. None recovered the full answer; full-suffix controls did. The input audit also found that bank's padded amount is `3  0 USD`, qualifying its earlier intended-label interpretation. No repair mechanism or selector is proposed from this negative result.

The following describes the preserved historical work and reproduction commands.

Current status: the two-gate attention-output-change experiment is complete. The calculation passed, but the selector duplicated local repair on all held-out cases. See the [decision](docs/feasibility-assessment.md), [measurements](docs/feasibility-results.md), and [reproduction commands](experiments/feasibility/README.md). The validated original baselines are unchanged. A [bounded diagnostic](docs/selector-diagnosis.md) found correlated signals rather than an implementation fault; no replacement method was implemented.

The subsequent [bounded oracle prerequisite screen](docs/oracle-bound-diagnostic.md) retained all 12 candidates and found no full-correct/reuse-wrong case. Oracle enumeration stopped at that gate; factual repair headroom remains unestablished. The selector stays rejected and the validated attention calculation is preserved.

A small inference research package studying whether unchanged text after an edit can reuse its KV cache. CacheBlend provides the conceptual starting point. Direct edit-repair prior work exists; this increment builds trustworthy reference paths and an explicitly approximate failure probe. It does not reproduce CacheBlend's fusion algorithm or claim a new repair mechanism.

Start with [observed results](docs/development-results.md), [factual assessment](docs/assessment.md), or the [related-work review](docs/related-work.md). The [dependency explanation](docs/cache-dependencies.md) describes why suffix states can become stale.

## Run

Tested: Python 3.13.1, PyTorch 2.8.0, Transformers 5.2.0, tokenizers 0.22.2, safetensors 0.7.0. The actual environment is Apple M3, 8 CPUs, 16 GiB RAM, macOS arm64. MPS is built but unavailable to this process, so all recorded inference used CPU float32, eager attention, 4 threads, seed 17, eval and inference mode. See [diagnostics](artifacts/environment.json).

An editable package is installed in `.venv` using the existing system dependencies. From this directory:

```sh
.venv/bin/kvrepair diagnose
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/kvrepair demo --include-naive
.venv/bin/kvrepair report artifacts/demo.json artifacts/demo.md
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
KVREPAIR_LOCAL_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_local_model.py -q -s
```

For a fresh environment, `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e '.[test]'`. Core inference versions are pinned in `pyproject.toml`; [tested dependency versions](artifacts/dependencies.json) record their installed dependency closure. Installing dependencies may require internet access. The loader **always** uses `local_files_only=True`; it never downloads a model. If no checkpoint is cached, run `kvrepair demo --toy --include-naive` for a random tiny Llama functional smoke test. Toy outputs have no factual meaning. Alternatively pass `--model-path /existing/local/llama-checkpoint`. MPS may be requested explicitly with `--device mps` and fails clearly when unavailable; its numerical behavior is unverified here.

Default pretrained model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, pinned to cached revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Its standard Llama internals expose all per-layer K/V tensors and rotary positions. The cached unquantized checkpoint is 2.20 GB on disk; loading in float32 uses 4.40 GB / 4.10 GiB of parameter storage. This avoids quantization as a confounder, though float32 conversion does not restore precision absent in stored weights. It is large enough to give interpretable short answers but demonstrably unreliable on some prompts. No weights were downloaded.

For its 22 layers, 4 KV heads and head dimension 64, float32 KV requires `2 × 22 × 4 × 64 × 4 = 45,056 bytes/token` per batch-one cache. A 65-token snapshot uses 2,928,640 bytes; a 2,048-token cache uses 88 MiB. Multiple snapshots, temporary copies and attention/FFN activations add memory. These are storage calculations, not measured peak RSS. The adapter supports unpadded batch-one full-attention Llama only; no sliding windows, hybrid recurrent states, quantization or serving-engine integration. CPU is verified. LMCache's CUDA-oriented blender was inspected rather than installed.

## Edit a document

Copy `examples/development.json`, change its `old`, `new`, `question`, `old_answers`, `answers` and `should_change` fields, then:

```sh
.venv/bin/kvrepair demo --cases /path/to/my-cases.json --include-naive --output artifacts/my-run.json
```

Full prompts, including the chat template, are retokenized independently. Reuse accepts equal-token-length replacement envelopes only. Unequal lengths are explicitly rejected, recorded and excluded from metrics; no-edit reuse is supported. `examples/rejected.json` contains natural-word replacements that actually change token length. If every case is rejected the CLI exits 2, rather than claiming a passed exactness check. Held-out fixtures are separate in `examples/heldout.json` and have not been used for model evaluation.

`--include-naive` is required to run approximate suffix reuse. The safe demo default runs only full recomputation and exact-prefix reuse. Next-token logits must meet `atol=2e-4, rtol=2e-4`; greedy continuations must also match. Exact divergence produces exit 1 and an investigation trace. JSON records actual token positions/pieces, reused/recomputed ranges, raw answers/IDs, KL/TV, logit errors, first divergent step and same-prefix teacher-forced trajectories when outputs differ. Strict short-answer scores are formatting-sensitive; read the raw answers and separate factual baseline failures from reuse errors.

The cache API is `CacheEngine.run(ids, mode='full'|'prefix'|'naive', old=snapshot)` returning a Run with its owned Snapshot, work ranges and measured prefill time; `generate(snapshot, ...)` branches independently. Snapshots carry engine identity and must not be shared across different weights or modified in place. Full recomputation accepts arbitrary nonempty prompts; the two reuse modes enforce the initial equal-length scope.

Recorded timings include cache copies, splicing and model calls, but exclude tokenization, model loading and the separately reported original prefill. Continuation times and their sum with prefill are also recorded. These single fixed-order runs are functional observations, not a benchmark or evidence of proportional token-count savings. No GPU performance is inferred.

Tests cover no edit, early/middle/late edits, synthetic and real tokenizer boundaries, explicit insertion/deletion rejection, cache contents, independent storage, order independence and greedy agreement. Random-weight tests check mechanisms; opt-in pretrained tests check numerical behavior on the cached model. See [test output](artifacts/local-tests.txt).

The [experiment log](docs/experiment-log.md) records decisions. The standalone [local-window sweep](docs/next-experiment.md) remains paused; the authorized feasibility pilot used local repair only as a fixed-budget control. Its factual comparison is inconclusive because clean TinyLlama failed the dependent questions under the frozen protocol. Local commits were subsequently authorized; see the [history provenance](docs/version-history.md). Nothing was pushed, published or run on cloud compute.
