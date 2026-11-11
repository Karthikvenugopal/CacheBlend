# CacheBlend reproduction

Read [protocol.md](protocol.md) before running. The fixed pilot is `pilot.json`; the previous edit-repair code and archive stay separate. All model loading is offline. This adapts the pinned release's single-check V-HKVD schedule to native Qwen3 with an absolute-position causal mask and an entirely fresh question suffix. It does not reproduce the paper's unspecified gradual-filtering schedule or the release's mask/suffix defects.

The completed [report](../../artifacts/cacheblend/report.md) includes all 400 outputs and both configurations. The [audit](../../artifacts/cacheblend/audit.json) passed; observed gold-quality gains are mixed. No additional experiments are queued.

- `prepare.py`: freeze the first 20 records of each approved downloaded workload and both chunk configurations. Do not rerun to change inputs after seeing results.
- `core.py`: independent prefills, position correction, full/naive/reference execution, independent greedy generation branches and numerical statistics.
- `run.py`: numerical milestone, full/naive pilot, gated selection pilot, aggregate summary. Each completed pair is appended and never replaced on resume.
- `diagnose_composition.py`: preserve and diagnose any native MPS tensor discrepancy at the unchanged tolerance.
- `verify_archive.py`: verify the separately preserved edit-repair archive.

Historical run commands, using the existing environment and cached weights:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/cacheblend/run.py milestone
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/cacheblend/run.py baseline
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/cacheblend/run.py selection
.venv/bin/python experiments/cacheblend/run.py summary
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests/test_cacheblend.py -q
```

MPS is hidden inside the execution sandbox; GPU runs need the approved outside-sandbox execution. The runner refuses a CPU fallback. Test fixtures use tiny random Qwen3 weights locally and carry no answer-quality claims.
