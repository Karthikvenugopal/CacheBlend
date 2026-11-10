# Reproduction-target assessment — 2026-11-09

Decision: **shelve method development now**. Preserve the archived foundation, reject the selector, and stop TinyLlama factual sweeps. There is a plausible small-model published failure to reproduce, but no meaningful failure has been reproduced on our available checkpoint/backend. A candidate is not evidence that we can improve it. No new inference implementation, model download or task run was performed in this review.

## Resources actually checked

Apple M3, eight CPUs, 16 GiB unified RAM, PyTorch 2.8.0 and Transformers 5.2.0. Inside the execution sandbox, PyTorch reports MPS built but unavailable. **Outside the sandbox MPS reports available, and a one-element tensor addition returned `[2.0]`.** The earlier CPU results remain valid observations of their environment; they do not measure GPU performance. The new check establishes basic GPU access, not support for our cache adapter or numerical primitive. In particular, its float64 statistics path is not validated on MPS. Future approved GPU commands require execution outside the sandbox rather than a model-code setting alone. See [recorded diagnostics](../artifacts/reproduction-review/environment.json).

The local cache includes TinyLlama, DialoGPT-medium and an MLX-community Qwen3-4B 4-bit checkpoint (2,263,022,529 weight-file bytes, revision recorded in diagnostics). Neither unquantized Qwen3-1.7B nor Qwen3-8B is cached. Encoder/classifier checkpoints do not substitute for causal generation. MLX quantized weights are not a drop-in checkpoint for the inspected PyTorch repair code. Dequantization would not recover the original weights. No MLX port or quantization-controlled experiment is authorized here.

## Candidate comparison

| Target | Clean reference and documented failure | Reproduction access | Local decision |
|---|---|---|---|
| SUR / Budgeted KV-Cache Repair, Qwen3-8B on Erasing-NIAH | Table 2 reports re-prefill and replay 98.96% versus position-corrected suffix reuse 0% on 96 cases. Two same-key records, delete one, return the remaining values. | Released allocator/executor; task details in paper, but release excludes datasets/results and is not a complete experiment harness. | Stronger clean-reference evidence than our tasks; unquantized 8B plus cache/workspace is not a practical 16-GiB target. |
| Contiguity, Qwen3-8B on derived alias/lookup edits | Retains clean-solvable cases with a stale/oracle margin gap; distant answer-bearing text defeats edit-local repair. | No original-author implementation located in this search; available reconstruction explicitly identifies itself as a separate course project. Missing public construction details still matter. | Fits the original question closely, but 8B memory and incomplete reproduction materials block recommending it locally. |
| PatchKV, GLM-4-9B edited QA | Evaluates selective recovery against full suffix recomputation; exact reproduction needs profiling, edited workloads and transport settings. | Prior inspection found insufficient released construction details and no original edited-context implementation; the same-name compression repository is a different method. | At least 9B weights and an eight-A800 evaluation setting are inappropriate here. |
| Models Take Notes at Prefill, Qwen3-1.7B policy decisions | Released eight-task record: full reference 7/8 correct preferences, field-only 1/8; six paired full-correct/field-wrong cases. | Original-author task templates, alignment/cache machinery and results available at a pinned revision. | Best **conditional** small-model candidate; memory plausible, checkpoint missing, readout and backend caveats below. |

Sources: [SUR paper, §§4.1–4.2 and Table 2](https://arxiv.org/html/2610.17338v1), [SUR release scope and requirements](https://github.com/yh0903/budgeted-kv-cache-repair/blob/d47ce251560b4b04ed035d90262a9c6470d1e7b9/README.md), [Contiguity setup and distance study](https://arxiv.org/html/2609.17983v1), [third-party reconstruction identity](https://github.com/hadigghazi/kv-edit-repair/blob/30feaf7b8afaf856bb104578f78dcf5465b1cf42/README.md), [PatchKV evaluation](https://arxiv.org/html/2609.26219v1). These are author/release reports, not local replications. KVEraser was also checked: it uses Qwen3-8B and a learned eraser with two-stage training, so it does not remove the resource barrier. [KVEraser setup](https://arxiv.org/html/2606.17034v1).

## Small-model candidate: what the source actually establishes

Pin `19PINE-AI/programmable-kv` at `e9085eafcc6c83e60c548de69060c4bd5b210c96`. For Qwen3-1.7B without reasoning, the published paired failures are retail refund, airline cancellation, bank withdrawal, document access, customs routing and on-call routing. Deployment is correct under both paths; the allergy example fails even under full recomputation. Retain those controls rather than presenting selected successes as overall reliability. The eight-case clean estimate has a wide reported interval (0.53–0.98). [Released result record](https://github.com/19PINE-AI/programmable-kv/blob/e9085eafcc6c83e60c548de69060c4bd5b210c96/results/mech_diverse_qwen3_1p7b.json).

Example: changing an account from administrator to suspended should change a refund decision to escalation. The documented failure is that old downstream contextual states preserve the earlier decision despite refreshing the field. The paper also reports errata and reasoning interventions; these already address parts of the problem. An erratum changes the prompt, and is not identical-prompt cache repair. [Paper, §§3–4 and Appendix B](https://arxiv.org/html/2606.17107v1).

Important implementation qualifications from reading the pinned files:

- `esys/mech_diverse.py` uses **argmax between the first tokens of two predefined alternatives**. Its 7/8 does not establish seven correct unrestricted generated answers. `ftok` takes the first token; token identity and other vocabulary alternatives require checking before interpreting answer utility. The reported erratum 8/8 uses the same limited readout.
- Its field-only path copies edited-field rows from a full clean cache. For a single aligned edit with an unchanged prefix this is a useful intervention, but it is not a demonstrated low-cost execution path.
- `e1/align.py` inserts space-token padding into the shorter edited span. The reproduced prompt is this aligned token sequence, not necessarily the plain retokenized document. The existing baseline's unequal-length rejection must remain unchanged.
- `esys/selective_K_sweep.py::patched` copies **selected downstream states from the clean cache**. Consequently its selective curve is a transplant diagnostic, not evidence for sparse recomputation with stale predecessors. This prevents treating its results as a deployable repair target.
- `esys/mech_suite.py` hardcodes CUDA device creation and BF16 loading. MPS/CPU would require a separate, checked adaptation; the released code cannot simply be run unchanged here. Its broad dependency bounds also do not establish compatibility with our Transformers version.

See the [pinned driver](https://github.com/19PINE-AI/programmable-kv/blob/e9085eafcc6c83e60c548de69060c4bd5b210c96/esys/mech_diverse.py), [alignment](https://github.com/19PINE-AI/programmable-kv/blob/e9085eafcc6c83e60c548de69060c4bd5b210c96/e1/align.py), [selective intervention](https://github.com/19PINE-AI/programmable-kv/blob/e9085eafcc6c83e60c548de69060c4bd5b210c96/esys/selective_K_sweep.py), and [device loader](https://github.com/19PINE-AI/programmable-kv/blob/e9085eafcc6c83e60c548de69060c4bd5b210c96/esys/mech_suite.py). Downloaded text was inspected only, never imported or executed. Hashes and one failed guessed filename are retained in the [reading manifest](../artifacts/reproduction-review/reading-manifest.json).

## Memory feasibility, not a performance forecast

Weight estimates below use nominal parameter counts, exclude runtime/OS memory, and are not measured peaks. For Qwen3-1.7B, 28 layers, eight KV heads and head dimension 128 imply `2 × 28 × 8 × 128 × bytes_per_element` per cached token. [Official architecture configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json).

| Configuration | Weights only | One 2,048-token cache | Assessment |
|---|---:|---:|---|
| Qwen3-1.7B FP32 | ~6.33 GiB | 448 MiB | Plausible short-context CPU/MPS functional target with bounded independent branches; peak unmeasured. Three caches add ~1.31 GiB. |
| Qwen3-1.7B BF16 | ~3.17 GiB | 224 MiB | Paper precision; actual MPS operation support/numerical behavior unverified. |
| Nominal 8B BF16 | ~14.90 GiB | Additional | Too little headroom within 16 GiB for OS, activations, copies and weights. Qwen3-8B has 36 layers and eight KV heads. |
| Nominal 9B BF16 | ~16.76 GiB | Additional | Weights alone exceed physical RAM. |

The [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) gives its cache dimensions. Eager attention maps, logits, weight-conversion temporaries and simultaneous snapshots can materially increase memory; retain only needed tensors and measure peak before claiming fit. MPS shares the same RAM and does not create more capacity. The papers' large CUDA servers are experimental settings, not proof that small-model correctness reproduction requires those GPUs; conversely this Mac cannot reproduce their serving results unchanged.

## Continue or shelve

There is a concrete **operator limitation to verify**, rather than another selector to propose: restoring a useful position from a clean cache can succeed while recomputing it against stale predecessors fails. That gap is already studied by Contiguity; it is not our novelty, and the inspected small-model transplant sweep does not establish remaining factual headroom for improving it.

Under currently available checkpoints, no target passes all gates: suitable unquantized weights, reliable full-reference answer utility, a reproduced reuse-induced error, and a locally established repair limitation. **Shelve this project now.** MPS availability changes the execution option, not this evidence threshold.

Reopening would require separately authorized access to the unquantized Qwen3-1.7B checkpoint and an explicitly bounded reproduction of the existing eight-task result, including unrestricted answer validation and the documented alignment semantics. Only after that failure and an actual-recomputation limitation reproduce should method development be reconsidered. This is a condition for reconsideration, not a proposed selector, implementation plan, new factual sweep or benchmark project. No downloads or runs are queued.
