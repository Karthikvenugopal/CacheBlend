# CacheBlend mechanism and quality reproduction

Internal learning work. The edit-repair direction remains closed in its [separate final archive](../../archives/kvrepair-final-2026-11-09/README.md). No novelty or serving speedup is claimed.

Completed all 40 fixed items × two configurations × five methods: 400 generated outputs, with no dropped items or configurations. The implementation demonstrates actual selective recomputation and a work–divergence tradeoff, but gold-quality gains are mixed and full-model quality is weak. It does not establish the paper’s quality-preservation result on this model.

At 16% document selection, actual QKV work is 24.0–25.7% of full. Word F1 is 0.203/0.135 for 4×128/8×128, versus naive reuse’s 0.135/0.138. The 32% setting uses 38.5–39.9% QKV work and reaches 0.222/0.143 F1, versus full recomputation’s 0.209/0.147. Mean initial next-token KL falls from 7.72/9.26 under naive reuse to 2.14/3.30 at 16% and 1.32/2.01 at 32%. Even at 32%, normalized answer agreement with full is only 50%/60%. These results show approximation effects, not reliable preservation of full behavior.

Paper: [arXiv 2405.16444v3](https://arxiv.org/html/2405.16444v3). Reference: [YaoJiayi/CacheBlend, 55ad026](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581). Cached Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; MPS float32/eager, deterministic greedy decoding, 32-token cap. [Protocol and source mapping](../../experiments/cacheblend/protocol.md).

The fixed pilot contains the first 20 MuSiQue and first 20 2WikiMQA records from the pinned release, with both first-4 and first-8 retrieved context configurations, capped at 128 Qwen tokens per context plus labels/separators. No item or configuration was dropped or selected by outcome. All 80 input assemblies, raw source records and truncations are in [pilot.json](../../experiments/cacheblend/pilot.json). Dataset access and downloads were confirmed by the user before fetching.

Composition uses independent chunk prefills and native-coefficient `R(global) R(local)^-1` correction of post-RoPE K. V is unchanged. Every method uses the same assembled token IDs, one native chat wrapper, and computes the entire question/assistant-prefix suffix fresh at every layer. Source identity, token identity, immutable cache ownership and independent generation branches are enforced.

The [milestone](milestone.json) validates every chunk/layer correction against captured pre-RoPE keys, first-layer K/V, first-prefix logits, native full/cache replay, native naive suffix execution, custom full logits/KV and independent storage on both configurations at atol=rtol=2e-4. The initial offset-only rotation failed; its [failed run](failed-milestone-1/milestone.json) is preserved. Five deep prefix V tensors still fail the same mixed tolerance in two native executions with different total attention shapes. First-prefix logits pass. These raw tensors are not claimed numerically equivalent. The numerical collection gate was refined after diagnosis; see the protocol. No CPU–MPS equivalence claim is made.

Reference V-HKVD selects once at zero-based layer 1 using summed squared V deviation from independently cached values. Layer 0 runs fully; layer 1 refreshes all QKV but only selected attention/MLP rows survive. Deeper layers recompute selected rows and splice them into owned stale KV. Ratios .16 and .32 were fixed before inference. This follows the release schedule, with a corrected absolute-position causal mask and full fresh suffix. It does not implement the paper’s underspecified gradual filtering. [Full-selection endpoint checks](selection-endpoint.json) compare the ratio-1 reference path to native full.

**Zero selection is separate from naive reuse.** It still refreshes document QKV in layers 0 and 1 and document attention/MLP in layer 0. Naive reuses all document KV on every layer. Mandatory fresh suffix work is included in every fraction.

| Contexts | Method | Gold EM | Gold word F1 | Answer agreement | Next-token agreement | KL(full‖method) | TV | QKV rows / full | Attention+MLP rows / full | Truncated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x128 | full | 12.5% | 0.209 | 100.0% | 100.0% | 0.0000 | 0.0000 | 100.0% | 100.0% | 0/40 |
| 4x128 | naive | 2.5% | 0.135 | 2.5% | 7.5% | 7.7218 | 0.8663 | 4.9% | 4.9% | 3/40 |
| 4x128 | zero | 2.5% | 0.131 | 2.5% | 10.0% | 7.4078 | 0.8517 | 11.7% | 8.3% | 4/40 |
| 4x128 | hkvd_16 | 15.0% | 0.203 | 42.5% | 42.5% | 2.1381 | 0.4469 | 25.7% | 22.9% | 0/40 |
| 4x128 | hkvd_32 | 15.0% | 0.222 | 50.0% | 60.0% | 1.3212 | 0.3454 | 39.9% | 37.6% | 0/40 |
| 8x128 | full | 7.5% | 0.147 | 100.0% | 100.0% | 0.0000 | 0.0000 | 100.0% | 100.0% | 0/40 |
| 8x128 | naive | 0.0% | 0.138 | 0.0% | 7.5% | 9.2589 | 0.9148 | 2.5% | 2.5% | 9/40 |
| 8x128 | zero | 0.0% | 0.138 | 0.0% | 7.5% | 8.8984 | 0.9144 | 9.5% | 6.0% | 8/40 |
| 8x128 | hkvd_16 | 7.5% | 0.135 | 50.0% | 52.5% | 3.3022 | 0.4879 | 24.0% | 21.1% | 0/40 |
| 8x128 | hkvd_32 | 10.0% | 0.143 | 60.0% | 57.5% | 2.0062 | 0.3977 | 38.5% | 36.1% | 0/40 |

Each row includes all 40 items; scores, divergence and work fractions are means over those items. Gold scores use normalized whole-output EM and word-overlap F1 (best gold alternative); this differs from the release’s tokenizer-token F1 and yes/no parsing. Whole-output scoring penalizes elaboration; EM losses are score transitions, not manually adjudicated factual errors. Inspect the raw answers. Answer agreement means normalized agreement with full; exact generated-ID agreement is also saved. KL/TV compare the initial next-token distributions on the identical prompt, not later divergent generation prefixes. Truncated outputs remain scored and flagged.

Before selection, full recomputation matched gold exactly on 8/80 pairs. Naive changed 79/80 normalized answers. Paired EM losses: 8; paired EM gains: 1 relative to full. The [pilot review](pilot-review.json) was recorded before the selection run. No workload or ratio was tuned after it.

For 4x128, a normalized gold string occurs in 15/40 assembled inputs; 160 retained contexts were truncated and 240 source contexts omitted. String presence is only a coverage heuristic, not proof of answerability. Missing evidence and small-model errors limit gold-quality interpretation.
For 8x128, a normalized gold string occurs in 17/40 assembled inputs; 320 retained contexts were truncated and 80 source contexts omitted. String presence is only a coverage heuristic, not proof of answerability. Missing evidence and small-model errors limit gold-quality interpretation.

The results characterize this shortened retrieval pilot on Qwen3-1.7B, not the paper’s model/workload quality. Improvement in KL or agreement alone is not gold-answer improvement. Full recomputation is the agreement reference, not an oracle gold solver.

Work fractions count projected token rows and attention/MLP rows separately; they are not measured FLOPs. A native full reference is computed for evaluation; neither its cache nor its logits feed selection. That evaluation overhead is excluded from method work and timing. Prefill timings in the raw records also exclude independent chunk prefill, model loading, validation and generation, and include the custom executor’s cache copies. These local fixed-order observations do not reproduce CUDA serving TTFT, throughput, SSD retrieval or overlap speedups.

Evidence: [baseline outputs](baseline.jsonl), [selection outputs](selection.jsonl), [summary](summary.json), [completeness and token/work audit](audit.json), [source manifest](source-manifest.json), [archive verification](edit-repair-archive-verification.json). The implementation is [core.py](../../experiments/cacheblend/core.py); meaningful tiny-model mechanism tests are [test_cacheblend.py](../../tests/test_cacheblend.py). All failures, items and configurations remain available.
