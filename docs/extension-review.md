> Follow-up: the authorized feasibility experiment is complete. See [the decision](feasibility-assessment.md). The numerical update passed; the tested selector duplicated local/attention-to-edit support on 24/24 held-out questions. This review remains the historical hypothesis, not an improvement claim.

# Extension review — 2026-11-08

The inference-method goal remains active. The local-window sweep is paused; replication is a necessary control, not the intended contribution. I recommend **one conditional extension hypothesis**, described below. It has a concrete operator change and an inexpensive rejection test, but neither novelty nor improvement is established. No repair code, new evaluation infrastructure or model experiment was added in this review.

## Mechanism map

Sources: [PatchKV, §§II–V, Eqs. 7–14](https://arxiv.org/html/2609.26219v1) and [Contiguity, method/results/discussion](https://arxiv.org/html/2609.17983v1). The comparison below describes their reported mechanisms; unavailable implementations prevent an execution audit of the originals.

| Dimension | PatchKV: edited-context recovery | Contiguity, Not Importance | Our validated foundation / paused proposal |
|---|---|---|---|
| Repair operator | Updated span plus block-rounded union of a local dirty window and nonlocal attention selections; restore the complement. | Recompute edit plus K selected positions layerwise; other states remain stale. | Full, exact-prefix, and approximate changed-span/suffix splice. Paused proposal adds a contiguous window. |
| Selection | Offline high-quantile predictor from prefix/edit/suffix lengths; stored old-checkpoint attention adds nonlocal blocks. | EditLocal versus structure, stale-query attention, layer-2 KV deviation and random selection. | Actual shared token prefix and changed envelope; no learned/attention selector. |
| Positions / representation | Aligned suffix permits length changes; correct key RoPE; calibrated FP16/K8V8/K8V4 transport. | Single equal-length replacement; fixed positions. | Equal-token-length envelope only; float32; unequal lengths rejected. |
| Evaluation | Qwen3-32B, GLM-4-9B, Llama-3.3-70B; three edited QA workloads; eight-A800 server; F1 and resume TTFT. Planning/offload costs reported separately. | Near-8B Llama/Qwen/Mistral; synthetic direct/derived facts in ~5K HotpotQA background; clean-solvable/gap filters; margin recovery and flip. RTX 5090 timings exclude fresh question. | TinyLlama 1.1B; CPU; six development prompts; next-token KL/TV/logits, continuations and manually assessed answers. |
| Important boundary | Fixed repair set; old attention; calibration dependence. Compression can retain information absent from the edited text. | Distant dependencies defeat EditLocal; even oracle carrier locations do not ensure recovery. Transplant is not recomputation. | No demonstrated reuse-induced factual regression yet. A local window would reproduce an established baseline. |

Additional qualifications: PatchKV's “exact” selected-block language is not a proof that sparse reconstruction yields full-prefill states: fresh selected queries can still read stale predecessors. Its Table I is an exact-state intervention, distinct from its recomputation ablation. The paper does not provide enough released artifacts to verify this distinction in code. Contiguity's numerical checks retain a failed Qwen value-tensor tolerance and report Mistral bfloat16 argmax changes resolved in float32. Neither paper establishes exactness of partial repair.

**Our protocol mismatch matters.** Contiguity caches the document and processes a fresh question. Our current naive run caches the complete old rendered question too, replaying only the final token. That remains a valid, deliberately naive staleness probe, but its drift is not directly comparable to Contiguity's edit-only baseline. PatchKV specifies checkpoint-time decode-query attention but does not fully specify query placement/freshness. My earlier statement that both papers definitely use a fresh question was too strong. Any future method comparison must define the document/query split consistently; current code remains unchanged.

## Implementation and reproducibility audit

I read both complete downloaded main texts, equations, tables and figure captions, and inspected relevant rendered PDF pages. The local reading copies are under `artifacts/review/`. Both arXiv source archives were also inspected: neither contains repair code, and Contiguity’s archive contains no supplementary-appendix file. Contiguity repeatedly refers to a supplementary appendix; its public eight-page PDF ends with references. Missing construction/threshold details must be marked unknown rather than filled in from a reconstruction. PatchKV's public text leaves details such as exact edited-dataset construction, cohort sizes and some calibration settings insufficiently specified for a faithful reproduction.

I checked paper/abstract external links and searched GitHub repositories by title, ID, authors and relevant terms. **I did not locate an original-author public implementation for either paper.** This is a search result, not proof none exists. In particular, [cusasak/PatchKV](https://github.com/cusasak/PatchKV) implements the different *Weight-Space Compensation of KV Cache* paper, not the edited-context method.

Available source inspected without execution or installation:

| Source | What the actual code establishes | Implication |
|---|---|---|
| [Third-party Contiguity reconstruction / extension, pinned commit](https://github.com/hadigghazi/kv-edit-repair/blob/30feaf7b8afaf856bb104578f78dcf5465b1cf42/extension/kvx.py) | `Runner.run` writes selected K/V per layer, attends over the full causally allowed cache, and runs residual/MLP computation. `evaluate` clones branches and appends the question freshly. `index_graph` starts with changed words from both old/new sentences and propagates rare names; graph endpoints set the end of a contiguous repair path. | An available reconstruction, not the missing original. Explicit-link dependency finding and edit-to-dependent-text repair are already implemented. Optional external LLM graph calls are present; none were invoked here. |
| [SUR allocation, pinned commit](https://github.com/yh0903/budgeted-kv-cache-repair/blob/d47ce251560b4b04ed035d90262a9c6470d1e7b9/src/kv_cache_repair/semantic_units.py) and [reference execution](https://github.com/yh0903/budgeted-kv-cache-repair/blob/d47ce251560b4b04ed035d90262a9c6470d1e7b9/src/kv_cache_repair/kernels/executor.py) | `score_semantic_units` aggregates top token scores; `select_semantic_unit_support` fills a region budget with a final partial window. The inspected reference executor runs maximal intervals in causal order and makes cache ownership explicit. | Region selection and interval execution are established. The release also offers a packed path; inspecting the reference file does not validate that path or published task results. |

These observations are from source inspection, not reproduction of those repositories' claims. Exact fetched files and revision identities are preserved locally. None of their code was imported into our package.

## Extensions screened out

I would not pursue “repair a larger local window,” “find the dependent paragraph,” “choose query-relevant semantic regions,” or “use different token sets at different layers” as our contribution without a substantially different mechanism. Besides the two target papers:

- [Budgeted KV-Cache Repair / SUR](https://arxiv.org/html/2610.17338v1) already allocates repair to query-relevant text regions and provides a layer-packed executor.
- [The third-party dependency extension](https://github.com/hadigghazi/kv-edit-repair/tree/30feaf7b8afaf856bb104578f78dcf5465b1cf42) already implements explicit-link and optional LLM dependency endpoints.
- [RelaxKV](https://arxiv.org/html/2609.33503v1), §3, already varies repair targets by layer and changes their available attention context.
- [KVEditBench](https://github.com/ShravanthReddy/KVEditBench/tree/1e392559be0391cdda9cf51f0a288e443c20b6a5) already releases local-window, length-changing and chained-edit experiments. This is a repository-level screening, not an independent validation of its measurements.

Likewise, feedback-driven selection alone is not enough of a distinction: [EchoKV](https://arxiv.org/abs/2611.05903) already studies query/workspace feedback for eviction, a related but different problem.

## One extension hypothesis: select repair regions using the edit's attention-output change

**Specific problem and example.** A cached order changes from Standard to Express. Several paragraphs later, a routing table maps Standard to ground delivery and Express to air delivery. A fresh question asks how the order will ship. A local window might miss that table; old query attention or lexical overlap might prefer the obsolete route or unrelated product descriptions. We want a signal that measures the effect of the particular replacement, including removal of the old contribution. This example is proposed, not a measured failure; complete-prompt token lengths must be checked.

**What existing methods handle.** Locality, nonlocal importance, shallow KV deviation, region coverage and explicit dependencies are already handled by the methods above. Attention to the edited sentence is also an implemented comparator in the third-party extension's `evaluate`. Merely renaming that signal would not be an extension. The narrow question is whether a value-sensitive old-to-new **attention-output difference**, computed without a full shallow-layer forward, offers a better cost/utility tradeoff.

**Mechanism to change.** Keep the ordinary target model and all-layer region-recompute operator. Replace the nonlocal region score with a cheaply computed first-layer output perturbation. At checkpoint creation, retain first-layer suffix queries Q, per-head attention outputs O and log-normalizers. For an equal-token-length replacement, the unchanged suffix's first-layer queries stay fixed: they come from unchanged local embeddings. Only the edited first-layer K/V columns change. Update their contributions to each suffix attention output, then score a fixed-size region by the sum of per-token `||O_new - O_old||₂`. Mandatory edit computation is outside the region budget; rank non-overlapping equal-size regions deterministically and repair a fixed number of them. No oracle answers, clean-cache comparisons or gold dependency locations may enter selection. Run a fresh question against the resulting mixed cache.

This changes the **selection signal**, not the transformer weights or attention mask used during actual repair. The predicted first-layer change is not itself a repaired deep KV cache. Its usefulness for selecting deeper repairs is the hypothesis. Better localization alone does not resolve the stale-input operator gap; the correctly located carrier can still require additional predecessor repairs. A failure there would reject this selector as a sufficient solution, even if it predicts first-layer changes perfectly.

For one unchanged query/head, write `s_j = q·k_j/√d`, `Z = Σ exp(s_j)` and `O = Σ exp(s_j)v_j / Z`. For changed positions E, the elementary update is:

```
Z' = Z + Σ_E [exp(s'_j) - exp(s_j)]
Z'O' = ZO + Σ_E [exp(s'_j)v'_j - exp(s_j)v_j]
```

Only causally visible edited positions enter each row. This is an algebraic identity for fixed Q; practical code would use rebased/log-domain statistics and fall back to direct row attention if subtraction is unstable. It is **not** exact for deeper layers whose Q has changed. Insertions/deletions, repeated repairs from mixed caches and model changes are outside the initial hypothesis.

**Extra work and storage.** With N suffix tokens, m changed tokens, H query heads and head dimension d, the score computation costs approximately O(NmHd), plus edited K/V projection, vector differences and region aggregation. It avoids a full N×N attention recomputation and a suffix-wide MLP merely to obtain a score. It still scans every saved Q/output row, and actual selected-region repair costs remain. Checkpoint instrumentation, cloning, fallback rows and memory traffic must all be charged. This could cost more than it saves.

For TinyLlama float32, saved Q and O each use 8,192 bytes/token; log-normalizers use 128 bytes/token. Added metadata is approximately **16.125 MiB at 1,024 tokens**, before allocator/copy overhead, versus 44 MiB for one ordinary 1,024-token KV cache. Without saved Q, recreating all suffix queries adds projection work; that alternative cannot be described as free.

**Prior-work boundary.** The identity and incremental attention arithmetic are not new. [Sharir and Anandkumar (2023)](https://arxiv.org/html/2307.14988), especially §3, already investigate incremental transformers for edits and warn about softmax precision; their architecture uses VQ and replaces softmax. [PRR](https://arxiv.org/html/2606.30389), §4.3, already repairs partial attention with online-softmax statistics for sparse decoding. The candidate here is using a fixed-weight, first-layer edit-output delta **as a repair-region selector**. I did not find that precise combination in this search; that is insufficient to claim novelty.

**Smallest supporting or rejecting experiment, if authorized later.** Use two gates, with no new benchmark framework:

1. On six short prompts with no edit and equal-length replacements, compare the statistics-based first-layer output against direct first-layer recomputation. Vary edit strength and position. Fail on unexplained float32 mismatch, unstable cancellation or expensive frequent fallback. Passing only establishes correct arithmetic, not a useful inference method.
2. Use eight development cases to freeze one region size and budget, then 24 new untouched cases across short/medium dependency distances, paraphrases, distractors and unchanged-answer controls (maximum 1,024 tokens). Keep every clean-model failure visible; report results both unconditionally and on cases whose old/new full recomputations solve the task. Compare the same region geometry/operator under delta scores, old attention-to-edit mass, fresh-query attention and shuffled scores. Include EditLocal, edit-only, exact-prefix and full references; include a matched shallow-deviation selector before claiming an advantage over CacheBlend-style selection. Oracle caches are scoring references only.

Support requires better answer preference/known-answer performance as well as distribution fidelity at matched repair work, without new invariant-answer errors, and a plausible total-cost tradeoff after selector/checkpoint overhead. Similar performance to old edit-attention, gains confined to explicit word matches, improvement only after more repaired tokens, or lower KL without useful answer recovery would reject or sharply narrow the hypothesis. These 24 cases could support a pilot decision, not a general efficacy claim.

**Hardware feasibility.** The numerical gate and small TinyLlama pilot fit the inspected M3/16-GiB CPU environment in principle: weights are 4.10 GiB; short KV copies and the proposed single-layer metadata are modest. Capture/release first-layer statistics without retaining all layers' dense attention maps. Peak memory and runtime remain to be measured. Our installed model and PyTorch support the required operations; the current cache interface does not expose these statistics, so an isolated experimental adapter would be needed later. MPS is unavailable. This hardware cannot reproduce the papers' large-model/CUDA/offload performance claims. TinyLlama's clean-task reliability may prevent an informative answer-level test even when the mechanism runs.

## Recommendation

Keep the inference direction, but authorize at most the numerical/signal feasibility gate next, not a larger window sweep or a training project. I am not proposing a second extension: the simpler alternatives found close prior implementations, and adding a weak second idea would obscure the decision.

If the first-layer signal cannot distinguish consequential updates from harmless ones, or its metadata/selection cost removes the benefit, abandon this selector. If TinyLlama cannot supply enough clean-solvable cases, stop the local efficacy claim; a separately authorized stronger model or a different inference problem would then be needed. The foundation remains useful, but replication and a failure map alone will not be called the final contribution.
