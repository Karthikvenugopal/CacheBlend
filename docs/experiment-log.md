# Experiment log

## 2026-11-08 — E0: scope and environment

Hypothesis: selective recomputation after a document edit may be a useful research direction, but the general mechanism may already exist.

Change: inspect the empty workspace, installed libraries, cached weights, hardware APIs, CacheBlend/Cache-Craft, a version-pinned LMCache blender and recent primary papers.

Evidence: existing unquantized TinyLlama checkpoint; Python 3.13.1 / torch 2.8.0 / transformers 5.2.0. Hardware queries confirm Apple M3, 8 CPUs, 16 GiB RAM; MPS available=False. Direct overlap in Contiguity, Not Importance and PatchKV, plus downstream-note editing and persistent-history studies.

Decision: build a small CPU float32 reference adapter. Narrow the research to a replication and failure-boundary study; do not claim edit repair or contiguous windows as novel. Do not install LMCache serving infrastructure. No downloads, paid services or cloud work.

## E1: exact-prefix correctness and ownership

Hypothesis: after retokenizing the entire prompt, the identical token prefix is reusable under unchanged weights, masks and positions; downstream recomputation should match a fresh forward numerically.

Change: owned snapshots, copy-on-entry DynamicCache, explicit suffix positions and total-length mask, eval/inference mode, fixed seed/threads, no-edit final-logit reuse. Add no-edit, first/early/middle/late edit, token-seam and mutation-isolation tests.

Evidence: 13 focused mechanism tests pass. Six opt-in pretrained tests pass in 15.86 s, including actual SentencePiece `playing` → `played` boundary handling. Short-prompt logit differences up to 2.20e-5 are below the predeclared float32 `atol=rtol=2e-4`; greedy continuations agree. Independent storage and unchanged original caches verified after naive/decode branches. Captured output: `artifacts/local-tests.txt`.

Decision: exact baseline is an adequate functional gate for this narrow adapter. Do not generalize backend tolerances to MPS, different dtypes, models or long contexts without validation.

## E2: naive stale-suffix probe, first development run

Hypothesis: rebuilding changed tokens while retaining downstream states can change the next-token distribution, even if generated text sometimes agrees.

Change: approximate suffix splice with mandatory final-token replay. Explicit opt-in; no repair selector. Run direct and derived factual edits plus unchanged-answer questions.

Evidence: `artifacts/development-v1.json` has four accepted cases with naive logit drift and two changed continuations. Two museum nine → noon cases were rejected because complete retokenization changed prompt length. Toy UTF-8 byte tokenizer likewise rejects red → blue due to different byte length; its toy outputs are never used as language evidence.

Decision: retain the rejected fixtures in `examples/rejected.json`. Replace only the development museum pair with nine → ten after inspecting complete tokenization; both use equal token lengths. This is development fixture correction, not a method improvement. Preserve the initial run. No held-out inference.

## E3: supported development suite

Hypothesis: the naive baseline exposes measurable distribution error across more than one edit/example, but greedy disagreement need not imply a wrong fact.

Change: rerun six supported development cases, same model/seed/configuration. Include original clean outputs, raw generated tokens, work ranges, KL/TV and teacher-forced traces for divergent continuations. Add static reporting, token-limit metadata, and an explicit no-accepted-cases exit status. These reporting changes do not change inference; existing JSON gained derived metadata without rerunning model outputs.

Evidence: all six full/prefix logits and continuations agree. Naive max-logit error 0.133–0.739, KL 0.000646–0.003798, continuation agreement 4/6. Shipping full-recompute output fails to answer within 12 tokens; naive gives the correct invariant answer. Derived old clean answer is wrong, making repair attribution uninformative. Strict short-answer scores penalize verbose correct answers; manual factual assessment is separate in `docs/assessment.md`. CPU median prefill times including copies are 223.3/208.6/190.3 ms (full/prefix/naive), single fixed-order measurements only.

Decision: retain mixed/negative evidence. No demonstrated reuse-induced factual regression yet on this development suite. A replicated contiguous-window sweep could test distribution fidelity and downstream dependence, but it is a proposal only. Freeze its selection/metrics before held-out evaluation. Pause here as requested.

## E4: extension review; local-window experiment paused

Hypothesis: a method contribution may remain possible beyond already covered edit-local/region/layer selection.

Change: documentation and read-only literature/source inspection only. Read both target papers in full, checked available code identities, inspected pinned third-party repair and SUR source, and screened adjacent incremental-attention work. Preserve all inference code, fixtures and earlier results.

Evidence: no original-author public implementation located for the two targets; one same-name PatchKV repository is unrelated. Contiguity uses a fresh question, unlike our deliberately naive complete-prompt cache; PatchKV query freshness is insufficiently specified. Released adjacent work already covers several obvious extensions.

Decision: pause the old local-window sweep. Propose one conditional first-layer attention-output-delta selection hypothesis, with numerical and answer-utility rejection gates and explicit metadata/selector costs. No novelty or improvement claim. Replication remains a control rather than the user’s final research objective. Details: docs/extension-review.md. No experiment or infrastructure was implemented in this review.

## E5: two-gate feasibility implementation and development

Hypothesis: the exact first-layer attention-output change from an equal-token-length edit may provide a useful repair-region selector without a full shallow-layer recomputation at edit time.

Change: add an isolated `repair_probe` adapter, two-gate runner and focused tests; leave the original engine, CLI and baseline tests unchanged. Gate 1 retains Q/O in float32 and normalization statistics in float64, accounts for the extra metadata score pass, and compares the update to native first-layer attention. Gate 2 processes questions freshly and applies every selector to the same 16-token region grid and 32-token extra repair budget.

Evidence: Gate 1 passed six pretrained cases with maximum absolute error 5.96e-8 and zero observed fallback rows. Twenty-one mechanism/baseline tests passed, including eight new probe tests. Native all-position endpoint checks passed. A short development fixture lacked enough downstream tokens for the frozen-sized budget; neutral context was added before freezing and the initial partial development run was preserved. Final development completed eight questions over three document edits. Original/edited clean answer matching succeeded in 6/8 pairs, with substantive two-hop and invariant-question failures.

Decision: freeze the method, scoring and case definitions before held-out inference. No factual improvement claim follows from development. The held-out pilot was paused after two complete questions at the user's explicit request; the process was terminated and results saved.

## E6: authorized continuation after the user pause

Change: resume only unfinished held-out cases with a separate checkpoint driver. Verify all frozen inference-source/configuration/dataset fingerprints and dependency versions; retain the original partial artifact and completed records. Selection, repair, prompts, scoring and method order remain unchanged. Execution-session metadata distinguishes measurements taken before and after the pause.

Evidence: the completed held-out pilot accepted 24 questions over eight document edits. Frozen inference source/config/data and all original baseline hashes remain unchanged. Completed pre-pause records were preserved verbatim. Delta selected the same support as local and attention-to-edit on 24/24 cases; each scored 13/24 factual matches and preserved 13/13 clean-correct answers. Full recomputation solved 0/8 derived cases. Median delta selection was 2.14 ms versus 0.22 ms for attention and 0.011 ms for local; metadata and end-to-end costs are recorded separately.

Decision: reject this selector unchanged while retaining its validated numerical primitive. Factual repair efficacy is inconclusive with TinyLlama and this frozen 12-token response protocol. No reliable speedup or novelty claim. Detailed numerical, utility, cost and manual-audit evidence is linked from docs/feasibility-assessment.md. No further experiment was started.

## E7: bounded diagnosis of identical selector choices

Question: did coupling, indexing, ties or mathematical equivalence cause delta to duplicate local/attention selection?

Change: source inspection, independent ranking/offset controls, three controlled causal-attention examples and score-only replay of the 11 distinct original development/held-out document edits. No answer generation, tuning, new benchmark questions or changes to frozen inference source/results.

Evidence: independent native delta score error <=2.24e-8; full-softmax old-mass discrepancy <=7.45e-9. The selector responds to distinct injected nonlocal peaks. No cutoff ties; reversing exact-tie preference has no effect. Token ranks differ in all 11 groups (Spearman 0.972–0.999), while region-sum ranks are identical and strictly ordered by edit distance. Controlled examples reverse selections through key changes and through head-profile differences, proving general non-equivalence.

Decision: keep the selector rejected and preserve the validated update. No implementation defect was found in audited paths. The tested first-layer signal plus aggregation did not create a different allocation; coarse/forced choices and weak clean-model utility limit the experiment. Factual efficacy remains inconclusive. No revised mechanism, speedup or novelty claim. Details: docs/selector-diagnosis.md.

## E8: bounded oracle prerequisite screen

Hypothesis: a separate short-document development set may expose full-correct/reuse-wrong cases suitable for an exhaustive one/two-token repair oracle, revealing whether useful headroom over local repair exists.

Change: fix 12 candidates before inference, six templates in both directions, with complete suffixes capped at 16 tokens. Retain all three screening arms and manually audit all outputs. Conditional search is capped at two cases and 272 choices; labels would score complete enumeration afterward. Existing inference source and earlier experiments remain unchanged.

Evidence: all 12 candidates supported; full recomputation correct on four, with reuse preserving all four. Other full outputs comprise four incorrect answers and four nonanswers. Zero qualifying reuse-induced answer errors. Screening took 49.72 seconds after loading. Raw outputs and actual token positions retained; frozen source hashes verified.

Decision: stop before oracle enumeration as required. No small-repair recovery, advantage over local, or useful repair positions established. This screened setup is unsuitable for factual repair evaluation, and factual efficacy remains inconclusive. Selector stays rejected; validated attention calculation preserved. No replacement method, speedup or novelty claim. Details: docs/oracle-bound-diagnostic.md.

## E9: archive and reproduction-target assessment

Change: archive committed source/results with a SHA-256 manifest; preserve the working source. Stop selector development and TinyLlama factual sweeps. Review existing paper/release targets without model downloads or new model inference. Confirm basic MPS tensor access outside the sandbox; preserve earlier CPU measurements.

Evidence: the strongest small-model candidate is the original-author Qwen3-1.7B gated-decision release, with seven of eight clean binary preferences correct and six paired reuse regressions. Source inspection finds a restricted token readout, artificial length alignment, clean-cache downstream transplantation and CUDA-specific execution. The unquantized checkpoint is absent locally. Its short-context memory estimate is plausible; larger 8B/9B unquantized targets are impractical within 16 GiB. None is a locally reproduced factual repair failure with demonstrated improvement headroom.

Decision: shelve method development now. Retain the small-model target only as a conditional reopening candidate; do not implement, download, sweep, or optimize a selector. This is not a successful repair method or a benchmarking project. Details: docs/reproduction-target.md and archives/kvrepair-2026-11-09/manifest.json.

## E10: prepare six published Qwen3-1.7B pairs, pending authorization

Change: the user authorized preparation of a bounded reproduction, with download and pretrained execution deferred. Freeze the six existing published regression pairs, complete prompt bodies/renderings, the field-only clean-state intervention and restricted first-token readout. Add a separate offline Qwen3 runner and a pinned, explicitly bounded downloader. Preserve the archived Llama source unchanged.

Evidence: author code/result do not record a model revision. Official candidate pin is 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; this is not author-confirmed. Official assets total approximately 4.08 GB. Meta-device construction counts 1,720,574,976 unique tied parameters, 3.205 GiB in BF16. Tiny random-Qwen3 BF16 API checks passed on CPU and MPS, with zero observed native/cached logit discrepancy and independent cache ownership; no pretrained assets were downloaded or loaded. Pretrained tokenizer/precision/backend fidelity remains unverified.

Decision: await explicit approval for one pinned download and the two fixed six-case stages. Keep restricted preference reproduction separate from unrestricted deterministic generated-answer errors. Preserve all outcomes; no prompt replacement, broader sweeps, repairs or selectors. If generated regressions do not reproduce, stop and shelve. Protocol and exact commands: experiments/qwen_reproduction/protocol.md.

## E11: authorized download and Qwen3 reproduction attempt

Change: download exactly 4,079,434,577 bytes at the approved revision, verify files and all frozen chat renderings, and launch restricted scoring on MPS/BF16. Retain the unknown author revision. A pre-case identity guard found separate embedding/output parameters; byte-identical payload checks justified restoring the config-declared alias without changing values. Preserve the original preparation and failed launch.

Evidence: the first factual case then failed the frozen numerical gate, max absolute logit difference 0.8125 between full prefill and its cached final-token replay. One same-case numerical investigation found repeatable full logits, exact prefix-cache agreement at all layers, and unchanged cached results with explicit masks/positions. Restricted preference and first vocabulary token agreed, but full/cached relative logit L2 discrepancy was 0.03237. No full/field-only scoring comparison was accepted and generation did not run. All six statuses are retained: first blocked by validation, five not run. Thinking was explicitly disabled; no cutoff answers exist to score.

Decision: stop and report validation failure, not factual repair failure or absence of headroom. Do not relax tolerance, change backend/precision, broaden cases or develop repair. Archived inference source and attention calculation hashes remain unchanged. Details: docs/qwen-reproduction-results.md.

## E12: authorized first-case backend/precision diagnosis

Change: sequential CPU float32, MPS float32 and MPS BF16 runs on the same downloaded checkpoint and first fixed edited prompt. Compare native full prefill with native prefix-cache continuation directly, without cache manipulation; trace layer outputs and verify the existing wrapper independently. No generation or repair experiments.

Evidence: all eval/inference, token, cache-length and ownership checks pass; wrapper and native incremental logits are exactly equal in every configuration. Full/incremental max absolute differences are 8.46e-5 / 1.77e-4 / 0.8125, with relative L2 2.78e-6 / 4.55e-6 / 0.03237. Both float32 paths pass unchanged atol=rtol=2e-4; MPS BF16 still fails its original 0.02 gate. Differences start at layer 0, including a BF16 query-projection difference despite identical normalized input values. All paths agree on the first token and restricted preference, which does not establish generated utility.

Decision: recommend MPS float32/eager with CPU float32 reference and unchanged per-case checks; do not relax BF16 tolerance. Original failure and frozen code/protocol artifacts remain preserved. Stop with diagnosis; no six-case rerun, downloads, further precision configurations, repair work or selector development. Details: docs/endpoint-precision-diagnosis.md.

## E13: authorized six-case MPS float32 continuation

Change: run CPU float32 native-cache numerical references on all six fixed edited prompts, then MPS float32/eager restricted scoring and deterministic generation using preserved inference functions. Thinking disabled, original 64-token limit, no downloads or additional cases. An added cross-backend stop blocked collection at case four; preserve the partial run and original driver, remove that extra stop while retaining failed comparisons as explicit flags, and finish only the remaining three scored cases. All native full/cache gates and numerical tolerances remain unchanged; first three scored records preserved verbatim.

Evidence: 18/18 published restricted choices matched. All six CPU and MPS within-backend full/cache checks passed atol=rtol=2e-4; MPS checks repeated before generation. CPU–MPS comparisons failed that same tolerance for doc_access and oncall_route while top-token/preference agreement held. Eighteen generations ended at EOS without truncation: original full 6/6 semantically correct; edited full four clearly correct, two inconclusive (`suspended`, `customs`); field-only six explicit wrong edited-context decisions. Four satisfy clear full-correct/reuse-wrong attribution; two of those also pass cross-CPU comparison. No memory failure; peak process footprint about 10.28 GiB.

Decision: a bounded MPS cache-reuse failure target now exists, but selective-repair headroom and method efficacy remain untested. Retain all six outcomes, ambiguities, CPU flags and the collection correction. Stop after reporting; no repair development, selector, broader sweep or cloud work. Prior failures and archived numerical/source artifacts unchanged. Details: docs/qwen-float32-reproduction.md.

## E14: bounded two-intervention actual-repair oracle

Hypothesis: 16/32 additional recomputed tokens may restore full-reference decisions, possibly with better region placement than local repair.

Change: audit the exact source cache intervention before inference. Discover airline's old noncanonical padding and bank's amount split into `3  0 USD`; qualify the relation to ordinary document edits and bank factual scoring. Freeze exact inputs/settings, uniform 16-token regions plus a tail block, all singleton/disjoint-pair supports (399 combined; hard cap 400), five controls per case, primary label rubric and unchanged numerical gates. Add an isolated indexed-cache adapter using native Qwen3 layers. No delta selector or practical selection method.

Evidence: random CPU/MPS functional checks pass. Target edit-only recomputation matches the source transplant within 2e-4, all-position K/V matches exactly, full-suffix controls recover native outputs, and old-cache ownership checks pass. Every airline support outputs `refund` (0/19 and 0/171 recover); every bank support outputs `Approve` (0/20 and 0/189 recover intended label/full output). Local repair also fails both budgets. All 409 control/trial sequences end at EOS. KL varies across every support and improves on some locations, but no answer improves. Total process wall 298.36 seconds, peak footprint 10.24 GiB; copy/repair/readout/generation/checkpoint costs retained.

Decision: no answer-level headroom in this fixed region family/operator/budget range. Do not propose another mechanism, revive the rejected selector or expand the search. Bank's prior intended-label result is not a literal 8200→30 document edit. Preserve every outcome and prior source/result artifact; stop after reporting. Details: docs/qwen-headroom-results.md.
