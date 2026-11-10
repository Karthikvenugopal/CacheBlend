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
