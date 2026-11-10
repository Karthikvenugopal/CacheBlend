# Six-case Qwen3 float32 reproduction

Completed the six fixed cases. **All 18 published restricted choices matched**, but unrestricted generated utility differs: edited full recomputation produced four clear correct decisions and two inconclusive responses. Field-only reuse produced a clear wrong decision on each of those four clean-correct cases. No repair method or selective-repair headroom was tested.

All answers below are MPS float32/eager results. CPU float32 supplies a numerical reference, not a second generated-answer experiment. The checkpoint pin is `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; the authors' exact model revision remains unknown. Float32 and the Mac backend differ from the published BF16/CUDA environment. These six pairs were retrospectively selected from an existing published eight-case record, so these numbers do not establish generalization or model-wide reliability.

## Published restricted scoring

The first-token comparison reproduced every published selected-case choice: full recomputation 6/6 correct preferences, field-only reuse 0/6 correct preferences, erratum 6/6 correct preferences: **18/18 choices matched**. Original-full controls also preferred their old-context alternatives on all six. This is restricted preference reproduction under the declared environment, not proof of correct generated answers or an efficient repair mechanism.

## Ordinary generated decisions

| Case | Expected edited decision | Edited full output | Field-only output | Conservative assessment |
|---|---|---|---|---|
| retail_refund | escalate | `suspended` | `refund` | Full response names account state; inconclusive |
| airline_cancel | deny | `deny` | `refund` | Clear reuse-induced wrong decision |
| bank_withdraw | decline | `Decline` | `Approve` | Clear reuse-induced wrong decision |
| doc_access | deny | `deny` | `grant` | Clear reuse-induced wrong decision; CPU comparison flagged |
| customs_route | require | `customs` | `skip` | Full response names paperwork topic; ambiguous |
| oncall_route | page | `page` | `queue` | Clear reuse-induced wrong decision; CPU comparison flagged |

Original-full outputs were `refund`, `refund`, `Approve`, `grant`, `skip`, `queue`, respectively; all six are semantically correct for the original conditions. In retail, `refund` expresses proceeding with the requested refund, despite differing from the published restricted label `proceed`. Capitalization is not treated as an error.

Edited full recomputation: **4/6 clearly correct, 2/6 inconclusive, zero counted explicit wrong decisions**. Field-only reuse: six explicit wrong decisions under the edited conditions. Only four meet the paired attribution criterion of clear full correctness plus clear reuse incorrectness. Retail's `suspended` is not the required escalation decision. Customs' `customs` is relevant but does not explicitly say require versus skip; a permissive reading might infer the right paperwork, so it is excluded conservatively from definite error counts. Both raw responses remain available for review. This avoids treating a formatting/interpretation judgment as a demonstrated cache regression.

There were **zero truncated sequences** across 18 generations. All ended at native EOS after two or three emitted tokens, totaling 41 tokens including EOS. The two inconclusive full responses are not truncations. Any non-EOS 64-token cutoff would have been classified inconclusive rather than incorrect.

Generation settings were unchanged: native chat template with `enable_thinking=False`, empty thinking block, seed 17, eval/inference mode, deterministic algorithms, greedy unrestricted argmax, one beam, no penalties or answer masking, native EOS IDs 151645/151643, maximum 64 new tokens. No new thinking markers were emitted. Erratum generation was outside scope; its restricted-scoring outcome is retained.

## Numerical validation and CPU-reference limitations

All six CPU native full/prefix-cache checks passed `atol=rtol=2e-4`. All six MPS native-full/cached-replay checks passed the same unchanged tolerance, and were repeated successfully before generation. Their worst maximum absolute difference was 1.86920e-4. Scoring choices and restricted margins reproduced exactly between the scoring and generation stages.

| Case | MPS full/cache max absolute difference | CPU full vs MPS replay max absolute difference | CPU–MPS mixed-tolerance comparison |
|---|---:|---:|---|
| retail_refund | 1.77383e-4 | 2.53677e-4 | Pass |
| airline_cancel | 5.05447e-5 | 1.64986e-4 | Pass |
| bank_withdraw | 1.07050e-4 | 2.26974e-4 | Pass |
| doc_access | 1.86920e-4 | 6.12259e-4 | **Fail** |
| customs_route | 9.20296e-5 | 2.70844e-4 | Pass |
| oncall_route | 8.94070e-5 | 3.16620e-4 | **Fail** |

The mixed criterion is `absolute_error <= atol + rtol × |CPU reference|`; a maximum above the absolute component alone can still pass. No tolerance changed. CPU and MPS agree on initial top token and restricted preference in every case, but **numerical equivalence to CPU is not established for doc_access and oncall_route at this tolerance**. Their generated findings are MPS-specific. Even excluding both flagged cases, airline cancellation and bank withdrawal supply two clear paired generated failures on cases whose CPU comparison also passed. CPU continuation trajectories were not generated, and next-token agreement does not prove those trajectories would match.

## Collection correction and preservation

An assistant-added cross-backend assertion stopped the first scoring launch after three completed cases when doc_access failed the CPU–MPS logit comparison, despite passing the requested within-MPS full/cache check. The original partial output, failed launch, runner and protocol were preserved. The remaining three cases were completed in a separate output directory; the first three scored records were copied verbatim rather than rerun.

The correction removed that extra cross-backend numerical stop, retained its failed comparison explicitly, and kept cross-backend top-token/preference checks and all within-backend tolerance gates mandatory. It changed the collection stop rule during the run, not the tolerance or any inference function, prompt, checkpoint value, precision, score or generation setting. The experiment must not be described as a fully unchanged preregistered collection procedure or as having passed every CPU-reference check. The correction and provenance hashes are retained alongside the results.

No memory-allocation failure occurred. Largest reported macOS peak process footprint was approximately **10.28 GiB** during restricted-stage execution. This is a process measurement, not a performance or speedup claim. Stages ran sequentially; all processes have exited. No downloads, additional cases, repairs, selector changes or cloud work occurred. Original BF16 failure artifacts, precision diagnostics, archived baselines and the validated attention calculation remain unchanged.

## Decision

Restricted scoring reproduced, and four clear MPS generated-answer regressions were established, including two on cases also passing the CPU-reference logit comparison. The full model's two inconclusive answers and both cross-backend numerical flags remain visible. This establishes a bounded cache-reuse failure target; it does not establish that small selective repair can fix it or that a new method is warranted. Stop after this report. The selector remains rejected and repair development remains stopped.

Evidence: [summary](../artifacts/qwen_reproduction/float32_resume/continued/summary.json), [manual utility audit](../artifacts/qwen_reproduction/float32_resume/continued/manual-utility-audit.json), [all restricted results](../artifacts/qwen_reproduction/float32_resume/continued/restricted.json), [all 18 generated outputs and IDs](../artifacts/qwen_reproduction/float32_resume/continued/generation.json), [all CPU numerical references](../artifacts/qwen_reproduction/float32_resume/cpu_reference.json), [collection correction](../artifacts/qwen_reproduction/float32_resume/cross-backend-correction.json), [integrity verification](../artifacts/qwen_reproduction/float32_resume/continued/verification.json). The [runner](../experiments/qwen_reproduction/resume_float32.py) imports the preserved inference/scoring/generation functions; no original baseline source was edited.
