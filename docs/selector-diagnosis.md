# Bounded selector diagnosis

**The selector remains rejected. The numerical attention update is preserved.** No implementation fault was found in the audited score/selection paths. The pilot shows highly correlated signals whose distinct token scores produce identical region rankings. This is not a mathematical equivalence of the signals. Factual efficacy remains inconclusive.

## Scope and implementation audit

Replayed only the original first-layer scores for three development document edits and eight original held-out edits. Repeated questions sharing the same document were deduplicated. No answers were generated, no new benchmark questions were added, and no parameter, selection rule or frozen source file changed. The controlled attention examples below are algebra/software tests, not added language-model evaluation cases.

- The runner calls `scores(..., "delta")` and `scores(..., "attention")` with distinct mode values synchronously; there is no deferred closure or shared score buffer. Delta reads changed attention outputs, while attention reads old probability mass on the edited span. Both intentionally use the same old context, candidate grid and sum/top-k routine. Sharing that routine makes the comparison consistent; it does not force identical choices.
- Injected late-peaked score vectors make the existing selector choose distinct nonlocal regions. An independently written Python ranking reproduces its choices. Tests use the actual 16-token region / 32-token budget geometry, plus explicit global-offset and tail checks.
- Suffix row zero maps to absolute position `edit.end`. Direct native first-layer recomputation independently verifies delta scores; complete causal softmax independently verifies attention-to-edit scores. Maximum discrepancies were 2.24e-8 and 7.45e-9 respectively. Reconstructed selections match every recorded pilot selection. Score arrays do not alias; original first-layer cache tensors remain unchanged.
- Candidate regions start at `edit.end`; a final partial region is excluded intentionally by the frozen rule. No off-by-prefix or off-by-one error was found. Six held-out questions have only two candidate regions, so their two-region choice is forced. Eighteen have 8–19 candidate regions.
- There are no exact second/third-place ties in any non-forced replay. Reversing the position preference for exact ties changes no selection. The delta cutoff gap is 22–41% of the second-place score across the non-forced original document groups, making cutoff rounding an implausible explanation.

## Rankings before selection

Across all 11 original document groups, token-level rank correlation (Spearman) is 0.972–0.999. The token rank orders differ in every group. Fitting a single proportional scale through the origin still leaves 5.4–36.0% relative L2 residual, so the difference is not just a constant scale factor or numerical noise.

**After summing each 16-token region, both entire region rank orders are identical in every group**, with Spearman correlation 1. Every region score decreases strictly with distance from the edit. Top-two selection therefore returns the local pair correctly. This holds beyond the six forced held-out questions. The inspected vectors already differ inside this pilot at token level; controlled cases are needed to demonstrate different region/position choices.

Example: the original development delivery document, showing all pre-selection region scores. Values are region sums, not normalized probabilities. No alternative configuration was evaluated.

| Region, absolute half-open token range | Sum of old attention-to-edit mass | Sum of delta norms | Rank in both |
|---|---:|---:|---:|
| [28, 44) | 0.55747628 | 0.16776441 | 1 |
| [44, 60) | 0.33412305 | 0.08715360 | 2 |
| [60, 76) | 0.23494804 | 0.05432071 | 3 |
| [76, 92) | 0.18601850 | 0.04282607 | 4 |
| [92, 108) | 0.15133522 | 0.03400454 | 5 |
| [108, 124) | 0.12411604 | 0.02700865 | 6 |
| [124, 140) | 0.10800405 | 0.02251171 | 7 |
| [140, 156) | 0.09308421 | 0.01886759 | 8 |

Changing only the number of whole regions would not distinguish these inspected rankings. The diagnostic does not establish that a different region size or aggregation would improve repair. Those changes were not tried.

## Why the signals are not generally equivalent

For one edited token and one unchanged query/head, let `a` be its old attention probability, `r = exp(q·Δk/√d)`, `v` its old value, `Δv` its value change and `O` the old head output. With `D = 1 + a(r−1)`, elementary softmax algebra gives:

```text
ΔO = a / D × [(r−1)(v−O) + rΔv]
```

Keeping suffix queries fixed across versions does not make them identical across positions or heads. It also does not keep edited keys fixed. Therefore old mass `a` alone does not determine the output change.

There is a useful special case: if the edited key does not change (`r=1`), then `ΔO=aΔv`. For a single head and one value-only edit, norm scores are proportional to old attention and give the same ranking. Aligned head profiles preserve that result. Our actual multi-head norm, however, aggregates `sqrt(Σ_h a_h² ||Δv_h||²)`, while the old-attention baseline averages `a_h`. Different head profiles can reverse the ranking even with unchanged keys. Multiple edited values can also reinforce or cancel. Those extra equivalence conditions are not assumptions of the pilot.

## Controlled causal-attention examples

The unchanged original score and selector functions were exercised with explicit Q/K/V tensors. Edited projections alone were supplied by the test harness. An independent scalar float64 causal-softmax implementation checks the expected outputs. Four query heads are used; no language-model answers or learned factual ability enter these tests. Maximum score discrepancy is 3.24e-7. Position 0 is edited; listed positions are eligible unchanged queries.

| Example | Old attention-to-edit scores | Delta norm scores | Old attention chooses | Delta chooses |
|---|---|---|---:|---:|
| Value-only, aligned heads | 0.500, 0.333, 0.250 | 1.000, 0.667, 0.500 | 1 | 1 |
| Key and value change | 0.500, 0.333, 0.250 | 1.000, 0.667, 1.600 | 1 | 3 |
| Value-only, different head profiles | 0.600, 0.520 | 1.200, 1.402 | 1 | 2 |

In the second example, the edited key raises only the last query’s edited-token logit by `log(12)`. Its new probability is `12/(12+3)=0.8`, reversing the choice despite lower old attention. In the third, the first position has head probabilities `[.6,.6,.6,.6]`; the second has `[.99,.99,.05,.05]`. The mean prefers the first, while the norm of value changes prefers the second. These are valid causal-attention counterexamples with fixed suffix queries across versions.

The tiny algebra examples use one-position selection so the reversal is transparent. Separate injected-score controls test the frozen 16/32 region geometry and prove the production selector can choose distant regions when its input warrants it. No controlled example was scored for answer utility.

## What failed and what remains unresolved

| Component | Diagnosis |
|---|---|
| Numerical attention calculation | Passed; preserved unchanged. |
| Implementation | No coupling, aliasing, indexing or tie-breaking fault found in audited paths. Independent references and separating inputs pass. This is scoped evidence, not a proof about all possible code paths. |
| Mathematical distinctness | Signals are generally distinct; pilot token scores already differ. Treating them as universally equivalent would be wrong. |
| Tested selection hypothesis | Extra first-layer information did not change any region ordering in the pilot. The proposed signal plus region-sum decision rule delivered the same local policy. The expectation of a different useful allocation was unsupported here. |
| Experimental design | Coarse regions suppress the observed token-ranking differences; six held-out choices are forced. TinyLlama’s clean derived answers failed under the frozen protocol, so factual benefit cannot be established. |

**No replacement mechanism is recommended from this diagnosis.** It identifies redundancy of this tested allocation, not which alternative would repair useful downstream dependencies. Smaller regions, additional layers or a new signal would require separate evidence and prior-work checks. Keep the selector rejected and the validated attention calculation available. Factual efficacy stays inconclusive; no speedup or novelty claim follows.

Reproduce this bounded audit with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  .venv/bin/python experiments/feasibility/diagnose_selection.py
```

See [raw rankings and counterexamples](../artifacts/feasibility/diagnostic/diagnosis.json), [summary checks](../artifacts/feasibility/diagnostic/summary.json), and [the diagnostic source](../experiments/feasibility/diagnose_selection.py). Frozen inference source and original development/held-out results were hash-checked before and after replay and remain unchanged.
