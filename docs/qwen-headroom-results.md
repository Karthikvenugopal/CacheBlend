# Bounded actual-repair headroom: negative at the tested budgets

**No answer-level headroom was found in the frozen search family.** All 399 partial repairs returned the old answer. Local repair also failed at both budgets. Actual full-suffix recomputation recovered the full-reference output in both cases, and implementation/numerical gates passed. Stop here: no new repair mechanism or selector is proposed, and no larger search was run.

## What was actually edited

The pre-run [intervention audit](../experiments/qwen_headroom/intervention-map.md) found that these published cache interventions differ from our original literal-document contract. Both independently rendered chats are tokenized and then padded inside the differing token envelope to preserve positions. The entire prompt—including policy, request and assistant preamble—is cached. Only the edited-span K/V is refreshed initially; the final prompt token is replayed freshly. The question is not appended anew.

- **Airline:** aligned length 324, edit [16,19), old field `fare_type: flexible  ` with two inserted space tokens, new field `fare_type: basic_economy`. Reuse K/V [19,323), replay token 323. The padded old sequence does not round-trip through the standard tokenizer. The edited policy clearly requires `deny`.
- **Bank:** aligned length 331, edit [16,19), old field `account_balance: 8200 USD`, actual edited field **`account_balance: 3  0 USD`**. The shared suffix starts with the final `0`, so two padding tokens split the intended `30`. Reuse K/V [19,330), replay token 330. This is not the intended ordinary 8200→30 edit. Interpret `Decline` recovery as full-output/intended-label recovery, not an unqualified literal monetary-fact result.

The original validated adapter retokenizes literal complete prompts, rejects unequal lengths and does not insert padding. Both raw pairs violate its equal-length contract. Its naive dependency pattern resembles this intervention only after these aligned IDs are stipulated. The earlier feasibility pilot's fresh-question protocol is different again. Neither setting is silently treated as equivalent here.

The source unrepaired intervention takes edited-field rows from a full clean prefill. The new adapter instead recomputes selected rows using native Qwen3 layers, including Q/K normalization and absolute RoPE, with full causal access to mixed fresh/stale K/V. It receives only old cache, new IDs and positions; no clean downstream rows or labels enter repair. Actual edit-only recomputation reproduced the stale baseline before any search.

## Frozen search and generated-answer results

MPS float32/eager, existing pinned checkpoint, eval/inference mode, seed 17, four threads, deterministic algorithms, thinking disabled, unrestricted greedy argmax, EOS 151645/151643, unchanged 64-token limit and `atol=rtol=2e-4`. Original authors' checkpoint revision is unknown. Prompts, IDs, scoring and numerical gates were frozen before target repair inference.

Mandatory edit: three tokens [16,19). Additional budgets: **16 or 32 tokens**, about 5% or 10% of the prompt. Use every singleton and every disjoint pair of 16-token regions on the frozen suffix grid. Add one end-aligned region for bank's short tail; the fresh final readout token is always outside the budget. Local repair uses [19,35) or [19,51). No score or answer chooses regions. Apply frozen answer labels only after the fixed attempts finish.

| Case | Full reference | Unrepaired / actual edit-only | Local +16 / +32 | Oracle +16 | Oracle +32 |
|---|---|---|---|---|---|
| Airline | `deny` | `refund` / `refund` | `refund` / `refund` | **0/19** recover | **0/171** recover |
| Bank, qualified | `Decline` | `Approve` / `Approve` | `Approve` / `Approve` | **0/20** recover | **0/189** recover |

Every airline candidate generated exactly `refund`; every bank candidate exactly `Approve`. There were no truncated, ambiguous or alternate-word candidate responses to rescue by semantic grading. All 399 region choices and ten baseline/validation controls completed: **409 sequences ended at EOS**, with 1,032 emitted tokens including EOS. No repair choice surpassed local repair in answer utility at a matched budget. Full-suffix controls recomputed 307/314 rows respectively, plus the common fresh readout, and recovered the full output.

This is exhaustive over this region family, two budgets and one all-layer recomputation operator. It does not rule out other region boundaries, larger budgets, arbitrary token subsets, layer schedules, different operators, or well-formed document tasks. It is an oracle diagnostic, not a deployable selector or a general impossibility result.

## Distribution changes and dependencies

The repairs were not no-ops: airline produced 190 distinct next-token KL values and bank 209, despite unchanged answers. Native full/cache and retained CPU-reference checks passed. Edit-only recomputation versus source field transplantation differed by at most 1.62e-5/2.53e-5 in logits. All-position recomputation matched native K/V exactly in both cases; all-position and full-suffix readouts matched the full-reference readout exactly. Original caches remained unchanged after every case's controls and entire search.

Some locations improved distribution fidelity without restoring the decision:

| Case / budget | Unrepaired KL | Local KL | Best observed KL | Best-KL regions |
|---|---:|---:|---:|---|
| Airline +16 | 23.2683 | 21.9014 | 21.6580 | [35,51) |
| Airline +32 | 23.2683 | 19.4248 | 12.4717 | [35,51) + [51,67) |
| Bank +16 | 8.5920 | 8.9687 | 8.0395 | [307,323) |
| Bank +32 | 8.5920 | 8.7125 | 7.9756 | [211,227) + [307,323) |

KL is full-reference next-token distribution to repaired distribution, in nats. These best locations are identified after enumeration for explanation only, not proposed selection rules.

Airline's best pair spans the fare-policy body, including the non-refundable condition and `deny` token. This is consistent with updating the conditional rule's contextual representation, but stale predecessors and downstream cached request states remain. It does not establish which dependency prevents recovery. Bank's best region contains the 500-USD request and decision instruction; its best pair additionally touches neutral filler. That weak distribution effect is insufficient to assign a meaningful causal role to the filler or motivate a mechanism. There are **no answer-useful positions** to explain within this family. Better KL alone is not a useful generated-answer repair result.

## Execution costs

Total process wall time was **298.36 seconds**; the runner measured 294.53 seconds including loading, setup, gates, all attempts and checkpoints. Model loading plus checkpoint verification took 12.95 seconds. Metadata/alignment/enumeration took 105.12/6.26 ms; original old-cache prefills took 1,076.73/714.50 ms for airline/bank. No attention-output or selector metadata was collected.

Medians across the fixed oracle attempts, milliseconds:

| Case / budget | Cache copy | Sparse forward | Diagnostic readout/copy | Generation/readout/copy | Trial wall incl. first checkpoint |
|---|---:|---:|---:|---:|---:|
| Airline +16 | 8.20 | 146.42 | 145.58 | 281.60 | 589.21 |
| Airline +32 | 8.30 | 157.46 | 143.78 | 280.81 | 597.88 |
| Bank +16 | 8.83 | 151.04 | 148.57 | 430.83 | 757.41 |
| Bank +32 | 8.37 | 156.79 | 141.35 | 411.42 | 725.17 |

Generic metric computation added roughly 2.2–2.3 ms per trial; first checkpoint writes roughly 0.8–1.8 ms. Components have independent medians and need not sum exactly. Both the diagnostic and generation replay the final token, so both readouts/copies are charged. Final report/checkpoint overhead is included in total run wall time. Each partial repair recomputes 19/35 rows (532/980 token-layers), plus separately recorded readout/decode work; counts do not imply proportional runtime savings.

Full-reference prefills took 668.70/654.40 ms, followed by 592.79/660.78 ms of diagnostic readout, generation, metrics and checkpointing. Source unrepaired-control evaluation took 447.11/572.96 ms **and depends on that clean prefill** to obtain the transplanted field. Actual edit-only controls took 1,569.41/902.77 ms in single cold measurements. Those controls, warm-up effects and diagnostic duplication preclude a practical speedup comparison; all raw component times are retained. No latency or speedup benefit is claimed.

Largest macOS process footprint was approximately **10.24 GiB**. No memory failure occurred. No downloads, cloud compute, additional cases or post-result repair attempts were run. The search ended at 399 supports, below the hard ceiling of 400.

## Decision and retained evidence

The small-budget answer-recovery hypothesis is unsupported under this operator and frozen family. Distribution improvements do not justify a selector or another repair mechanism. Keep the rejected delta selector rejected and stop method development on these results; do not expand this search. The input audit also narrows how the earlier bank decision-pair result may be interpreted. The original baseline source, validated attention calculation, BF16 failure and all previous results remain intact.

Evidence: [pre-run intervention map](../experiments/qwen_headroom/intervention-map.md), [frozen protocol](../experiments/qwen_headroom/protocol.md), [freeze hashes](../experiments/qwen_headroom/frozen.json), [exact IDs/supports](../experiments/qwen_headroom/fixed_cases.json), [all airline attempts](../artifacts/qwen_headroom/airline_cancel/outcomes.json), [all bank attempts](../artifacts/qwen_headroom/bank_withdraw/outcomes.json), [post-run assessment and successful-support lists](../artifacts/qwen_headroom/assessment.json), [manual output audit](../artifacts/qwen_headroom/manual-output-audit.json), [verification](../artifacts/qwen_headroom/verification.json), [native Qwen3 operator](../experiments/qwen_headroom/sparse_qwen.py). No target inference remains running.
