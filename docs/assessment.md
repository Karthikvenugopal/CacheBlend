# What the development outputs establish

Six cases ran on cached TinyLlama, CPU float32. All six exact-prefix continuations and next-token logit vectors matched full recomputation (observed maximum difference 0 in this demo). Separate short-prompt pretrained tests observed nonzero rounding differences up to 2.20e-5 and passed the documented tolerance. The naive experiment changed logits in every case: maximum absolute error 0.133–0.739; KL(full‖naive) 0.000646–0.003798. Greedy continuations differed in 2/6 cases. These are development observations, not estimated population rates.

| Case | Full recomputation: factual assessment | Naive relative to full |
|---|---|---|
| Refund period 30 → 14 | Gives the updated 14-day period correctly; verbose rather than number-only. | Same answer despite logit drift. |
| Shipping unchanged | Repeats the document prefix and fails to answer the question within 12 tokens. This is a baseline failure; exact-prefix reproduces it. | Says shipping is free, which is factually correct, despite disagreeing with full. Disagreement is not automatically a factual error. |
| Access code 42 → 73 | Gives updated code correctly; verbose. | Same answer despite drift. |
| Active label red → blue | Edited full answer mentions Lima correctly but continues past the limit. **Original** clean model also said Lima when red should mean Rome, a baseline reasoning failure. This is not a sound demonstration of repair success. | Mentions Lima; wording diverges at step 11. No demonstrated new factual error. |
| Museum nine → ten | Gives Monday at 10:00 AM correctly; ignores one-word instruction. | Same answer despite drift. |
| Free entry unchanged | Correctly identifies children; verbose. | Same answer despite drift. |

The JSON `answer_correct` field uses strict normalized short-answer equality. All six fail that strict format-sensitive score; it must not be interpreted as six wrong factual answers. The table above is a manual reading of the preserved raw outputs, not an automated semantic evaluator. `reuse_introduced_failure` is conditional on passing that strict baseline check and is therefore uninformative in this initial run. Raw text, token IDs and token-limit flags remain available so a better task-specific scorer can be frozen before later evaluation.

For the refund-change example, actual changed tokens are positions [26,28) in the full chat prompt, not character offsets in the document. Full recomputes [0,65); exact-prefix reuses [0,26) and recomputes [26,65); naive reuses [0,26) and stale [28,64), recomputing [26,28) and [64,65). Both answers say 14 days; naive max-logit drift is 0.220. See the generated per-case report for all ranges and token pieces.

Median measured prefill times across these six fixed-order CPU runs, including cache copying: full 223.3 ms, exact-prefix 208.6 ms, naive 190.3 ms. They are single functional observations with no uncertainty estimate. Naive processed only 2–3 prefill positions versus 57–65 for full, yet runtime did not shrink proportionally. Extra model invocations, memory movement, and small-matrix CPU behavior matter. Each generated token continues attending to the retained full cache. No GPU inference claim follows.

Decision: the foundation works. There is evidence of distribution drift and wording divergence, but **no observed new factual failure from naive reuse on a clean-correct development answer**. This small model/task combination cannot yet support claims that a selective method repairs answer errors. Preserve the negative/ambiguous evidence. Pause before implementing repair; use the next increment to test a replicated local-window mechanism and whether harder, clean-solvable development cases expose a useful failure boundary. Held-out cases have not been evaluated.
