# Bounded factual-repair oracle diagnostic

Decision: **stop at qualification**. No full-correct/reuse-wrong case was established in 12 fixed, separate development candidates. This screened TinyLlama/task setup is unsuitable for evaluating factual repair. Factual repair efficacy remains inconclusive; there is no evidence here for another selector or mechanism.

All 12 candidates were supported and retained. Full recomputation answered 4/12 correctly; unrepaired suffix reuse preserved all four correct answers. Four other full outputs were incorrect and four were nonanswers. Original full recomputation also answered 4/12 correctly. Reuse corrected one full-model error, which is the opposite of the required qualification direction. These are descriptive screening outcomes, not generalization estimates.

| Candidate | Edited full | Unrepaired suffix | Interpretation |
|---|---|---|---|
| slot-forward | 8, correct | 8, correct | Direct value edit |
| slot-reverse | 3, correct | 3, correct | Direct value edit |
| gate-forward | left, wrong | left, wrong | Gate 9 requires right |
| gate-reverse | left, correct | left, correct | Original full was wrong for gate 9 |
| access-forward | yes, wrong | yes, wrong | Active D requires no |
| access-reverse | yes, correct | yes, correct | Original full was wrong for D |
| pin-forward | expression, no result | expression, no result | Requested result 6 never supplied |
| pin-reverse | expression, no result | expression, no result | Requested result 3 never supplied |
| switch-forward | insufficient-information claim | same nonanswer | Record actually determines shut |
| switch-reverse | insufficient-information claim | same nonanswer | Record actually determines open |
| route-forward | lake, wrong | R, wrong response type | Requested destination hill |
| route-reverse | R, wrong response type | lake, correct | Reuse improves this instance |

The access answers hit the decoding limit but already assert their answer; their yes/no judgments do not depend on treating truncation as factual error. The arithmetic answers do not compute the requested result, even where they name operands. Manual review checked all outputs, including nonmatches and ambiguous phrase matches. No formatting-only discrepancy was promoted into a reuse-induced factual regression.

The fixed protocol used six short templates in both edit directions, independently retokenized equal-length edits, and complete unchanged suffixes of 4–11 tokens. These deliberately short records make exhaustive one/two-token search feasible. They do not cover long-distance documents or establish that selective repair lacks headroom generally. No questions from the previous held-out pilot were used or tuned.

“Unrepaired suffix” here refreshes the actual edited span at every layer using the existing repair operator and leaves all downstream document KV states stale. Every method processes the identical question freshly. It is distinct from the original deliberately naive complete-prompt baseline; that baseline and its code remain unchanged. Each run owns its cache independently.

The preregistered next gate would exhaust every one/two-token subset of the complete suffix for at most the first two qualifying cases, with the edited span mandatory and local repair at matched budgets. Answer labels would score all generated choices afterward. **No case passed the prerequisite, so no oracle repair choices were evaluated.** Consequently small-repair recovery, advantage over local repair, and useful repair positions are all unestablished. No repair-position explanation can be inferred from this screen.

Cached TinyLlama 1.1B chat ran locally on CPU in float32/eager mode, seed 17, four threads, eval/inference mode, 24 greedy output tokens for all screen arms. Screening took 49.72 seconds after model loading, including original/full/reuse inference, decoding and checkpoint writes. This is a wall-clock screening duration, not a repair-speed comparison. No cloud work, downloads, or GPU performance extrapolations occurred.

All frozen inference-source hashes, including the validated attention calculation and original baselines, still match. The selector remains rejected. Stop method development in this setup: qualification failed, rather than an oracle disproving small-repair headroom. A different reliable factual evaluation setup would have to establish the prerequisite before further repair-method development is justified. No replacement selector, speedup, novelty or generalization claim follows.

Artifacts: [fixed protocol](../experiments/oracle_bound/protocol.md), [12 candidates](../experiments/oracle_bound/candidates.json), [runner](../experiments/oracle_bound/run.py), [all raw outcomes and actual token positions](../artifacts/oracle_bound/screen.json), [manual audit](../artifacts/oracle_bound/screen-audit.json), [stop status](../artifacts/oracle_bound/oracle-status.json), [integrity verification](../artifacts/oracle_bound/verification.json). The raw outcomes retain answers, generated token IDs, logit differences and work counts; utility judgments above are separate from logit measurements.
