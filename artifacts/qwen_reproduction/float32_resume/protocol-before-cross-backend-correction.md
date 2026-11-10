# Authorized float32 resumption

Resume only the same six fixed cases, checkpoint and scoring/generation definitions. Preserve all original BF16 and precision-diagnostic artifacts and frozen source. Separate outputs under `artifacts/qwen_reproduction/float32_resume/`; no overwrites or model downloads.

Execute sequentially: CPU float32 numerical reference for all six edited prompts; MPS float32/eager restricted scoring; MPS float32/eager deterministic generation only if all preceding gates complete. CPU reference evaluates full native prefill versus native prefix prefill plus one final token without cache manipulation, retaining full logits and score margins. It does not generate answers or evaluate reuse interventions on CPU.

Use unchanged `atol=rtol=2e-4`, top-token and restricted-preference agreement. Every MPS case retains the original `states()` native-prefill/cached-replay gate. In addition compare its full-reference replay logits against CPU's native full logits at the same unchanged tolerance. Unexpected disagreement stops for investigation; do not relax thresholds. Revalidate every case in the generation stage and check scoring reproducibility against the saved restricted stage.

Keep all prompts, token alignment/padding, artificial field-only intervention, seed 17, eval/inference mode, eager attention, independent branches, thinking disabled, unrestricted greedy decoding, native EOS IDs and 64-token limit exactly as prepared. Restricted stage includes full/field-only/erratum and original-full control. Generation includes original full, edited full and field-only (18 sequences at most). Six fixed pairs only; no replacements or additional sweeps.

Checkpoint all six statuses before model loading, and partial outcomes after each branch. Stop on memory-allocation failure or memory pressure preventing completion; do not retry with reduced precision, alternate backends or shorter prompts. Each stage is a separate process, so only one model is active. Record MPS allocation samples and process footprint where available; these are diagnostics, not performance claims.

After generation, preserve raw text/IDs and manually report separately: (1) agreement with the 18 published restricted choices, (2) original and edited full-recomputation answer utility, (3) explicit reuse-induced wrong generated decisions among clean-correct cases, (4) truncations/nonanswers/ambiguity. Any non-EOS cutoff at 64 tokens is inconclusive, even if a fragment looks wrong. Do not count token preference or first-token equality as generated-answer correctness. These six pairs were retrospectively selected from a published eight-case record and cannot establish generalization.

Authors' model revision remains unknown; the authorized checkpoint pin is a declared reproduction environment, not author-confirmed. Float32 differs from the paper's BF16 execution. No repair method, selector, headroom sweep or cloud compute. Stop after reporting these outcomes.
