# Gate 1: first-layer attention-output update

Passed on six cached-TinyLlama CPU float32 cases, with three repetitions each. The numerical gate was fixed at `atol=rtol=2e-4`. No direct-row fallback was required. The calculation applies only to unchanged suffix queries at the first layer; it does not repair deeper KV states.

| Case | Tokens | Max absolute error | Relative L2 error | Retained MiB | Metadata postprocess ms | Update ms | Direct whole first-attention ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-edit | 167 | 0 | 0 | 2.650 | 2.816 | 0.223 | 6.740 |
| early-edit | 167 | 5.96e-08 | 1.99e-07 | 2.650 | 3.512 | 4.337 | 6.146 |
| middle-numeric | 341 | 5.96e-08 | 2.57e-07 | 5.411 | 8.042 | 3.378 | 10.747 |
| late-edit | 174 | 2.98e-08 | 2.16e-07 | 2.761 | 2.491 | 0.869 | 5.286 |
| token-boundary | 169 | 4.84e-08 | 1.94e-07 | 2.682 | 2.271 | 1.809 | 5.396 |
| semantic-reversal | 334 | 4.47e-08 | 2.54e-07 | 5.300 | 7.210 | 1.864 | 10.149 |

All timings are medians of three component measurements. Direct first-attention includes embedding normalization, QKV/RoPE and output projection over the complete prompt, but no MLP. The update covers only the unchanged suffix and includes edited K/V projection. No-edit copies the original outputs. These component comparisons do not establish end-to-end speedup.

Assumptions: native dense full-attention Llama, evaluation/inference mode, no padding, batch one, fixed model weights, fixed positions and equal token lengths after complete retokenization. The changed envelope is excluded from the fixed-query formula because its queries may change. Original metadata must match the clean token sequence and engine. Insertions/deletions and accumulated mixed-cache edits are unsupported.

The first layer uses per-token normalized embeddings for Q/K/V. For unchanged suffix tokens Q stays fixed. Replace the edited columns in the softmax numerator and denominator while retaining all other contributions. Rebase exponentials against old log-normalizers/new scores, evaluate subtraction in float64, and fall back to direct attention when the unedited old mass is below 1e-4 or the update is non-finite. Forced-fallback tests exercise that branch; zero observed fallbacks is not a guarantee for other prompts.

Retained metadata consists of first-layer Q and pre-output-projection head output O in float32 plus per-head log-normalizers in float64. For TinyLlama this is `(2×2048×4 + 32×8) = 16,640 bytes/token`, or 16.25 MiB at 1,024 tokens, excluding the existing KV cache and temporary buffers. No full attention matrix is retained. Peak process memory was not measured.

Native eager attention does not expose its normalization statistics. Collection therefore includes hooks/copies during the complete original prefill and an extra chunked QK pass plus logsumexp afterward. This is an actual implementation cost, not assumed free metadata. The extra postprocessing is reported above; hook-copy time, total collection including original prefill, uninstrumented prefill and signed paired incremental estimates are preserved in [the raw report](../artifacts/feasibility/gate1.json). Negative incremental differences reflect measurement noise/cache warming and must not be interpreted as negative metadata cost.

The first Gate 1 run passed before additional timing instrumentation; it remains in `gate1-v1.json`. The final run revalidated the unchanged calculation with component timing. Gate 2 was not started until passing Gate 1. The isolated sparse repair adapter additionally has a full-position/native endpoint gate and ownership tests. Existing engine source is untouched.
