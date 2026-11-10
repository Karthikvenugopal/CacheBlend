# Six-pair Qwen3-1.7B reproduction: stopped at numerical validation

Later authorized result: the [float32 continuation](qwen-float32-reproduction.md) completed all six cases, preserving this BF16 failure. Restricted preferences matched 18/18; generated full answers were clear on four cases and inconclusive on two, with four paired MPS reuse-induced errors. CPU–MPS numerical flags and a collection-stop correction are documented in that separate report.

Follow-up: the separately authorized [first-case precision diagnosis](endpoint-precision-diagnosis.md) reproduces this failure through the native MPS/BF16 cache API. CPU and MPS float32 pass the unchanged float32 tolerance. The six-case reproduction below remains stopped; its original outcomes were not overwritten.

The authorized attempt **did not establish restricted-scoring reproduction or generated-answer failures**. It stopped on the first case's native-versus-cached numerical gate. Generation was not run. Preserve this as a compatibility/validation failure, not a negative factual-repair finding or a successful repair method. No selector, repair, additional factual sweep or cloud job followed.

## Checkpoint and thinking settings

Downloaded exactly **4,079,434,577 bytes** into ignored `.models/qwen3-1.7b`, pinned at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Sizes and SHA-256 checks passed. The authors' model revision remains **unknown**; this explicit official pin is not author-confirmed. Source/result revision remains `e9085eafcc6c83e60c548de69060c4bd5b210c96`.

The downloaded tokenizer reproduced all 18 frozen old/new/erratum chat renderings exactly. Each uses `enable_thinking=False` and the empty assistant thinking block `<think>\n\n</think>\n\n`. The checkpoint defaults to sampling, but the prepared generation routine explicitly uses unrestricted argmax, no sampling or penalties, seed 17, and EOS IDs 151645/151643. Its limit remains 64 new tokens. An answer cut off there would be inconclusive, not incorrect. No generation occurred, so there are no observed truncations or generated answers to classify.

## Retained case outcomes

| Fixed pair | Restricted stage | Generation | Generated-answer utility |
|---|---|---|---|
| retail_refund | Numerical gate failed before pair scoring | Not run | Inconclusive |
| airline_cancel | Not run after gate failure | Not run | Inconclusive |
| bank_withdraw | Not run after gate failure | Not run | Inconclusive |
| doc_access | Not run after gate failure | Not run | Inconclusive |
| customs_route | Not run after gate failure | Not run | Inconclusive |
| oncall_route | Not run after gate failure | Not run | Inconclusive |

Do not report agreement as 0/18 or factual failures as 0/6: those comparisons were not evaluated. The first-case numerical investigation contains two restricted-reference margins, but not an accepted full/field-only reproduction comparison. Its matching first vocabulary token is not a generated decision.

## Compatibility checks and failure

The first launch stopped before any case because Transformers 5.2 left the checkpoint's duplicate embedding/output parameters as separate objects despite `tie_word_embeddings=True`. Streaming SHA-256 of both 622,329,856-byte BF16 tensor payloads matched exactly. A compatibility amendment verifies loaded values with `torch.equal` and restores their declared shared-storage alias before MPS transfer. This changes no weight value, prompt, precision or score. Original runner, prepared hashes and failed-launch log remain retained. Unique live parameters then match the prepared count, 1,720,574,976.

The subsequent restricted command used the approved MPS/BF16/eager path, PyTorch 2.8.0, Transformers 5.2.0, evaluation/inference mode and deterministic algorithms. For the 331-token retail prompt, the edit occupies token positions `[16,18)` with no alignment padding. A native edited prefill versus replaying its final token from its own cache failed the frozen `atol=rtol=0.02` check, with **maximum absolute logit discrepancy 0.8125**.

One bounded investigation used only this same first case at unchanged settings; no answers were generated:

| Numerical control | Max absolute error | Relative L2 | Observation |
|---|---:|---:|---|
| Repeated full prefill | 0 | 0 | Repeatable |
| Full prefill versus its cached final-token replay | 0.8125 | 0.03237 | Frozen numerical gate fails |
| Full prefill versus prefix-prefill then final token | 0.8125 | 0.03237 | Same discrepancy with independently constructed prefix cache |
| Cached replay versus explicit mask/positions and attention output | 0 | 0 | These controls do not change the cached result |

Full-prefill prefix K/V versus independent prefix-prefill K/V matched exactly at all 28 layers. Both full and cached readouts had the same first vocabulary token; the correct-versus-stale restricted margin changed from 3.0 to 2.75 without changing that preference. This is evidence of sensitivity to processing the final query in the full versus incremental path in this MPS/BF16 environment. It does **not** establish that BF16 alone is responsible, that all MPS inference is invalid, or that the cache intervention causes a factual error. The precise numerical cause remains unresolved; no precision/backend sweep or post-hoc tolerance relaxation was performed.

The failed restricted launch took 16.34 seconds including loading/hashing and reported 4,857,382,544 bytes (4.52 GiB) peak memory footprint through macOS `/usr/bin/time -l`. This is a failed-run process measurement, not a complete unified-memory bound, inference benchmark or speedup claim.

## Decision and evidence

Stop here as requested. This attempt cannot establish reliable clean generated decisions, reuse-induced generated errors or selective-repair headroom. Method development remains shelved; the archived baseline and validated attention calculation are unchanged. No new settings, cases, repairs or selectors are proposed.

Retained evidence: [authorized scope](../artifacts/qwen_reproduction/authorization.json), [model manifest](../artifacts/qwen_reproduction/model-manifest.json), [thinking-mode audit](../artifacts/qwen_reproduction/thinking-mode-audit.json), [all six stage statuses](../artifacts/qwen_reproduction/execution-status.json), [raw restricted checkpoint](../artifacts/qwen_reproduction/restricted.json), [generation not-run status](../artifacts/qwen_reproduction/generation-status.json), [embedding check](../artifacts/qwen_reproduction/embedding-compatibility.json), [numerical investigation](../artifacts/qwen_reproduction/endpoint-diagnosis.json), [verification](../artifacts/qwen_reproduction/verification.json). Original and amended preparation files and all console logs remain alongside these artifacts.
