# Native-cache precision diagnosis: first fixed case only

**Diagnosis:** the large discrepancy reproduces in the model's native MPS/BF16 execution without our cache manipulation. CPU float32 and MPS float32 pass the existing float32 tolerance. Our wrapper's logits exactly equal the native incremental logits in all three configurations. The evidence identifies precision-dependent, sequence-shape-sensitive numerical behavior in the native MPS path; it does not identify a particular Metal kernel defect or establish that other BF16 backends behave the same way.

**Recommended configuration:** MPS float32 with native eager Qwen3 attention, using CPU float32 as the numerical reference. Keep `atol=rtol=2e-4`, the already existing float32 tolerance; do not increase the failed BF16 tolerance. This recommendation is supported on this single numerical case, not yet on the other five cases or generated answers. No six-case reproduction was resumed.

## Fixed input and execution checks

Use only the downloaded checkpoint at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and the edited `retail_refund` prompt. The authors' model revision remains unknown. Each configuration ran in a separate process, sequentially: CPU float32, MPS float32, then MPS bfloat16. All used eager attention, seed 17, four CPU threads and deterministic algorithms. No autocast context, precision flags, fallback option or weight quantization was introduced. Float32 conversion changes arithmetic precision; it cannot recover precision absent from the stored BF16 checkpoint.

All checks passed:

- Every model module remained in evaluation mode; every observed forward hook ran under inference mode.
- The complete prompt contains 331 tokens. All configurations used identical IDs, SHA-256 `11e51bb7e1b56a6d4768ba3f60492dc6a137417bfd021a5facfaf5f13aa80457`. The first case requires no alignment padding.
- Native prefix prefill consumes exactly the first 330 tokens. Its returned cache goes directly back to the model, with just the remaining token, and grows from 330 to 331. The concatenated inputs exactly equal the full input. There is no duplicated final token.
- The primary continuation supplies no custom cache construction, cloning, splicing, position IDs or attention mask: the model derives positions from its native cache length.
- Native full and prefix caches have independent K/V storage. Full-cache SHA-256 remains unchanged after both the native continuation and the secondary wrapper check. The wrapper also owns independent storage. Native in-place growth is confined to its own prefix branch.

The checkpoint's byte-identical duplicate embedding/output entries retain the previously documented shared-storage compatibility fix. No weight values or archived inference sources changed.

## Logit results

Full prefill versus native prefix-prefill plus one token:

| Configuration | Maximum absolute difference | Relative L2 | Normalized maximum difference | Top-token agreement | Correct−stale margin: full → incremental | Existing gate |
|---|---:|---:|---:|---|---|---|
| CPU float32 | 8.46386e-5 | 2.78395e-6 | 2.06668e-6 | Yes | 3.107191 → 3.107254 | Pass, 2e-4 |
| MPS float32 | 1.77383e-4 | 4.55020e-6 | 4.33130e-6 | Yes | 3.107191 → 3.107260 | Pass, 2e-4 |
| MPS bfloat16 | 0.8125 | 0.0323725 | 0.0200617 | Yes | 3.000000 → 2.750000 | Fail, 0.02 |

Relative L2 is `||difference||₂ / ||full||₂`. Normalized maximum is `max|difference| / max|full|`. The literal maximum elementwise relative errors, `max(|difference| / max(|full|,1e-6))`, are **0.231915, 1.369418 and 7783.399902**, respectively. Those maxima are sensitive to reference logits near zero; they are retained explicitly rather than substituted for the mixed absolute/relative gate. All vocabulary logits participate in the actual `allclose` checks.

The unrestricted top token is ID 82, decoded as `s`, in both paths for every configuration. This is one token, not a generated answer or evidence that the policy decision is correct. The restricted margin compares first-token IDs 81082 (`escalate` alternative) and 776 (`proceed` alternative); every margin remains positive. Restricted preference alone does not establish generated-answer utility.

MPS float32 also agrees with CPU float32 under the unchanged mixed tolerance: maximum absolute difference is **2.13623e-4** for full prefill and **1.90735e-4** for incremental execution. Full-prefill relative L2 is 6.05420e-6. The full comparison's maximum slightly exceeds the absolute tolerance alone, but passes the existing `atol + rtol × |reference|` condition. Its restricted full margin and top token agree with CPU. These cross-backend results support float32 as the local configuration without relaxing a threshold.

## Where divergence begins

The first nonidentical block output is layer **0** (zero-based) in all configurations; float32 differences are small. In MPS/BF16:

- Embedding and last-token input-normalization values match exactly.
- Layer-0 query projection differs by up to **0.00390625** despite identical normalized last-token inputs; the full-sequence and one-token calls present different operation shapes. Query-normalization output differs by 0.001953125. Value projection differs by 1.52588e-5; key projection matches.
- Layer-0 attention output happens to match exactly. Its MLP output differs by 0.0001220703, and the block output differs by **0.00048828125**. The differences then propagate through later blocks to the 0.8125 logit discrepancy.

The original failure is therefore not explained by cache aliasing, a duplicated token or explicit positions/masks. Both native and wrapped paths produce exactly the same incremental result. Switching the arithmetic to float32 on the same MPS backend removes the large discrepancy under the existing gate. We did not run CPU BF16, FP16, alternate kernels, additional prompts, or a precision sweep beyond the three requested configurations; the precise low-level accumulation/rounding implementation is not isolated.

PyTorch documents that whole-tensor and sliced/batched computations can differ numerically even with mathematically equivalent inputs, and that CPU/GPU bitwise equality is not guaranteed. That explains why bitwise equality is too strict generally; it does **not** excuse accepting this BF16 failure. [PyTorch 2.8 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html#batched-computations-or-slice-computations).

## Configuration recommendation and limits

Use CPU float32 as the reference and MPS float32/eager as the supported local candidate. Retain `atol=rtol=2e-4`, plus top-token and restricted-margin checks, on each later case before interpreting reuse effects. This is the original float32 tolerance, not a fit to make BF16 pass. The measured within-MPS float32 maximum is below its absolute component, and cross-CPU comparisons pass the same existing mixed criterion. Near-tie decisions still require investigation rather than a tolerance change.

The MPS float32 diagnostic's macOS peak process footprint was **10.04 GiB** on the 16-GiB machine. That includes diagnostic hooks, loading and copying; it is not a guarantee of peak memory for other workloads. Sequential execution avoids holding multiple models at once. The CPU worker completed successfully; its outer `/usr/bin/time -l` wrapper returned 1 because the sandbox denied `sysctl kern.clockrate`. CPU result artifacts were complete, and no inference rerun was needed for that profiling restriction.

All original failure evidence and frozen source hashes remain unchanged. No downloads, generated answers, six-case rerun, repair experiments, selector work or cloud compute occurred. Stop with this diagnosis and configuration recommendation; factual efficacy and selective-repair headroom remain unestablished.

Reproducible diagnostic source: [diagnose.py](../experiments/qwen_reproduction/endpoint_precision/diagnose.py). Raw results, layer traces, small saved diagnostic tensors, console logs and integrity checks: [summary](../artifacts/qwen_reproduction/endpoint_precision/summary.json), [CPU float32](../artifacts/qwen_reproduction/endpoint_precision/cpu-float32/result.json), [MPS float32](../artifacts/qwen_reproduction/endpoint_precision/mps-float32/result.json), [MPS bfloat16](../artifacts/qwen_reproduction/endpoint_precision/mps-bfloat16/result.json).
