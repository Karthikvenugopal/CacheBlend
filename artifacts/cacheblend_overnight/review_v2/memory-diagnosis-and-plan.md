# Failed preflight: memory diagnosis and an unexecuted plan

No inference, model loading, allocation experiment or power-setting change was performed for this review. The checkpoint was inspected only through small safetensors headers. The loader/executor and all failed-run files remain unchanged. This is a proposal, not a new run or a demonstrated memory improvement.

The failure has two relevant stages, not just a large cache-validation peak:

| Time from supervisor start | Observation |
|---|---|
| 0s | Kernel pressure1, OS free statistic80%; existing swap used2.8GiB approximately. Historical swap alone was not a failure. |
| 10.0s | Critical pressure4; new swapouts1.23GiB. The model loader had not returned, so validation-cache overlap cannot explain this early event. |
| 18.3s | First post-load sample: MPS tensors7.57GiB, driver8.33GiB. |
| 23.3s | After independent cache composition: tensors7.85GiB, driver9.82GiB. |
| 30.1–60.3s | Warning pressure2 at each sample, with heavy rolling swap traffic. At60.3s the frozen sustained-pressure rule requested a stop. |
| 52.5s | After the custom-full endpoint check: tensors9.07GiB, driver10.70GiB, recommended MPS working set11.84GiB. |
| 60.7s / 70.3s | Worker checkpointed the stop / supervisor cleanup finished. No generated method result completed. |

The custom-full check passed all57 comparisons (logits plus28 K/V pairs), with maximum absolute difference zero. The next reference ratio1 endpoint was interrupted. The worker's `ACTIVE.phase` still said composition; it was stale. The traceback and completed-gate files locate the actual stop in validation. Five native deep-prefix tensor flags remain the previously documented diagnostics, without a changed numerical tolerance.

New system-wide swapouts totaled10.28GiB and swapins7.74GiB over60.3s. This is cumulative paging traffic, not10.28GiB of simultaneously resident data. Other processes and existing compressed/swapped memory also contribute. Process RSS high-water5.75GiB and MPS tensor/driver values overlap; summing them would be wrong. The MPS samples are sparse, not guaranteed peaks, and the driver-minus-tensor difference is not evidence by itself of a leak. No recorded OOM or nonfinite output occurred. No defensible complete five-method runtime estimate is available.

## Static payload accounting

From the exact cached checkpoint headers and model config:

| Payload, float32 | Size |
|---|---:|
| All stored parameters, preserving both embedding and LM head | 8,126,959,616 bytes =7.569GiB |
| Each of embedding and LM head | 1,244,659,712 bytes =1.159GiB |
| One KV token across28 layers,8 KV heads,128 head dimension | 229,376 bytes =224KiB |
| One full1356-token cache | 311,033,856 bytes =296.625MiB |
| One1295-token document cache | 297,041,920 bytes =283.281MiB |
| Source + native full + custom candidate caches | 876.531MiB |
| Maximum additional128 generated-token KV | 28MiB per decoding branch |
| One16-head full1356×1356 attention matrix | 112.228MiB |

These are tensor payloads, not a peak estimate. Eager scores, probabilities, attention outputs, FFN intermediates, temporary copies and backend resources add memory. The multi-GiB paging cannot be attributed precisely to one allocation from these logs.

Concrete source observations:

1. `experiments/cacheblend/run.py:37–38` loads a full CPU float32 model and then calls `.to('mps')`. This allows host/device staging overlap while checkpoint pages and conversion temporaries are present. It does **not** prove that two entire float32 model copies were simultaneously resident. Loading, hashing, dtype conversion, transfer and unrelated system load were not separately instrumented.
2. `worker.py:58,66,75–78` retains the independent source cache and native-full cache while creating each candidate cache. `del cache`/`empty_cache()` cannot free `source` or `native`, which remain live. The native output survives into the second endpoint. At least three cache payloads can overlap during a candidate forward.
3. Native replay clones the native cache; naive validation keeps candidate and native-suffix caches together. These later steps had not yet run but would also increase liveness. Generation clones an already independent result cache while the caller retains the original result.
4. `core.py:56–67` accumulates corrected chunk tensors, then concatenates them into a second full source cache; validation also retains pre-RoPE key clones for the current chunk. `core.py:120–121` clones old rows before `cat`, even though `cat` already owns its output. These are avoidable copies, but the latter saves only about10MiB for a layer here, not several GiB.
5. Eager attention computes full score/probability tensors. Keep its numerical implementation fixed for the first memory-reduction proposal; changing attention kernels or query tiling would need separate numerical justification.

## Proposed execution order at the same validated precision

This plan is **not implemented or authorized to run**. It keeps cached revision70d244, MPS float32/eager, all retained token IDs, both intact windows, the five methods, ratios,128-token cap, ownership rules and2e-4 mixed tolerance. Excluding Getty is an annotation decision, not a memory adjustment; the largest prompt remains unchanged.

1. **Make loading genuinely streaming and serialized.** Preserve exact stored parameter values, both embedding/head tensors and the original untied runtime topology despite the config's tie flag. Instantiate without allocating a complete CPU float32 parameter set, then materialize/cast/place one checkpoint tensor at a time on MPS, releasing host scratch before the next. Loading the largest tensors first reduces the chance of a large cast scratch buffer overlapping an otherwise full model. Retain all required buffers and assert no meta tensors, missing keys, dtype/device changes or accidental reties.

   Static inspection confirms Accelerate and safetensors are already installed. Transformers5.2 loads with a four-worker conversion pool by default; its installed `core_model_loading.py:1105–1109` supports `HF_DEACTIVATE_ASYNC_LOAD` to serialize that stage. The installed loader documents `device_map='mps'`, but using that blindly is not a guaranteed fix: `modeling_utils.py:4151,4692` invokes an allocator warm-up that can request a model-sized device allocation. That path was not used by the failed CPU-then-`.to()` loader, so it is not a claimed cause of this failure. A reviewed meta/tensor-streaming adapter should avoid that large warm-up, rather than assuming a new device map is safe. No such adapter has been written or tested here.

2. **Gate model loading before cache work.** Add phase markers before/after checkpoint reads, conversions, placement, source composition and each endpoint/decode. Retain the same OS/MPS telemetry and comfort thresholds. If loading alone reaches the warning/critical or swapping limits, stop before computing caches. Do not interpret a driver number below recommended memory as system-wide comfort.

3. **Serialize numerical references to disk instead of keeping caches side by side.** Compute native full once with no composed source resident, record its logits/prefix diagnostics, and write expected per-layer K/V as lossless float32 files, one layer at a time. Release the native output and every alias, synchronize, collect garbage and release unused allocator blocks. Compare the next candidate against one expected layer at a time from disk. Store only the compact native next-token logits for later divergence. Validate complete K/V and logits at the original tolerance—do not weaken checks to save memory.

   Allow the custom full path to run without a composed source cache: its full branch never uses old KV values, although the present API unnecessarily requires/reads a source object. This is a proposed storage/API change with identical full-path arithmetic, not an implementation already made. Native replay and naive reference comparisons should likewise be staged sequentially, never keeping replay, direct, source and candidate caches together unnecessarily. CPU-only offload is not a physical-memory saving on unified memory; disk files must replace deleted live tensors, not accompany a complete CPU copy.

4. **Bound source-composition liveness.** Preallocate owned final layer buffers using the known document length and fill each chunk range directly, rather than retaining all chunk copies plus concatenations. Perform pre-RoPE correction diagnostics one layer at a time rather than retaining28 extra keys. Preserve source checksums, exact corrected values and independent storage. If still necessary, a lossless disk-backed immutable source with one-layer fetches could reduce live source storage, at additional I/O cost. It must preserve the same masks, positions, values and selected indices. This is not a speedup claim.

5. **Use one private method cache at a time.** A method receives an immutable source and creates its own owned result. After initial scoring, transfer that private result into its single greedy continuation instead of cloning another full branch that nobody else needs. Source ownership must remain independent; no cache can be mutated and reused by the next method. Release the result/decode cache before the next method. Stream all comparisons/checksums with bounded scratch, retaining the existing finite checks and immediately fsynced method checkpoints.

6. **Require a newly authorized preflight before any batch.** Before factual evaluation, prove loader parameter equality, native/custom full and ratio1 K/V/logit endpoints, RoPE correction, replay, source immutability and fresh-suffix coverage at the existing tolerance. Observe load, composition, validation and all five methods, including decode high-water memory. Preserve severe-pressure stops, no automatic retries, one model/job, the six-hour watchdog and process-bound caffeinate. No claim that this plan fits comfortably is justified until such a preflight passes. No output-derived workload selection or evidence trimming is part of this proposal.

Serial references can remove roughly one full296.6MiB cache from simultaneous validation, and avoiding a decode clone removes another full cache plus continuation growth at that stage. They are not additive guarantees across unrelated phases. The unavoidable float32 parameter floor remains7.57GiB, and the early pressure shows why load-stage and backend/OS headroom remain the primary uncertainty. This is a lower-memory plan, not proof that the16GiB Mac can complete the evaluation under current conditions.
