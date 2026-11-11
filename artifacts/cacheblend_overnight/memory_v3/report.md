# Single frozen-v2 preflight result

**Stopped on sustained severe memory pressure during streaming model loading. 0/5 method outputs completed. No retry or remaining-item execution occurred.**

Input: longest retained frozen-v2 item `wikimqa:025`, 1356 input tokens, unchanged. Cached model revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, MPS float32/eager. All source/checkpoint hashes verified before loading. The original inputs, annotations, executor, failed run and review files passed preservation verification (45 files).

| Observation | Value |
|---|---:|
| Supervisor elapsed including cleanup | 45.22s |
| Worker elapsed to stop checkpoint | 36.43s |
| Process high-water RSS | 1.413GiB |
| Sampled MPS tensor maximum | 7.569GiB |
| Sampled MPS driver maximum | 7.586GiB |
| New system-wide swapouts | 5.725GiB |
| New system-wide swapins | 0.413GiB |
| Highest kernel pressure level | 4 (critical) |
| Lowest recorded OS free statistic | 24% |

These counters overlap and must not be added. MPS maxima are samples at one-second intervals plus phase events, rather than exact allocation high-water marks. RSS is the process high-water counter. Swap traffic is cumulative and system-wide; it cannot be attributed entirely to this process or read as simultaneous resident memory.

The streaming loader approached the unchanged7.57GiB parameter floor but was stopped before it returned. No native reference, model-level numerical gate, composition, or generated method output was reached. No numerical failure was observed; numerical correctness on this real-model preflight remains **unverified**, not passed. Partial generated-token IDs are empty. The stop checkpoint records the loading phase, error and traceback. Supervisor waited for worker exit and released its process-bound caffeinate; process exit releases device/host allocations. There are no completed outputs to checkpoint.

Implementation verification: three CPU tests passed. The new executor exactly matched original K/V, logits, selected rows and work across full, naive, zero,16%,32%, and full-selection endpoints on a tiny random Qwen model; native decode preserved source ownership. Existing scoring and pressure-guard tests also passed. All changed Python modules compile. Disk validation and real-model gates remain untested on MPS because this single authorized preflight stopped during loading.

The new version implements bounded-slice parameter placement with exact stored-value checks, serialized complete disk K/V references, a source-free custom full path, preallocated source buffers, removal of redundant cache copies, and private cache transfer into decode. These implementation changes passed their stated CPU checks; this run does not demonstrate that they reduce total physical-memory pressure enough for this machine. The original failed run is not a controlled comparison: OS workload and baseline swap differed. Lower process RSS alone is not proof of sufficient memory headroom on unified memory.

**There is insufficient demonstrated memory headroom for serial execution of the remaining items under the observed conditions.** The stop happened before any attention or validation-cache work, so reducing validation overlap cannot resolve the loading pressure observed here by itself. No new precision, workload, assets or additional experiments were substituted. The remaining batch is neither authorized nor queued.

Evidence: `status.json`, `stop-request.json`, `worker-status.json`, `worker.log`, `memory.jsonl`, `worker-memory.jsonl`, `sampled-worker-memory.jsonl`, `run-state.json`, `scope.json`, and `preservation-verification.json` in this directory. Implementation: `experiments/cacheblend_overnight/memory_v3/README.md` and adjacent Python files.
