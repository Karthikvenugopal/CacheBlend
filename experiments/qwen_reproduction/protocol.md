# Prepared six-pair reproduction; awaiting download and pretrained-run authorization

Scope: only `retail_refund`, `airline_cancel`, `bank_withdraw`, `doc_access`, `customs_route`, `oncall_route`, in that order. These are the six paired regressions identified retrospectively in an existing published eight-task record. This is a selected reproduction set, not a held-out evaluation or reliability estimate. No replacement cases, prompt tuning, parameter sweeps, cloud compute, repair methods or selectors.

## Provenance and unresolved exactness

Paper: [Models Take Notes at Prefill](https://arxiv.org/html/2606.17107v1). Original-author repository revision: `e9085eafcc6c83e60c548de69060c4bd5b210c96`. Already retained source: `esys/diverse_tasks.py`, `esys/mech_diverse.py`, `esys/mech_suite.py`, `e1/align.py`; results: `results/mech_diverse_qwen3_1p7b.json`. See `artifacts/reproduction-review/reading-manifest.json` for hashes. The six published full/field-only/erratum choices are correct/stale/correct in every selected case. The complete eight-case record, including its two excluded controls, remains preserved.

**The inspected loader and result do not record a model revision. The paper identifies the official checkpoint but not its exact hash.** Its exact original model/tokenizer revision and runtime dependency versions cannot be verified from these artifacts. Our explicit candidate pin is `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, currently shown in official repository metadata. It is not claimed to be an author-confirmed pin. A successful run can reproduce the reported choices under this declared environment; it cannot establish a bitwise recreation of an unrecorded environment.

## Prompts, cache intervention and scoring

`cases.json` freezes the six task definitions and full old/new/erratum **user-message bodies**, including all 18 filler lines, newlines and one-word request. They were constructed from inspected literal task definitions without importing the external modules. Apply the candidate tokenizer's native user-only chat template, `add_generation_prompt=True`, `enable_thinking=False`. Do not add a system message, decision scaffold or chain of thought. The reviewed official single-user template expands to `<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. Freeze these full expected renderings in `cases.json` and assert the downloaded tokenizer reproduces them. Actual token IDs and alignment padding are checked and retained after authorization; they cannot be verified from a tokenizer not yet downloaded.

Mirror the published alignment: independently tokenize both complete renderings with `add_special_tokens=True`; identify their longest shared prefix and nonoverlapping shared suffix; append the tokenizer's final space-token ID to the shorter changed middle so old and new sequences have equal length. Record padding counts explicitly. This artificial token alignment is part of the reproduced intervention, not the original foundation's valid equal-length-edit interface. For erratum only, the source tokenizes the complete chat with `add_special_tokens=False`; retain that asymmetry. Stop if a prompt exceeds 2,048 tokens; never shorten it.

Full state is a native edited prefill. Field-only is an independent old-prefill cache with edited-span K/V replaced by rows from the native edited cache, at every layer. This reproduces the source's intervention and includes its full clean-prefill cost; it is not an efficient repair algorithm. All later cached prompt rows remain stale. The fresh decision readout discards the cached final token, feeds the edited final token at position L−1, and obtains logits. Masks/positions follow the native Qwen3 causal forward. No ordinary fresh-question split from our previous pilot is substituted.

Restricted scoring exactly follows the source: tokenize the declared correct and stale words **without a leading space**, take each first token only, compare their float32 logits, and choose correct on a tie (`>=`). Log full alternative tokenizations, first IDs, scores, margin and unrestricted vocabulary argmax. Assert distinct first-token IDs. These preferences do not show that a model actually generates either answer.

Stage 1 records all six cases for full, field-only and erratum (18 published comparisons); original-full preference is an additional six-case control. The expected selected-set published result is full 6/6, field-only 0/6, erratum 6/6. Preserve every discrepancy instead of adjusting settings. A tiny random architecture preflight is required; native edited-prefill versus cached-final-token numerical/readout mismatches stop for investigation.

## Generation and utility

Stage 2 uses precisely the same six aligned old/new prompts, cache intervention and last-token convention. Run ordinary full-vocabulary greedy decoding, batch one, `model.eval()` and inference mode, seed 17, deterministic algorithms enabled, four CPU threads, no sampling, one beam, no penalties, no vocabulary masks or logit bias. Stop at either native generation-config EOS ID (151645 or 151643) or **64 new tokens**, with `enable_thinking=False` already in the rendered prompt. No appended decision text, forced answers, label-guided stopping, retry with a longer limit or prompt rewrites. Retain raw IDs, text and truncation flags for original full, edited full and field-only: 18 sequences, at most 1,152 emitted tokens. Erratum generation is outside scope; its restricted result is retained separately.

Score reproduction and generated utility are separate tables. Manually classify generated answers against the fixed policy as correct, explicitly wrong, nonanswer/ambiguous, or truncated. Count a meaningful reuse regression only when edited full supplies a clear correct generated decision and field-only supplies a clear wrong one. Never promote token preference, harmless verbosity, inability of the full model, or a missing truncated answer to a demonstrated factual error. Preserve all six outcomes regardless of eligibility.

If none qualify, stop and shelve. If some qualify, report which failures survive unrestricted generation and what the published intervention does and does not establish about repair headroom. Small actual-recomputation headroom remains a separate gate; do not run a transplant sweep, add a repair operator or propose a selector in this reproduction. Reconsider development only after useful actual-repair headroom is established under a separately bounded assessment.

## Compatibility and resources

Use a separate native Qwen3 runner. Existing `CacheEngine` explicitly supports Llama only; Qwen3 also normalizes Q/K and must not be passed through the Llama repair adapter. All frozen foundation/attention code remains unchanged. Installed Transformers 5.2.0 exposes native Qwen3, `DynamicCache`, `cache_position`, eager attention and `logits_to_keep`. Use `trust_remote_code=False` and offline-only model/tokenizer loading. Download and execution are separate commands.

Author code uses CUDA/BF16; our proposed primary run uses **MPS/BF16/eager**, preserving precision but changing backend/runtime. The default eager loader is visible in source; the original run's attention environment variable is not recorded. Random two-layer Qwen3 BF16 checks passed on CPU and MPS: native/cached final logits had zero observed max error, cache ownership check passed. This does not validate pretrained output fidelity. No implicit CPU/FP32 fallback or dependency installation; stop on pretrained incompatibility and retain the error.

Official file listing is approximately **4.08 GB decimal (3.80 GiB)**, including two shards and tokenizer/config files. The tensor index declares 4,063,479,808 payload bytes; headers/tokenizers are additional. Tied embedding/output storage explains why download payload exceeds unique live parameter storage. A meta-device construction from official dimensions counts **1,720,574,976 unique parameters**, 3,441,149,952 BF16 bytes (3.205 GiB), or 6.410 GiB FP32. Actual tying must be confirmed when loading.

BF16 KV storage is `2×28×8×128×2 = 114,688 bytes/token`; one 2,048-token cache is 224 MiB, with roughly 4–5 transient/branch caches possible. Keep one case at a time, retain only final logits, and do not retain all-layer dense attention maps. Budget approximately 6–10 GiB for loading/runtime/allocator peaks, **an estimate rather than a measured peak**, in addition to other applications' needs. MPS shares the machine's 16 GiB. At preparation, free disk was about 12.94 GB; downloader rechecks space and requires payload plus a 2-GiB reserve. No saved FP32 checkpoint or additional model copy is planned.

## Exact commands to authorize

From the project root, command 1 requires network access and permission for approximately 4.08 GB of weights/tokenizer assets. It selects only ten named files, max one download worker, disables Xet chunk caching, pins revision, checks a 4.2-GB ceiling and records sizes/digests. Files stay in ignored `.models/`. The network metadata query occurs before weight transfer, and a size increase beyond the bound stops it.

```sh
.venv/bin/python experiments/qwen_reproduction/download.py
```

Commands 2 and 3 require execution outside the sandbox for MPS. Model loading is offline-only; neither command can initiate downloads. Stage 3 verifies prepared inputs, backend/precision, token IDs and restricted decisions against stage 2's saved record before generating. No command is run automatically on approval of a different command.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u experiments/qwen_reproduction/run.py restricted --device mps --dtype bfloat16
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u experiments/qwen_reproduction/run.py generation --device mps --dtype bfloat16
```

Approve these as one bounded scope only if the declared model-revision uncertainty and backend difference are acceptable. No changes to dependencies, no remote-code execution, cloud compute, broader sweep, repair development or selector optimization are included. Results checkpoint to `artifacts/qwen_reproduction/`; existing outputs cannot be overwritten.

Sources for resource figures: [official file listing](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e), [official config](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json), [tensor index](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/model.safetensors.index.json). The model history's current pin predates the paper, which makes it a reasonable candidate but does not authenticate the authors' exact run.

## Compatibility amendment before any factual case

The first authorized restricted-stage launch stopped at the embedding/output identity assertion before evaluating a case. Transformers 5.2 retained two separate parameters despite the config declaring tied embeddings. Streaming checks of both BF16 tensor payloads found identical SHA-256 values, recorded in `embedding-compatibility.json`. Restore the declared alias only after `torch.equal` verifies the loaded tensors, before MPS transfer. This changes storage ownership, not any weight value, prompt, precision, scoring rule, token budget or generation setting. Preserve the original prepared hashes, runner and failed-launch log under `artifacts/qwen_reproduction/`; amend the runner/protocol hashes before evaluating the six cases. No dependency downgrade or extra factual evaluation is involved.
