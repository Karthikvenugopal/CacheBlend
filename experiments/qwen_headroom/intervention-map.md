# What caused the retained failures

This diagnostic preserves two **published synthetic token-alignment interventions**. It is not a test of arbitrary document edits and is not equivalent to the earlier fresh-question pilot.

## Exact intervention

The source prefills the entire old chat prompt: role, mutable field, policy, 18 filler notes, request, user/assistant delimiters and empty thinking block. It independently renders/tokenizes the new chat, finds the longest common prefix/suffix and inserts space-token ID 220 into the shorter changed middle. It then creates an independent old cache, transplants only aligned changed-span K/V from a complete new prefill at every layer, leaves the downstream prompt cache old, discards the final cached token, and feeds that final token again. The request is already in the old cached suffix; it is not a newly appended question.

| Case | Actual edit envelope | Aligned length | Actual old field text | Actual new field text | Reused downstream K/V | Fresh readout |
|---|---|---:|---|---|---|---|
| airline_cancel | [16,19) | 324 | `fare_type: flexible  ` | `fare_type: basic_economy` | [19,323), 304 tokens | token 323 |
| bank_withdraw | [16,19) | 331 | `account_balance: 8200 USD` | `account_balance: 3  0 USD` | [19,330), 311 tokens | token 330 |

All ranges are zero-based and half-open. Prefix [0,16) is identical. Both changed envelopes contain three tokens. The transplant source has already paid for a complete edited prefill; the resulting unrepaired cache is an intervention, not an efficient execution implementation.

Airline's raw old/new token lengths are 322/324. Two space tokens pad the old middle. Decoding and retokenizing that padded old cache **does not** reproduce its aligned token IDs. Bank's raw lengths are 331/329. The shared suffix begins with digit `0`, so padding splits the intended `30` into `3  0`; this is not a literal edit to the intended well-formed amount. Bank's padded strings do retokenize to their aligned IDs, but their monetary-field semantics remain questionable. Do not silently clean either input.

The observed full/field-only outputs were `deny`/`refund` and `Decline`/`Approve`. Airline has a clear policy-decision label. For bank, report recovery of the full-reference output and agreement with the **intended** `decline` label separately from literal-document factual correctness. The original classification treated the amount as the intended 30 USD; this audit narrows that interpretation. The malformed field prevents claiming an ordinary 8200→30 factual-edit result here.

## Relation to the original project

The original safe baselines independently retokenize complete literal prompts and reject unequal token lengths. They would reject these raw old/new pairs. They do not insert padding. The old Llama `naive` path recomputes the changed envelope from its exact prefix, splices old downstream states, and freshly replays the last prompt token. Its dependency pattern resembles this complete-prompt intervention **once aligned token IDs are stipulated**, but its input contract and implementation differ. Our previous feasibility pilot instead cached only the document and processed a fresh question, so its query-conditioned states were not reused. None of these settings is interchangeable.

For the new repair test, perform actual native Qwen3 selected-row recomputation from independently owned old K/V, always including [16,19). Unselected downstream states remain stale; selected rows attend over the full causally visible mixed cache. Respect Qwen3's Q/K normalization and RoPE through its native model/layer code. The repair function must not receive a clean downstream cache or answer labels. Confirm edit-only actual recomputation matches the published field-only intervention's logits and generated answer before searching. Confirm full-suffix recomputation matches native edited prefill. These gates establish operator compatibility, not exactness of partial repair.

Frozen exact IDs, decoded prompts, padding, region coordinates and all enumerated supports are in `fixed_cases.json`. Original baselines, source reproduction, BF16 failure and rejected selector remain unchanged.
