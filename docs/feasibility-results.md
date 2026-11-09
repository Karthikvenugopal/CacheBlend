# Attention-output-change feasibility: results

This is a CPU float32 pilot on the cached TinyLlama checkpoint. No novelty claim. Existing baselines were preserved. The local repair control is part of this authorized matched-budget experiment, not a separate window sweep.

Gate 1: **PASS**, maximum absolute first-layer attention-output error 5.96e-08. See [the numerical report](feasibility-gate1.md).

The method was frozen at `2026-11-09T04:21:18.033797+00:00` before held-out execution. Source/config/data fingerprints are in [frozen.json](../artifacts/feasibility/frozen.json).

## Answer utility

| Cohort | Accepted questions | Original clean correct | Edited full correct | Both clean correct | Factual gate passed |
|---|---:|---:|---:|---:|---|
| Development | 8 | 7 | 6 | 6 | False |
| Held-out | 24 | 15 | 13 | 13 | False |

Development has eight questions over three edits; held-out has 24 planned questions over eight edits. Questions within an edit share text and are correlated. Rejected cases remain in JSON and are excluded from denominators. No held-out cases were selected based on model success.

| Held-out stratum | Questions | Edited full correct | Both clean correct |
|---|---:|---:|---:|
| direct | 8 | 8 | 8 |
| derived | 8 | 0 | 0 |
| invariant | 8 | 5 | 5 |

The prespecified reliability gate requires at least 80% correct original+edited clean answers overall and in every stratum. **The factual experiment is inconclusive because this gate failed.**

Known-answer utility requires the expected phrase and excludes a predefined competing answer. It is scored separately from strict short-answer formatting. This conservative rule can reject verbose/uncertain answers and is not a general semantic judge; raw text and token-limit flags are preserved below and in JSON.

| Method | Known-answer correct (all) | Correct on clean pairs | New failures on edited-full-correct cases | Strict short-answer matches |
|---|---:|---:|---:|---:|
| full | 13/24 | 13/13 | 0/13 | 0/24 |
| edit-only | 13/24 | 13/13 | 0/13 | 0/24 |
| local | 13/24 | 13/13 | 0/13 | 0/24 |
| attention | 13/24 | 13/13 | 0/13 | 0/24 |
| delta | 13/24 | 13/13 | 0/13 | 0/24 |
| random-17 | 13/24 | 13/13 | 0/13 | 0/24 |
| random-29 | 13/24 | 13/13 | 0/13 | 0/24 |
| random-43 | 13/24 | 13/13 | 0/13 | 0/24 |

| Method | Direct utility | Derived utility | Invariant utility |
|---|---:|---:|---:|
| full | 8/8 | 0/8 | 5/8 |
| edit-only | 8/8 | 0/8 | 5/8 |
| local | 8/8 | 0/8 | 5/8 |
| attention | 8/8 | 0/8 | 5/8 |
| delta | 8/8 | 0/8 | 5/8 |
| random-17 | 8/8 | 0/8 | 5/8 |
| random-29 | 8/8 | 0/8 | 5/8 |
| random-43 | 8/8 | 0/8 | 5/8 |

Random seeds are three instances of the same frozen random policy, not three separately tuned methods. Counts are descriptive; no significance claim is made.

## Distribution fidelity

| Method | Median next-token KL(full‖method) | Median max absolute logit error | Greedy token matches | Median absolute answer-margin error vs full |
|---|---:|---:|---:|---:|
| full | 0 | 0 | 24/24 | 0 |
| edit-only | 0.00315872 | 0.700882 | 14/24 | 0.148366 |
| local | 0.00109842 | 0.528061 | 16/24 | 0.0727875 |
| attention | 0.00109842 | 0.528061 | 16/24 | 0.0727875 |
| delta | 0.00109842 | 0.528061 | 16/24 | 0.0727875 |
| random-17 | 0.000679159 | 0.32241 | 20/24 | 0.0340004 |
| random-29 | 0.00156664 | 0.549548 | 15/24 | 0.0681851 |
| random-43 | 0.00148334 | 0.528074 | 15/24 | 0.0726055 |

Answer margins are differences between length-averaged teacher-forced expected/competing answer log probabilities. Lower next-token KL does not imply better factual answers. Greedy matches can hide drift; wording differences can preserve the answer.

| Delta versus | KL wins / ties / losses | Identical selected position sets |
|---|---:|---:|
| local | 0 / 24 / 0 | 24/24 |
| attention | 0 / 24 / 0 | 24/24 |
| random-17 | 4 / 6 / 14 | 6/24 |
| random-29 | 17 / 6 / 1 | 6/24 |
| random-43 | 12 / 6 / 6 | 6/24 |

6 questions have only two eligible regions, so selection there is forced. The other 18 questions provide genuine region-choice comparisons; delta still matches local and attention on all of them.

## Measured costs

Every partial selector repairs the same actual edit envelope plus 32 extra document positions through all layers. The common region grid is anchored at the actual edit end. Full and edit-only are reference endpoints, not matched-budget selectors. Selection never uses oracle states or expected answers.

| Method | Selection ms | Repair/prefill ms incl. copy | Fresh question ms | Greedy decode ms | Resume through answer ms | Original collection + resume ms |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.00 | 442.06 | 0.00 | 707.28 | 1189.41 | 1596.89 |
| edit-only | 0.00 | 71.97 | 174.00 | 666.84 | 912.33 | 1434.91 |
| local | 0.01 | 210.36 | 174.78 | 762.91 | 1140.88 | 1564.92 |
| attention | 0.22 | 211.81 | 172.75 | 662.97 | 1051.38 | 1560.59 |
| delta | 2.14 | 211.13 | 174.00 | 691.00 | 1094.78 | 1555.91 |
| random-17 | 0.03 | 209.16 | 175.34 | 587.13 | 981.40 | 1547.45 |
| random-29 | 0.03 | 207.37 | 175.15 | 703.46 | 1126.70 | 1569.81 |
| random-43 | 0.03 | 208.56 | 173.54 | 677.57 | 1099.58 | 1564.02 |

These are medians across cases of two actual timed repetitions, with method order shuffled per case/repetition. The first two questions ran before the user pause and the remaining questions after resumption; raw execution-session metadata is retained. Conditions across sessions were not controlled, so these are functional timing observations. Component medians do not necessarily sum to the total median. Complete prompt tokenization, selection, repair/copies, fresh question and generation are included in resume time. Full recomputation processes the question within its complete native prefill, so its separate question column is zero. Text detokenization, oracle/scoring work and model loading are excluded.

Lifecycle cost charges the original native prefill plus method-specific metadata collection once per edit/question; it does not assume multi-question amortization or hiding work behind tool execution. This is not a GPU benchmark and no proportional savings are inferred from token counts.

| Preparation measurement | Median ms |
|---|---:|
| Original uninstrumented prefill | 408.19 |
| Delta: original prefill + metadata | 416.16 |
| Attention: original prefill + metadata | 409.67 |
| delta: metadata postprocessing | 2.87 |
| attention: metadata postprocessing | 2.41 |

Gate 2 collection has one measurement per method/case; Gate 1 retains three paired repetitions. Signed differences from uninstrumented prefill can be negative because of noise/warming and are not negative metadata costs. Q/O are float32 and log-normalizers float64. Delta retains 16,640 bytes/token (16.25 KiB); attention-to-edit retains 8,448 bytes/token (8.25 KiB), in addition to ordinary KV. Peak process memory was not measured.

## Per-case evidence

Answers below are literal decoded outputs. The actual span and selected positions are recorded in the raw JSON, including token IDs/pieces. Distance applies to the derived table dependency; invariant questions have no declared dependency on the edit.

| Case | Kind / distance | Expected old → new | Original clean answer | Edited full answer | Delta answer | Local answer | Attention answer |
|---|---|---|---|---|---|---|---|
| heldout-dispatch-direct | direct / 0 | east → west | 'The selected direction is east.' | 'The selected direction is west.' | 'The selected direction is west.' | 'The selected direction is west.' | 'The selected direction is west.' |
| heldout-dispatch-derived | derived / 259 | York → Oslo | 'This dispatch goes to York and Oslo.' | 'For this dispatch, east goes to York and west goes to' | 'This dispatch goes to York and Oslo.' | 'This dispatch goes to York and Oslo.' | 'This dispatch goes to York and Oslo.' |
| heldout-dispatch-invariant | invariant / None | receipt → receipt | 'The document is the archive records routine inspections. Each' | 'The document is the archive records routine inspections. Each' | 'The document is the archive records routine inspections. Each' | 'The document is the archive records routine inspections. Each' | 'The document is the archive records routine inspections. Each' |
| heldout-warehouse-direct | direct / 0 | red → blue | 'The tag assigned is red.' | 'The tag assigned is blue.' | 'The tag assigned is blue.' | 'The tag assigned is blue.' | 'The tag assigned is blue.' |
| heldout-warehouse-derived | derived / 94 | copper → silver | 'The material belongs to the red tag, which is used for' | 'The material belongs to the blue tag, which is held by' | 'The material belongs to the blue tag, which is assigned to' | 'The material belongs to the blue tag, which is assigned to' | 'The material belongs to the blue tag, which is assigned to' |
| heldout-warehouse-invariant | invariant / None | Nora → Nora | 'The answer is Nora.' | 'The answer is Nora.' | 'The answer is Nora.' | 'The answer is Nora.' | 'The answer is Nora.' |
| heldout-clinic-direct | direct / 0 | low → high | 'The patient priority is low in the given document.' | 'The patient priority is high in the given document.' | 'The patient priority is high in the given document.' | 'The patient priority is high in the given document.' | 'The patient priority is high in the given document.' |
| heldout-clinic-derived | derived / 3 | Tuesday → Friday | 'The patient priority is low, and the appointment for low priority' | 'The patient priority is high, so the answer is "On' | 'The patient priority is high, so the answer is "T' | 'The patient priority is high, so the answer is "T' | 'The patient priority is high, so the answer is "T' |
| heldout-clinic-invariant | invariant / None | online → online | 'The answer is given in the document. The patient priority is' | 'The answer is given in the document.' | 'The answer is given in the document.' | 'The answer is given in the document.' | 'The answer is given in the document.' |
| heldout-account-direct | direct / 0 | basic → plus | 'The account tier is basic.' | 'The account tier is plus.' | 'The account tier is plus.' | 'The account tier is plus.' | 'The account tier is plus.' |
| heldout-account-derived | derived / 259 | 10 → 20 | 'The allowance table grants basic accounts 10 cred' | 'The allowance table grants basic accounts 10 cred' | 'The allowance table grants basic accounts 10 cred' | 'The allowance table grants basic accounts 10 cred' | 'The allowance table grants basic accounts 10 cred' |
| heldout-account-invariant | invariant / None | monthly → monthly | 'The answer is monthly.' | 'The answer is monthly.' | 'The answer is monthly.' | 'The answer is monthly.' | 'The answer is monthly.' |
| heldout-museum-direct | direct / 0 | nine → five | 'The opening hour is nine.' | 'The opening hour is five.' | 'The opening hour is five.' | 'The opening hour is five.' | 'The opening hour is five.' |
| heldout-museum-derived | derived / 94 | morning → afternoon | 'The answer is "Morning" based on the given document' | 'The answer is "Morning."' | 'The answer is "Morning" based on the given document' | 'The answer is "Morning" based on the given document' | 'The answer is "Morning" based on the given document' |
| heldout-museum-invariant | invariant / None | children → children | 'The opening hour is nine, and the archive records routine ins' | 'The opening hour is five, and the archive records routine ins' | 'The opening hour is five, and the archive records routine ins' | 'The opening hour is five, and the archive records routine ins' | 'The opening hour is five, and the archive records routine ins' |
| heldout-permit-direct | direct / 0 | gold → silver | 'The answer is "gold."' | 'The permit category is silver.' | 'The permit category is silver.' | 'The permit category is silver.' | 'The permit category is silver.' |
| heldout-permit-derived | derived / 3 | north → south | 'The permit category is gold, which means it is a gold' | 'The permit category is silver, which means it is a silver' | 'The permit category is silver, which means it is a silver' | 'The permit category is silver, which means it is a silver' | 'The permit category is silver, which means it is a silver' |
| heldout-permit-invariant | invariant / None | June → June | 'The permit expires in June, as indicated in the document' | 'The permit expires in June, as indicated in the document' | 'The permit expires in June, as indicated in the document' | 'The permit expires in June, as indicated in the document' | 'The permit expires in June, as indicated in the document' |
| heldout-rail-direct | direct / 0 | day → night | 'The service chosen is day.' | 'The service was night.' | 'The service chosen is night.' | 'The service chosen is night.' | 'The service chosen is night.' |
| heldout-rail-derived | derived / 259 | Rome → Paris | 'The chosen service ends in Rome and Paris.' | 'The chosen service ends in Rome.' | 'The chosen service ends in Rome and Paris.' | 'The chosen service ends in Rome and Paris.' | 'The chosen service ends in Rome and Paris.' |
| heldout-rail-invariant | invariant / None | digital → digital | 'The answer is "digital."' | 'The answer is "Digital."' | 'The answer is "digital."' | 'The answer is "digital."' | 'The answer is "digital."' |
| heldout-parcel-direct | direct / 0 | 12 → 18 | 'The answer is 12.' | 'The parcel code is 18.' | 'The answer is 18.' | 'The answer is 18.' | 'The answer is 18.' |
| heldout-parcel-derived | derived / 94 | land → sea | 'According to the carrier table, code 12' | 'According to the carrier table, code 12' | 'According to the carrier table, code 12' | 'According to the carrier table, code 12' | 'According to the carrier table, code 12' |
| heldout-parcel-invariant | invariant / None | Ada → Ada | 'The answer is Ada.' | 'The answer is Ada.' | 'The answer is Ada.' | 'The answer is Ada.' | 'The answer is Ada.' |

No original baseline source or earlier results were replaced. The first partial development run is retained separately; it motivated a fixture-length correction before freezing. There was no held-out method tuning. See [the experiment protocol](../experiments/feasibility/protocol.md), [raw held-out records](../artifacts/feasibility/heldout.json), and [development records](../artifacts/feasibility/development.json).
