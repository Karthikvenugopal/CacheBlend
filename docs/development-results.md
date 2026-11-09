# Development observations

Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`; torch.float32; cpu; seed 17.

These are development examples, not evidence of general repair quality. `answer_correct` in JSON is strict normalized short-answer matching. A false score may reflect verbosity rather than a wrong fact. Inspect raw answers below. Logit drift, continuation agreement and factual correctness measure different things.

Ranges below are zero-based, half-open indices in the complete rendered prompt, including chat tokens. Times are single functional measurements including cache copying; they are not speedup estimates.

## refund-change

Old: Refunds are permitted within 30 days. Shipping is free.

Edited: Refunds are permitted within 14 days. Shipping is free.

Question: Within how many days are refunds permitted? Reply with only the number.

Expected: ['14']; should change: True.

Original full-recompute answer: `'Refunds are permitted within 30 days.'`

Actual changed envelope: [26, 28); prompt length 65.

| Position | Old ID / decoded piece | New ID / decoded piece |
|---|---|---|
| 26 | 29941 / '3' | 29896 / '1' |
| 27 | 29900 / '0' | 29946 / '4' |

| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |
|---|---|---|---|---|---|---|---|---|
| full | 'Refunds are permitted within 14 days.' | [] | [[0, 65]] | 0 | 0 | True | 286.1 | 1222.4 |
| prefix | 'Refunds are permitted within 14 days.' | [[0, 26]] | [[26, 65]] | 0 | 0 | True | 282.8 | 1196.6 |
| naive (approximate) | 'Refunds are permitted within 14 days.' | [[0, 26], [28, 64]] | [[26, 28], [64, 65]] | 0.220125 | 0.000646406 | True | 245.8 | 1120.6 |

## refund-invariant

Old: Refunds are permitted within 30 days. Shipping is free.

Edited: Refunds are permitted within 14 days. Shipping is free.

Question: Is shipping free? Reply with only yes or no.

Expected: ['yes']; should change: False.

Original full-recompute answer: `'Yes, shipping is free.'`

Actual changed envelope: [26, 28); prompt length 61.

| Position | Old ID / decoded piece | New ID / decoded piece |
|---|---|---|
| 26 | 29941 / '3' | 29896 / '1' |
| 27 | 29900 / '0' | 29946 / '4' |

| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |
|---|---|---|---|---|---|---|---|---|
| full | 'Document:\nRefunds are permitted within 14' | [] | [[0, 61]] | 0 | 0 | True | 223.6 | 1079.5 |
| prefix | 'Document:\nRefunds are permitted within 14' | [[0, 26]] | [[26, 61]] | 0 | 0 | True | 210.8 | 1041.2 |
| naive (approximate) | 'Yes, shipping is free.' | [[0, 26], [28, 60]] | [[26, 28], [60, 61]] | 0.132815 | 0.000660096 | False | 225.2 | 761.8 |

## controlled-direct

Old: Record: Mira's access code is 42. The office is in Rome.

Edited: Record: Mira's access code is 73. The office is in Rome.

Question: What is Mira's access code? Reply with only the code.

Expected: ['73']; should change: True.

Original full-recompute answer: `"Mira's access code is 42."`

Actual changed envelope: [28, 30); prompt length 65.

| Position | Old ID / decoded piece | New ID / decoded piece |
|---|---|---|
| 28 | 29946 / '4' | 29955 / '7' |
| 29 | 29906 / '2' | 29941 / '3' |

| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |
|---|---|---|---|---|---|---|---|---|
| full | "Mira's access code is 73." | [] | [[0, 65]] | 0 | 0 | True | 254.7 | 1093.1 |
| prefix | "Mira's access code is 73." | [[0, 28]] | [[28, 65]] | 0 | 0 | True | 206.4 | 1049.1 |
| naive (approximate) | "Mira's access code is 73." | [[0, 28], [30, 64]] | [[28, 30], [64, 65]] | 0.240728 | 0.000882627 | True | 234.6 | 1098.2 |

## controlled-derived

Old: The active label is red. Lookup: red means Rome; blue means Lima.

Edited: The active label is blue. Lookup: red means Rome; blue means Lima.

Question: Which city does the active label mean? Reply with only the city.

Expected: ['Lima']; should change: True.

Original full-recompute answer: `'The active label means Lima, as it is the city with'`

Actual changed envelope: [23, 24); prompt length 64.

| Position | Old ID / decoded piece | New ID / decoded piece |
|---|---|---|
| 23 | 2654 / 'red' | 7254 / 'blue' |

| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |
|---|---|---|---|---|---|---|---|---|
| full | 'The active label means Lima, as it is blue and is' | [] | [[0, 64]] | 0 | 0 | True | 223.0 | 1071.8 |
| prefix | 'The active label means Lima, as it is blue and is' | [[0, 23]] | [[23, 64]] | 0 | 0 | True | 214.9 | 1097.6 |
| naive (approximate) | 'The active label means Lima, as it is blue and means' | [[0, 23], [24, 63]] | [[23, 24], [63, 64]] | 0.738904 | 0.00232016 | False | 151.3 | 985.7 |

## museum-change

Old: The museum opens at nine on Monday. Entry is free for children.

Edited: The museum opens at ten on Monday. Entry is free for children.

Question: When does the museum open on Monday? Reply with one word.

Expected: ['ten']; should change: True.

Original full-recompute answer: `'Answer: Monday at 9:00 AM'`

Actual changed envelope: [23, 24); prompt length 60.

| Position | Old ID / decoded piece | New ID / decoded piece |
|---|---|---|
| 23 | 14183 / 'nine' | 3006 / 'ten' |

| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |
|---|---|---|---|---|---|---|---|---|
| full | 'Answer: Monday at 10:00 AM' | [] | [[0, 60]] | 0 | 0 | True | 217.6 | 1061.6 |
| prefix | 'Answer: Monday at 10:00 AM' | [[0, 23]] | [[23, 60]] | 0 | 0 | True | 203.8 | 1060.6 |
| naive (approximate) | 'Answer: Monday at 10:00 AM' | [[0, 23], [24, 59]] | [[23, 24], [59, 60]] | 0.359679 | 0.00164348 | True | 153.0 | 984.5 |

## museum-invariant

Old: The museum opens at nine on Monday. Entry is free for children.

Edited: The museum opens at ten on Monday. Entry is free for children.

Question: Who gets free entry? Reply with one word.

Expected: ['children']; should change: False.

Original full-recompute answer: `'Free entry for children.'`

Actual changed envelope: [23, 24); prompt length 57.

| Position | Old ID / decoded piece | New ID / decoded piece |
|---|---|---|
| 23 | 14183 / 'nine' | 3006 / 'ten' |

| Method | Answer | Reused | Recomputed | Max logit delta | KL(full‖method) | Same greedy tokens | Prefill ms | Through answer ms |
|---|---|---|---|---|---|---|---|---|
| full | 'Free entry for children.' | [] | [[0, 57]] | 0 | 0 | True | 220.4 | 601.3 |
| prefix | 'Free entry for children.' | [[0, 23]] | [[23, 57]] | 0 | 0 | True | 206.5 | 587.6 |
| naive (approximate) | 'Free entry for children.' | [[0, 23], [24, 56]] | [[23, 24], [56, 57]] | 0.43793 | 0.00379795 | True | 155.4 | 519.8 |

