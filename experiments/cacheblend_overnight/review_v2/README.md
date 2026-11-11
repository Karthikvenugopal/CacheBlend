# Annotation and memory review — no inference

The [annotation audit](../../../artifacts/cacheblend_overnight/review_v2/annotation-audit.md) documents the original selection and all ten frozen items, with source locations, exact quotations, explicit chains and qualifications. The annotations are **agent-created, not official and not human-validated**. Nine items are retained; Getty is conservatively excluded for ambiguous parent-link coreference, without replacement. A separate spelling/alias limitation is not treated as a missing reasoning hop.

[frozen-v2.json](frozen-v2.json) is the final review-only corrected set. All retained input IDs, prompts, labels and method settings remain unchanged. Its parent is the original ten-item `../frozen.json`, whose SHA-256 remains `6cd0b3bf5d384372a29c377c5cdc2f0be69c8b42babf84c9409d450cf91fe519`. The original set and failed-run code/logs were not edited. A preliminary v2 metadata draft is also preserved; it is not the final set.

The [memory diagnosis and proposal](../../../artifacts/cacheblend_overnight/review_v2/memory-diagnosis-and-plan.md) distinguishes observed loading pressure from later cache overlap, calculates tensor payloads and proposes storage/lifetime changes at the same precision. It does not assert these changes will fit. No loader, executor, guard or selector change was implemented, and no run is queued.

`audit_annotations.py` and `diagnose_memory.py` import only the Python standard library. They inspect existing JSON, text, log and checkpoint-header metadata; they do not load tensor weights, initialize a model/tokenizer, make network calls, access MPS or launch inference. The annotation script refuses to overwrite an existing version. These review scripts do not authorize any evaluation command.
