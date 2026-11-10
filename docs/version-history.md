# Local history provenance

This directory had no Git repository before the user requested separate commits for completed work. The initial history groups the existing files into six logical milestones; it does not reconstruct every intermediate source revision. Some documentation contains later assessments of earlier work.

At the user's request, author dates use recorded milestone dates below. Commit creation dates remain the actual creation times. The dates are taken from retained artifact modification times, rather than invented work sessions. Raw experiment timestamps remain unchanged inside artifacts.

| Group | Author date | Timestamp evidence |
|---|---|---|
| Runnable inference foundation and baseline evidence | 2026-11-08 19:54:08 -0800 | artifacts/local-tests.txt |
| Related-work review and extension scoping | 2026-11-08 20:04:32 -0800 | artifacts/review/foundation-preserved.json |
| Attention update, sparse repair and numerical gate | 2026-11-08 20:16:47 -0800 | artifacts/feasibility/gate1.json |
| Frozen feasibility pilot and rejection assessment | 2026-11-09 15:49:24 -0800 | artifacts/feasibility/final-verification.json |
| Bounded selector diagnosis | 2026-11-09 17:50:17 -0800 | artifacts/feasibility/diagnostic/final-checks.json |
| Bounded oracle screen and consolidated project documentation | 2026-11-09 17:57:02 -0800 | artifacts/oracle_bound/verification.json |

The virtual environment, Python bytecode, pytest cache and generated package metadata are ignored. Evidence artifacts, including literature/source snapshots, are retained. No remote publishing is part of this operation.

Four subsequent commits group the archive/reproduction work. Author dates use recorded JSON milestone timestamps; commit creation dates remain current. As above, these group the final retained files rather than reconstructing every intermediate edit.

| Group | Author date | Timestamp evidence |
|---|---|---|
| Archive and reproduction-target assessment | 2026-11-09 18:20:40 -0800 | archives/kvrepair-2026-11-09/manifest.json, archived_utc |
| Bounded Qwen3 reproduction setup and retained BF16 failure | 2026-11-09 18:42:50 -0800 | artifacts/qwen_reproduction/execution-status.json, completed_utc |
| Native-cache backend/precision diagnosis | 2026-11-09 18:48:35 -0800 | artifacts/qwen_reproduction/endpoint_precision/summary.json, completed_utc |
| Completed six-case float32 reproduction and utility audit | 2026-11-09 19:04:02 -0800 | artifacts/qwen_reproduction/float32_resume/continued/summary.json, completed_utc |

Downloaded checkpoint weights and their local download cache are excluded through `.models/`. The tracked `.pt` files contain small diagnostic logits/activation traces, not the model parameters. All original BF16 failures, partial runs, collection corrections and final outcomes are preserved.
