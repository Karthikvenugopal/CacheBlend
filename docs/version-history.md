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

## GitHub backup and remaining milestones

On 2026-11-20 the user authorized committing the remaining retained work in development order and pushing the project to GitHub. This supersedes earlier statements withholding commit/push permission, solely for this backup. It does not reopen experiments. Historical run reports and their scope statements remain unchanged.

The existing twelve commits are preserved without rewriting their dates or contents. The last two cover the bounded repair-headroom negative result (`376169d`) and the CacheBlend mechanism/fixed quality pilot (`6713d78`). Three additional commits group the previously untracked final files by their recorded development stages:

| Group | Author date (UTC) | Timestamp evidence |
|---|---|---|
| Frozen overnight diagnostic and pressure-stopped preflight | 2026-11-11 08:01:36 +0000 | `artifacts/cacheblend_overnight/run-state.json` start_epoch + `status.json` elapsed_s |
| Annotation audit, corrected frozen-v2 set and memory diagnosis | 2026-11-11 15:25:24 +0000 | `artifacts/cacheblend_overnight/review_v2/review-manifest.json` recorded_utc |
| Storage refactor and single pressure-stopped memory preflight | 2026-11-11 16:03:35 +0000 | `artifacts/cacheblend_overnight/memory_v3/run-state.json` start_epoch + `status.json` elapsed_s |

These author dates identify the recorded milestones, truncated to whole seconds; they do not claim each retained file already had its final content at that instant. Some reports were finalized shortly after the recorded run. Committer dates record the actual commit creation times. Intermediate source revisions cannot be recovered from the final files. A separate present-day documentation commit records this provenance and links the retained outcomes from the README.

The GitHub backup defaults to a private repository because this remains internal learning work. Raw failures, incomplete runs, annotation caveats and negative results are included. Model weights, the virtual environment and caches remain excluded.
