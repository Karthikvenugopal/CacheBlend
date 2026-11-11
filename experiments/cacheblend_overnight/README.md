# Bounded manually annotated diagnostic

This separate one-pass evaluation was authorized by the user. Its representative preflight stopped on sustained memory pressure and swapping before any of the five generated-method evaluations completed. **Do not automatically restart, shrink inputs, change precision, or launch another experiment.** No commit or publication is authorized.

Read [the status report](../../artifacts/cacheblend_overnight/report.md), [protocol.md](protocol.md), and immutable [frozen.json](frozen.json). The original CacheBlend implementation, 400-output pilot and edit-repair archives are unchanged.

`prepare.py` records manual claim quotations, character offsets, a fixed disjoint candidate pool and every inclusion/exclusion; it refuses to overwrite a freeze. `common.py` provides atomic fsynced checkpoints and reference-compatible Qwen scoring. `worker.py` uses the unchanged core and cached model. `launch.py` supervises one inference worker with memory and wall-time stops. `report.py` preserves incomplete runs and descriptive paired statistics when results exist. The protocol and code hashes at launch are recorded in run-state.json.

The historical authorized command was:

```sh
/usr/bin/caffeinate -i .venv/bin/python -u experiments/cacheblend_overnight/launch.py > artifacts/cacheblend_overnight/supervisor.log 2>&1
```

It is now stopped, including caffeinate. The supervisor refuses automatic retries of a pressure-stopped run. Explicit resume is reserved for an interrupted run with identical hashes and remaining time under the original deadline; it never reruns completed method results. That capability is not permission for another experiment.
