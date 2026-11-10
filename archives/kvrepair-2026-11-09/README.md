# Archived experiment

Status: closed. Stop selector development and further TinyLlama factual sweeps.

`experiment-and-results.tar.gz` preserves the complete committed project at `697909cb603c832f753ab9e88076e9139d411771`: source, tests, examples, protocols, literature snapshots, raw results and assessments. The working source remains in place for inspection. Model weights and the virtual environment are not included. `manifest.json` records the source tree, SHA-256 of the archive and preserved inference-source hashes.

The validated first-layer attention-output calculation is retained. The region selector was rejected. The held-out factual comparison was inconclusive. The final separate screen retained 12 candidates, found four full-correct answers preserved by reuse, and established zero qualifying reuse-induced errors. No oracle repair enumeration ran. These outcomes do not establish useful selective-repair headroom or a successful inference method.

The later documentation-only [reproduction assessment](../../docs/reproduction-target.md) is outside this snapshot. Nothing in that review authorizes renewed selector development, factual sweeps, model downloads or cloud jobs.
