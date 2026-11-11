# Execution notes

- The first sandboxed milestone launch stopped before loading or inference because MPS was hidden (`MPS required; no CPU fallback`). GPU commands were subsequently approved outside the sandbox. No CPU fallback was used.
- The server restarted while the first outside-sandbox milestone was running. A process check confirmed the original process remained active; no duplicate was launched. It finished with a preserved raw-tensor gate failure.
- `failed-milestone-1/` preserves the original source, numerical results and launch log. The original unchanged tolerance was not relaxed. See the protocol's numerical collection correction for the refined gate and native-coefficient rotation fix.
- `milestone-rephase.log` and `milestone.json` contain the completed two-configuration milestone, including the raw native prefix tensor failures as diagnostics.
- No model weights were downloaded or changed. Dataset downloads were explicitly confirmed before fetching. No cloud execution occurred.
