# Provenance exemptions

Result files that do not carry the full four fields, with the reason. Each
entry is a decision on the record. Format: `- ` + backticked path + ` — reason`.

- `experiments/delong-2026-07-27/results-delong.json` — derived analysis, not a fresh run; it consumes `replicates-redraw.npz` from `redraw-2026-07-24` and inherits that run's seed and pin. Recorded via `source_replicates`.
