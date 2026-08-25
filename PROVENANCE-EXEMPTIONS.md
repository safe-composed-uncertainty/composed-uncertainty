# Provenance exemptions

`tools/check_provenance.py` requires four fields in every result file: the seed, the pinned
`safeai` commit, the NumPy version and a UTC timestamp. These runs predate the rule. Each file
below is listed with the exact reason, so that a gap is a decision on the record rather than
silence. The check prints these rather than hiding them.

⛔ **None of these were repaired by writing in a value.** A seed can be recovered from a run
script; a library version and a timestamp are facts about a run that already happened and cannot
be recovered afterwards. Inventing them would make the record worse, not better.

## Not applicable — the run does not depend on `safeai`

- `experiments/coverage-2026-07-27/results-coverage.json` — a coverage simulation against analytic truth. Its imports are numpy and scipy only; it never calls `safeai`, so there is no pin to record. The seed is present as `master_seed` and the NumPy version as `numpy_version`. The missing timestamp is a genuine omission.

## Provenance inherited from an upstream artifact

- `experiments/delong-2026-07-27/results-delong.json` — a derived cross-check rather than a fresh run. It consumes `replicates-redraw.npz` produced by `redraw-2026-07-24` and inherits that run's seed and pin, recorded through its `source_replicates` field.

## Recorded, but in the run summary rather than the result file

- `experiments/aggregation-2026-07-25/results-aggregation.json` — the NumPy version is recorded as "numpy 2.5.0" in `AGGREGATION-SUMMARY.md` in the same directory, together with Python 3.12.3 and scikit-learn 1.9.0. The value is documented; it is simply not in the JSON.

## Genuine omissions, not recoverable

These are gaps. They are listed rather than quietly passed.

- `experiments/curve-composite-2026-07-25/results-step1-vectors.json` — NumPy version not recorded anywhere in the run directory.
- `experiments/redraw-2026-07-24/results-redraw.json` — NumPy version and timestamp not recorded anywhere in the run directory.
- `experiments/real-agentic-2026-07-25/results-real-agentic.json` — timestamp not recorded. A date does appear in `build_public_release.py`, but it is a hard-coded constant in a script rather than the moment the run happened, so it is not used here.
- `experiments/real-agentic-2026-07-25/pilot/results-pilot.json` — timestamp not recorded.
