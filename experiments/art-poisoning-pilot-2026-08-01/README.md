# Poisoning pilot — SEPARATE from the frozen-artifact arm

Deliberately a sibling of art-robustness-2026-08-01, never merged. Ingest
poisoning changes the estimand: it perturbs TRAINING rows/labels, retrains, and
evaluates the resulting artifact. It therefore needs its own certificate scope
(certificate-profile-0.3 already defines uncertainty_scope=training_procedure,
distinct from fixed_artifact_evaluation — no schema change needed), its own
seeds, its own retrain-per-fraction protocol, and its own uncertainty treatment
(bootstrap or repeat the whole train/poison/fit/evaluate loop, not just the
evaluation).

Primitive: ART PoisoningAttackBackdoor(perturbation) with a declared tabular
trigger and a target-label switch. Fraction grid 0, 0.5%, 1%, 2%, 5%
(severity_grids/grids.json poisoning_fraction).

STATUS: skeleton only; build after the 27.08 meeting settles decision #1
(is poisoning a headline measurand?) and #4 (uncertainty over one retrain vs
the full procedure). Do NOT pool any number from here with the evasion arm.
