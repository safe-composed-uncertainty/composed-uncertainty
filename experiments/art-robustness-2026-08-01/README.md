# Frozen-artifact ART robustness harness (pre-build for mid-August)

Pre-built and smoke-tested 2026-08-01 so the robustness extension is "run +
review" when Vasily starts, not "build". Widens the SAFE robustness metric
(RGR) beyond the two existing perturbations (Gaussian noise, published
tail-swap) with IBM Adversarial Robustness Toolbox evasion attacks, keeping
the existing bootstrap/covariance machinery unchanged.

## Layout

    env/            pinned ART+torch lockfile, install script, live-verified digest
    constraints/    FeatureConstraints projection + unit tests + per-dataset DRAFT specs
    severity_grids/ grids.json (per-family, frozen; equal knot counts NOT required)
    adapters/       base.py (AttackResult + curve metadata), hopskipjump_adapter.py
    smoke_test/     real-model smoke test + first measured run
    run_art_robustness.py   full-run entry point (skeleton; wires adapters -> curves_point)
    OUTPUT-SCHEMA.md, PRE-BUILD-CHECKLIST.md

## Design invariants

- Every adversarial row is legality-projected (clip continuous / snap
  categorical / restore immutable) BEFORE it reaches rgr_score. safeai's own
  rgr_curve(method='adversarial') clips only to a single global (min,max) — it
  does NOT respect per-feature legality, so the adapter wraps it.
- Fixed-draw: attack generated once per (model), never per bootstrap replicate
  (HSJ vs RF ~10 s/row — full-draw at B~2000 is infeasible).
- FGM is logit-only (needs gradients the RF wrapper doesn't expose); HSJ is the
  cross-model arm.
- The poisoning pilot lives in a SEPARATE sibling directory with its own
  certificate scope (uncertainty_scope=training_procedure) and is never pooled
  with the frozen-artifact arm.

Scope note: this is engineering infrastructure. The five methodological
decisions in PRE-BUILD-CHECKLIST.md stay open for Vasily/Paolo and the 27.08
meeting; the harness runs whatever they set.
