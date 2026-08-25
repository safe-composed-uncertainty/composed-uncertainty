# Vasily's day-1 runbook — run + review, not build

Everything below is already built and smoke-tested (2026-08-01). Your first
day is to run it against the real models and review the results, then answer
the five open decisions.

## 1. Install (~15 s, verified)

    ./env/install_art_env.sh
    . .venv-art/bin/activate

## 2. Fast checks

    python -m unittest -v constraints.test_constraints      # 9/9, no ART/network
    python -m smoke_test.smoke_test_hsj_fgm 15               # real models, ~2-3 min (RF dominates)

Expect: logit ~0.02 s/row, RF ~10 s/row; monotone curves; SMOKE-RESULT.json.

## 3. Full frozen-artifact run  (once the constraint specs below are confirmed)

    python run_art_robustness.py                            # writes results-art-robustness.json + .npz

Reads severity_grids/grids.json and constraints/feature_constraints_*.json;
generates each attack ONCE per (model), reads the curve by
interpolation+projection, and emits the RGR_* keys of OUTPUT-SCHEMA.md for the
existing bootstrap/covariance machinery. Budget: RF ~10 s/row * n_rows, one
time; keep n_rows modest (200-400) or lower max_iter and record it.

## 4. Poisoning pilot — SEPARATE (do not merge)

    cd ../art-poisoning-pilot-2026-08-01                    # own dir, own certificate scope

## Five decisions that are YOURS / the meeting's (kept open on purpose)

1. Is the robustness claim evaluation-time evasion, ingest poisoning, or both
   as separately labelled measurands?  (drives which arms are headline)
2. Which fields are mutable for a realistic attacker in German credit and
   Taiwan?  -> edit constraints/feature_constraints_*.json (currently DRAFT:
   all continuous except {personal_status, age, foreign_worker} immutable for
   German; Taiwan spec is an empty placeholder).
3. Attack success measured against class flips, probability drift, or the RGR
   curve definition?  (flip_rate and probability tensor are both emitted.)
4. Poisoning uncertainty: over one poisoned retrain, or the full training
   procedure?  (schema already supports uncertainty_scope=training_procedure.)
5. Per-family severity grids inside the Compliance Score, with the grid and
   integration weights carried in the certificate?  (grids.json holds them as
   data; the composite does NOT require equal knot counts.)

None of these is pre-decided in code — the harness runs whatever you set.
