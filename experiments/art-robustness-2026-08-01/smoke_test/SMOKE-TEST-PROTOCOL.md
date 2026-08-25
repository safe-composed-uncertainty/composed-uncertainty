# Smoke-test protocol and first measured run

Command (first thing to run):

    . venv-art/bin/activate
    cd papers/safe-composed-uncertainty/experiment/art-robustness-2026-08-01
    python -m smoke_test.smoke_test_hsj_fgm 15

Fits the exact deposited German-credit logit + 300-tree RF (SEED 20260723,
test_size 0.3, StandardScaler on logit), builds data-fitted continuous
constraints, runs HopSkipJump (max_iter=20) on a held-out sample, projects to
the legal manifold, prints per-knot achieved L2 / validity / flip rate.

## First measured run (2026-08-01, 12 held-out rows), SMOKE-RESULT-2026-08-01.json

- logit: HSJ generate 0.26 s total, 0.022 s/row; at s=1.0 l2_med 0.514,
  validity 0.25, flip 0.25.
- rf: HSJ generate 122 s total, 10.2 s/row; at s=1.0 l2_med 0.613,
  validity 0.42, flip 0.42.

Curves are monotone and sane (distortion and flip rate grow with severity;
s=0 is the identity). The ~460x logit-vs-RF cost asymmetry is REAL and is why
the fixed-draw lock (attack once per model, not per bootstrap replicate) is a
hard design decision, not a convenience.

## Full-run budget (from measured per-row cost)

- RF arm dominates: ~10 s/row * n_rows * (attack generated once per model).
  200 held-out rows ~= 34 min; 1000 rows ~= 2.8 h, ONE TIME, then all bootstrap
  replicates resample rows from the frozen tensor for free.
- Reduce with a smaller held-out attack sample or lower max_iter; record both
  in the run manifest.
