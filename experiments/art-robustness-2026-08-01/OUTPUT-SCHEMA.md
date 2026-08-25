# Output schema — ART robustness arm -> curves_point

The ART arm adds RGR-family curves to the `curves_point` dict already produced
by pavia-composite-2026-07-25/pavia_composite_experiment.py, matching the
existing `RGR_tail_swap` / `RGR_tail_swap_p_grid` convention (length-L lists).

New keys per attack family (HopSkipJump shown; FGM logit-only analogous):

    RGR_hopskipjump                       : length-L list  (RGR score per severity knot)
    RGR_hopskipjump_severity_grid_requested : length-L list  (the predeclared grid)
    RGR_hopskipjump_achieved_l2_median    : length-L list  (median projected L2 distortion)
    RGR_hopskipjump_achieved_l2_p10/p90   : length-L lists
    RGR_hopskipjump_validity_rate         : length-L list  (fraction of rows actually attacked+legal)
    RGR_hopskipjump_flip_rate             : length-L list  (fraction of argmax class flips)
    RGR_fgm            (logit only)       : length-L list
    RGR_fgm_eps_grid   (logit only)       : length-L list
    ... same achieved/validity/flip suffixes

The AttackResult.p_perturbed tensor (L, n_rows, n_classes) is stored to the
run's .npz alongside the existing replicate arrays, in the SAME row order as
`p_full`, so the bootstrap/covariance machinery consumes it unchanged.

## Fixed-draw lock (design decision, not implicit)

The attack is generated ONCE per (model[, top-severity]); the severity curve is
read by interpolation+projection. It is NEVER regenerated per bootstrap
replicate. Measured cost forces this: HopSkipJump vs the 300-tree random forest
is ~10 s/row at max_iter=20 (vs ~0.02 s/row for logit) — regenerating per
replicate at B~2000 is computationally infeasible. Bootstrap resamples ROWS
from the frozen p_perturbed tensor, exactly as the tail-swap arm already does.
