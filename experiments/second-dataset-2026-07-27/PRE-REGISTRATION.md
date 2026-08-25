# Pre-registration: the second dataset for the composed-uncertainty article

Item A11 of `UPGRADE-PLAN-2026-07-27.md`.
Written 27 July 2026, before any composite, correlation or interval has been
computed on the second dataset.
Author: Anton Sokolov, Tyche Institute, Tallinn.

This document is written to be timestamped ahead of the result. Everything in
it is a commitment. The four probes that precede it measured availability,
shape, cost and cross-dataset comparability only; not one of them evaluated a
composite, a correlation between metric families, or an interval, and none of
them may be re-run with different settings once the primary run has been seen.

---

## 1. What the article currently claims, and on what evidence

The article's headline mechanism is that the TAIL-SWAP perturbation family
anti-correlates the robustness curve with the accuracy and explainability
curves deep in the severity sweep. The consequence it draws is a contrast:
declaring the cross-terms zero barely moves the volume composite's interval,
while it understates a two-link chain's interval by 23.6 per cent.

That contrast currently rests on one dataset (Statlog German credit, OpenML
`credit-g` v1, n_test = 300, d = 20), two models (a standardised logistic
regression and a 300-tree random forest), one split and one seed. The article
says so and bounds the direction to that dataset.

We add exactly one second dataset. Both outcomes are publishable, and the
purpose of this document is to fix which outcome we will be looking at before
we look.

## 2. The second dataset

UCI *Default of Credit Card Clients* (Taiwan), served by OpenML as data id
42477, `default-of-credit-card-clients` version 1. It is the priority dataset
named in the plan and it proves usable, so no fallback is invoked.

Measured under the pinned split, against German credit for reference:

| | German credit | Taiwan |
|:--|--:|--:|
| n | 1,000 | 30,000 |
| d | 20 | 23 |
| n_train / n_test (stratified 70/30, seed 20260723) | 700 / 300 | 21,000 / 9,000 |
| positive share, whole sample | 0.7000 | 0.2212 |
| positive share, test | 0.7000 | 0.2212 |
| positives in test | 210 | 1,991 |
| exactly duplicated rows | 0 | 56 |
| integer-valued columns | 20 of 20 | 23 of 23 |
| columns with at most 12 distinct levels | 17 of 20 | 9 of 23 |
| non-finite cells | 0 | 0 |
| curve grid GRID = d | 20 | 23 |
| curve length L = d + 1 | 21 | 24 |
| composite tensor cells L^3 | 9,261 | 13,824 |
| tail-swap severity spacing 0.5 / GRID | 0.02500 | 0.02174 |

The dataset loads with the socket layer deliberately disabled, in 0.15 s, from
the local scikit-learn OpenML cache. The study therefore requires no network,
and the reproducibility statement may say so.

Two structural differences are recorded here because they are the leading
candidate explanations if the mechanism does not reproduce, and we will not be
allowed to reach for them after the fact:

1. **Positive class.** The pinned German-credit driver codes y = 1 for "good",
   which is the *majority* class at 70.0 per cent. The Taiwan target is
   default, a *minority* class at 22.1 per cent. RGA and RGR are rank
   functionals, so the coding direction is not obviously innocuous.
2. **Column type mix.** German credit as the pinned driver loads it is 17 of 20
   low-cardinality ordinal-encoded columns. Taiwan is 9 of 23 low-cardinality
   codes and 14 near-continuous money and age columns. The tail swap is a rank
   exchange, so the substrate it acts on differs.

The perturbation's measured bite -- the share of test cells whose value
actually changes, averaged over columns -- is comparable and slightly stronger
on Taiwan: 0.250 / 0.500 / 0.881 at p = 0.125 / 0.25 / 0.5, against
0.234 / 0.441 / 0.702 on German credit. The severity axes are therefore
commensurable, and no rescaling of p is applied.

## 3. What stays identical

The following are inherited verbatim from the pinned drivers
`redraw-2026-07-24/redraw_experiment.py`,
`pavia-composite-2026-07-25/pavia_composite_experiment.py` and
`pavia-composite-2026-07-25/verify_cross_terms.py`. Changing any of them would
make the comparison meaningless, and none of them will be changed.

- Substrate: `koleso500/safeai` pinned at commit
  `39768fcd5264c881f7174268bbffda52b298ae89`, the vendored read-only clone and
  the inert import stubs of the redraw run. Nothing under
  `redraw-2026-07-24/` or `pavia-composite-2026-07-25/` is written to.
- SEED = 20260723 and every stream derived from it: paired bootstrap
  `default_rng(SEED + 1)`; three independent streams `default_rng(SEED + 10 + k)`,
  k = 0, 1, 2; the chain's shared stream `default_rng(SEED + 99)`; the Gaussian
  sweep `default_rng(SEED)`; the scheme-B redraw bases SEED + 100000 (logit) and
  SEED + 200000 (rf), consumed as base + b; the Monte Carlo seeds SEED + 777 and
  SEED + 779.
- Split: stratified, test_size = 0.3.
- Models, untuned: `StandardScaler` + `LogisticRegression(max_iter=2000,
  random_state=SEED)`, and `RandomForestClassifier(n_estimators=300,
  random_state=SEED, n_jobs=-1)`. We do not retune on 21,000 training rows.
  Tuning would confound "different dataset" with "different model" and would
  destroy the only comparison this exercise exists to make.
- B = 2000 paired replicates and 2000 independent-stream replicates.
- alpha = 0.05, z = 1.959963984540054, percentile endpoints at 2.5 and 97.5.
- The tail swap, verbatim: for each column, the outermost floor(p*n) ranks on
  each side exchange values pairwise from the outside in, p in [0, 0.5], on the
  grid p_t = 0.5 t / GRID. The p-range is unchanged; only the number of knots
  on it follows d.
- The fixed Gaussian draw at 0.5 column standard deviations, and the Gaussian
  sweep at sigma_t = t / GRID column standard deviations.
- The composite in all three variants (arithmetic, geometric, root mean square)
  as the mean of the L x L x L tensor; the matched-severity diagonal readings;
  TOPSIS with the hand-built ideal and anti-ideal vectors read at the run's L;
  the weighted sensitivity arm at w = (0.50, 0.20, 0.30), declared as a
  departure from the source authors' stated position exactly as before.
- The chain functional h = C_A * C_B with C the geometric mean of the three
  scalar metrics, link A = logit, link B = rf, first-order propagation, and the
  contrast between the measured-covariance interval and the interval with the
  cross-terms declared zero.
- Both conditioning schemes, in both senses the pinned work uses the phrase:
  (a) the scalar driver's fixed-draw versus per-replicate-redraw perturbation;
  (b) the curve driver's frozen fast path versus full re-search of the greedy
  removal order and rebuild of the perturbation on the resampled rows.
- The six-rung estimator bridge, in the same order and with the same
  definitions.
- `class_order = [0, 1]` where the pinned drivers pass it, and the same
  measurement of what the bare-vector convention does instead.

## 4. What changes, and why

Four changes. Each is forced by the data, and each is declared here rather than
discovered in the results.

1. **L = 24 instead of 21.** The rule L = d + 1 with GRID = d is what the pinned
   study fixed, not the number 21. The RGE greedy curve has exactly one knot per
   feature plus the zero-removal anchor, so L is determined by d and is not free.
   Holding L = 21 on 23 predictors would either stop the greedy removal before
   all features are removed, which changes the estimator, or interpolate knots
   that were never computed. We keep the rule and report the consequence: the
   composite averages 13,824 tensor cells instead of 9,261, and the severity
   grid is finer by 13 per cent. The two length-d truncation readings that the
   pinned study already runs (anchor dropped, terminal knot dropped) carry over
   at L = 23.
2. **n_test = 9,000 instead of 300.** This is the point of the exercise. The
   split rule is unchanged; only the sample it is applied to is larger.
3. **The positive class is a minority.** We keep the UCI dictionary's own
   target, y = 1 for default, because it is what every published Taiwan
   baseline predicts. Because the direction is not obviously innocuous, we
   pre-register a coding sensitivity that runs unconditionally and cheaply: the
   point curves, the curve-summary covariance and one paired pass under
   y' = 1 - y, reported next to the primary. This is registered now precisely so
   that it cannot function as a post-hoc rescue. If the sign of the curve-mean
   RGE-RGR correlation differs between the two codings, we report both and treat
   the finding as coding-dependent rather than selecting the coding that
   resembles German credit.
4. **Nothing else.** B stays at 2000. The conditioning check stays at
   B_full = 300 (logit) and 30 (rf), because its per-replicate cost is
   essentially independent of n -- it is dominated by the number of model calls
   in the greedy re-search, 276 here against 210 on German credit -- and it
   measures 1.1 s and 24.9 s a replicate against 0.09 s and 26.2 s on German
   credit.

One operational note that is not a design change: at B = 2000 and n = 9,000 the
curve driver holds one paired and three independent index streams, 576 MB as
int64. The machine has 125 GB, so the streams stay int64 and the draws stay
bit-identical to the pinned recipe.

## 5. Cost

Measured on zeus1 with `/srv/tyche/repos/safeai-fork/.venv-test/bin/python`
(Python 3.12.3, numpy 2.5.1, scikit-learn 1.9.0), 24 threads, under a light
background load. Two independent timing passes agree within 13 per cent on
every constant.

One full metric-vector evaluation, three curves, no reuse: 1.08 s (logit),
27.36 s (rf). One paired bootstrap replicate on the fast path: 0.0983 s
(logit), 0.0842 s (rf). Projected total for the whole study -- both drivers,
both conditioning schemes, the chain and the estimator bridge -- is 3,348 s,
that is 55.8 minutes serial, plus roughly 7 minutes for the coding sensitivity
of section 4.3. That is inside the ninety-minute bar, so no reduction is
pre-authorised.

If the run nevertheless overruns, exactly one reduction is permitted, and it
must be reported in the article rather than absorbed silently: the rf
conditioning check drops from B_full = 30 to B_full = 10, saving 8.3 minutes.
Its cost in precision is that the standard-deviation ratio it reports acquires
a materially wider Monte Carlo band -- it is already labelled a gross check at
30 -- so a ratio near 1.0 would stop being evidence that freezing the greedy
order is harmless, and we would have to say so. B itself will not be reduced:
halving B to 1000 would save 18.7 minutes and widen the Monte Carlo error on
every correlation and every width share by a factor of sqrt(2), which is
precisely the precision the sign calls in section 7 depend on.

## 6. The quantities that will be compared

These, and only these, are the pre-registered comparisons. German-credit values
are read from the deposited artefacts and are fixed here so that they cannot
drift.

**P1. Correlation of the curve-mean RGE with the curve-mean RGR under tail
swap.** German credit: logit -0.1372, rf -0.3102.

**P2. Correlation of the curve-mean RGA with the curve-mean RGR under tail
swap.** German credit: logit -0.0583, rf -0.0800.

**P3. Delta-method width understatement from declaring the cross-terms zero,
for the three volume variants and TOPSIS.** German credit, logit:
arithmetic -0.91, geometric -1.86, rms +1.26, TOPSIS -1.50 per cent. Random
forest: -3.08, -3.09, -1.50, -3.11 per cent. Every value is small and several
are negative, which is the article's "barely moves it" claim.

**P4. Empirical width understatement from the independent-stream bootstrap, same
arms.** German credit, logit: +0.89, +1.45, -1.41, -3.10 per cent. Random
forest: -6.49, -5.21, -4.86, -7.56 per cent.

**P5. The chain.** Cross-link correlation and the width understatement of the
two-link product. German credit: fixed draw rho = +0.7222 and 23.79 per cent;
redraw rho = +0.7120 and 23.57 per cent. The article quotes 23.6 per cent.

**P6. The estimator bridge, six rungs, RGE-RGR correlation.** German credit:

| rung | construction | logit | rf |
|--:|:--|--:|--:|
| 1 | scalar RGE (bare) x scalar RGR (bare), the deposit pair | +0.5967 | +0.3415 |
| 2 | the same with the class-order convention | +0.6144 | +0.3437 |
| 3 | curve-mean RGE x scalar RGR | +0.4249 | +0.2879 |
| 4 | scalar RGE x curve-mean tail-swap RGR | -0.2083 | -0.1529 |
| 5 | curve-mean RGE x Gaussian-sweep RGR mean | +0.5354 | +0.4711 |
| 6 | curve-mean RGE x curve-mean tail-swap RGR, the study pair | -0.1372 | -0.3102 |

Rungs 4 and 5 are the mechanism. Rung 4 swaps only the RGR estimator to the
tail swap and the sign goes negative; rung 5 keeps the sweep structure and
changes only the perturbation family back to Gaussian and the sign stays
positive. That pair is what attributes the anti-correlation to the tail-swap
family rather than to the curve construction or to the sweep.

**Secondary, reported but not decisive.** Scalar-metric correlations on the same
index streams (German credit logit +0.2858, +0.1596, +0.5967; rf +0.2922,
+0.1530, +0.3415); the deterministic-knot invariance check, which must reproduce
to better than 1e-12; the arithmetic and geometric factorisation identities; the
Euler identity for the rms gradient; the power-mean ordering; the two length-d
truncation readings; the TOPSIS column-normalisation sensitivity.

**Explicitly not compared.** The point values of RGA, RGE, RGR, of the three
composites, of TOPSIS and of the chain product. Different data and separately
fitted models, so no numerical agreement is expected and none will be claimed.

## 7. The three verdicts, fixed in advance

A sign is called only where the absolute value exceeds twice its Monte Carlo
standard error, computed by the second-level resampling blocks the pinned
drivers already carry. Values inside that band are reported as unresolved, not
as zero and not as agreeing.

**The mechanism GENERALISES** if all five hold:

- (a) bridge rung 4 is negative on both models;
- (b) bridge rung 5 is positive on both models, so the sign is attributable to
  the perturbation family and not to the sweep structure;
- (c) bridge rung 6, the study pair, is negative on both models;
- (d) the delta-method width understatement for the three volume variants lies
  within 6 percentage points of zero on both models, so the cross-terms remain
  immaterial for the volume composite; and
- (e) the chain understatement is positive and at least 10 percentage points,
  with a positive cross-link correlation under both conditioning schemes.

What we would then write: the anti-correlation is a property of the tail-swap
family on tabular credit data, reproduced at thirty times the sample size and
three more predictors, and it is not an artefact of n = 300.

**The mechanism FAILS TO GENERALISE** if any one of these holds:

- (a') bridge rung 4 is positive on both models; or
- (b') bridge rung 6, the study pair, is positive on both models; or
- (c') the volume-composite delta understatement exceeds +15 percentage points
  on both models, that is, the cross-terms become material where they were
  immaterial; or
- (d') the chain understatement falls below 5 percentage points, or changes
  sign, on both models.

What we would then write: the sign is dataset-specific. This is not a defeat for
the article; it sharpens its map thesis, because a certificate that must carry
the measured covariance is exactly what a quantity whose sign is unpredictable
from the metric family alone requires.

**The result is INCONCLUSIVE** if the two models disagree in sign on rung 4 or
rung 6, or if the deciding quantity lies inside two Monte Carlo standard errors
of zero, or if the coding sensitivity of section 4.3 flips the sign. We then
report every number, decline the generalisation claim in both directions, and
state that the mechanism is bounded to the conditions under which it was
measured. We do not add a third dataset inside this article to break a tie; that
is the factorial of plan item B4 and it belongs to the sibling article.

## 8. Standing commitments

- No quantity in section 6 is dropped, and no quantity outside section 6 is
  promoted to decisive, after the run.
- If the run must be repeated for a defect, the defect and the repetition are
  logged, and the first run's outputs are kept.
- Every run writes a log next to its results. The frozen-corpus rule of the
  article applies: a number enters the text only after two independent
  recomputations agree.
- The article will state the number of models, splits, seeds and datasets behind
  every sign it interprets, and will keep the existing note that the sign
  discussion is descriptive rather than a multiplicity-corrected test.
