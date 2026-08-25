# Aggregation: one interval construction, several composites

Study of 25 July 2026. Anton Sokolov, Tyche Institute, Tallinn.
Companion to the composed-uncertainty experiment; feeds Section 5 ("Aggregation") of the
outline of 24 July.

Every statistical number below comes from the run recorded in `results-aggregation.json` and
`run-aggregation.log`, produced by `aggregation_experiment.py` in this directory. Nothing is
quoted from memory or from an earlier note. The only figures not drawn from that run are the two
comparative wall-clock timings in Section 10, which come from reruns whose log the run of record
has since overwritten and which are labelled as such where they appear.

If you read one thing, read Table 4.0: seven composites for the same two systems, with the point
value, the interval width, and what declaring the cross-terms zero costs each of them. Section 5
reads that table, bounds the effect over the whole weight simplex, and states the mechanism.

## 1. Why this study exists

Section 5 of the agreed outline promises that a weighted sum, a TOPSIS composite and a
Shapley-based composite each get an interval by the same construction, and that the intervals
differ. Until now the deposited experiment built that interval for one composite only, the
geometric mean. This study builds the layer for a family of composites, so that whichever
composite the Pavia integrated-metrics line actually uses, only the composite function has to be
swapped and the interval machinery is already in place.

The immediate reason for doing it now is that Pavia's composite is not available. The public
`safeai` package at the pinned commit ships no aggregation function of any kind: we enumerated
the public API and it is `rga_score` / `rge_score` / `rgr_score` with their curve and comparison
variants, plus Cramer and utility helpers, and the only aggregation anywhere in the package is
the per-metric class-weight aggregation used for multiclass problems. The integrated numbers in
the Giudici-Kolesnikov article therefore came from code that is not in the public repository.
That is an open question in the outline, and it stays open here. We do not guess at Pavia's
composite; we make the guess unnecessary.

## 2. What was run

Substrate and data are inherited unchanged from the deposited runs, so that every number here is
commensurable with the published ones.

- Substrate: `safeai` at pinned commit `39768fcd5264c881f7174268bbffda52b298ae89`, tabular code
  paths only, reusing the vendored clone and the inert import stubs of the 24 July redraw run.
  This study never clones, never writes into the substrate, and never touches the two protected
  experiment directories.
- Data: Statlog German credit (OpenML `credit-g` v1), 700 train / 300 test, d = 20, seed
  20260723, stratified split.
- Models: a standardised logistic regression and a 300-tree random forest, fixed after training.
- Replicates: B = 2000 paired bootstrap replicates of the metric vector (RGA, RGE, RGR),
  estimated on one shared test sample, one row resample per replicate, the whole vector
  recomputed on that same resample. Both conditioning schemes are run: the fixed perturbation
  draw of the deposited v0.1 experiment, and the per-replicate redraw of the 24 July rerun. The
  redraw quantities are the primary ones, as in v0.2.
- Also regenerated: the independent-stream ("wrong-way") replicates, one index stream per metric,
  which are the assumption-light empirical counterpart of declaring the cross-terms zero.

The deposited runs persist no replicate matrix -- neither `results.json` nor
`results-redraw.json` stores the B x 3 array, and no side file holds it -- so this study
regenerates the replicates from data and seeds. Section 6 shows the regeneration is exact.

The interval construction, identical for every composite, which is the whole point:

1. Push each replicate vector through the composite and take the percentile interval.
2. Linearise the composite at the point estimate and form the delta interval with
   `var = grad' Sigma grad`.
3. Form the same delta interval with the cross-terms declared zero,
   `var0 = sum_k grad_k^2 Sigma_kk`, and report the gap.

A notation warning for whoever splices this into the article. `Sigma` here is the empirical
covariance of the B replicate vectors, which is an estimate of the covariance of the estimator
`theta_hat` itself -- the object Section 3 of v0.2 writes as `Sigma/n`, not the one it writes as
`Sigma`. The deposited scripts use the same loose name (`Sigma_paired`), and the JSON keys of this
study follow them for continuity. Nothing downstream depends on the distinction, because every
formula here consumes the same matrix, but Section 5 must not be written as if `Sigma` meant two
things in two sections.

Also recorded for each composite and scheme, and used in Section 5: the pairwise decomposition of
the cross-covariance contribution, `2 g_k g_l Sigma_kl` for each of the three pairs, and the share
of the gradient in absolute value sitting on each metric. Without these the per-composite share is
a bare number and the ordering across composites cannot be checked.

Conventions follow the deposited chain block exactly, so the percentages are comparable with the
23.6% quoted in the article: understatement of width is `100 * (1 - width_declared_zero /
width_measured)`, with the wider, honest width in the denominator. We report the variance share
dropped separately, `100 * (1 - var_declared_zero / var_measured)`, because width goes as the
square root and the two numbers are not interchangeable. The article quotes the variance share at
the single-system level (6.9% and 11.4%) and the width understatement at the chain level (23.6%);
reporting both, per composite, removes the ambiguity.

The two-sided normal quantile is the same hardcoded 1.959963984540054 the deposited runs use.

## 3. The composites, stated plainly

| Arm | Definition | Status |
|:--|:--|:--|
| `wsum_equal` | weighted sum, w = (1/3, 1/3, 1/3) | linear; delta method exact |
| `wsum_accuracy_led` | weighted sum, w = (0.50, 0.20, 0.30) | declared unequal profile, not a recommendation |
| `wsum_accuracy_light` | weighted sum, w = (0.20, 0.40, 0.40) | second unequal profile; downweights the variance-dominant component |
| `geomean` | (RGA x RGE x RGR)^(1/3) | the deposited composite; continuity case |
| `topsis` | ideal fixed at (1,1,1), anti-ideal at (0,0,0), Euclidean distance, closeness coefficient d-/(d- + d+) | see caveat below |
| `shapley_min` | v(S) = min over S, v(empty) = 0; composite = sum of the three Shapley values | one defensible instantiation, not canonical |
| `shapley_prod` | v(S) = product over S, v(empty) = 0; composite = sum of the three Shapley values | smooth counterpart of `shapley_min` |

**TOPSIS instantiation.** The ideal point is fixed at 1 and the anti-ideal at 0 in every
coordinate, which is available to us because the rank-graduation metrics are already normalised
to [0,1]. Distances are Euclidean and the composite is the closeness coefficient. A
data-dependent ideal point -- the usual TOPSIS practice, where the ideal is the best value
attained by any alternative -- would make one model's composite depend on the other models in the
comparison set, and an interval for it would need the full joint covariance across models, not
just across metrics. That is a different and larger object and it is out of scope here. We say so
rather than quietly using the fixed-ideal version and calling it TOPSIS.

**Shapley instantiation, and an honest caveat about it.** We take a characteristic function over
subsets of the three metrics, compute the exact Shapley value of each metric, and sum them. This
is one defensible instantiation and we do not present it as canonical. The Shapley-Lorenz line of
the Pavia group is a different construction, and this arm is a placeholder until Pavia's actual
composite is known.

There is a structural point that follows immediately and that we would rather state than have a
referee state for us. The Shapley values are efficient: they sum to the grand-coalition value
v(N) by construction. So a composite defined as "the sum of the Shapley values" **is** v(N),
whatever the characteristic function. With v(S) = min over S the composite is exactly the minimum
of the three metrics; with v(S) = product over S it is exactly their product. The Shapley step
reallocates credit among the metrics without changing the total. We verified the identity
numerically rather than asserting it: over all 2000 replicates the largest discrepancy between
the summed Shapley values and v(N) is 2.220e-16 for both games and both models.

The consequence is that the interesting Shapley object is the attribution vector, not the sum.
At the point estimate, for the min game:

| Model | phi(RGA) | phi(RGE) | phi(RGR) | shares |
|:--|:--|:--|:--|:--|
| logistic | 0.16318 | 0.32984 | 0.31220 | 20.27% / 40.96% / 38.77% |
| forest | 0.20802 | 0.32923 | 0.28996 | 25.15% / 39.80% / 35.05% |

## 4. Interval tables

### 4.0 The whole study in one table

Redraw scheme, the primary one. `C` is the point value of the composite; `width` is the width of
the paired-bootstrap percentile interval; `var%` is the share of the variance of the composite
that is thrown away by declaring the cross-terms zero; `under%` is the resulting understatement of
the interval width. `g(RGA)` is the share of the composite's gradient, in absolute value, sitting
on RGA -- the column that makes the ordering readable. Rows are sorted by it.

| Composite | g(RGA) log | C log | width log | var% log | under% log | g(RGA) rf | C rf | width rf | var% rf | under% rf |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `shapley_min` | 100.0% | 0.8052 | 0.11580 | 0.00 | 0.00 | 100.0% | 0.8272 | 0.10899 | 0.00 | 0.00 |
| `topsis` | 70.5% | 0.8883 | 0.05989 | 1.97 | 0.99 | 58.4% | 0.8896 | 0.05536 | 6.87 | 3.49 |
| `wsum_accuracy_led` | 50.0% | 0.8868 | 0.05892 | 4.23 | 2.14 | 50.0% | 0.8839 | 0.05794 | 7.72 | 3.94 |
| `geomean` | 37.6% | 0.9127 | 0.04556 | 6.33 | 3.22 | 36.4% | 0.9056 | 0.04608 | 10.95 | 5.63 |
| `shapley_prod` | 37.6% | 0.7603 | 0.11363 | 6.33 | 3.22 | 36.4% | 0.7428 | 0.11365 | 10.95 | 5.63 |
| `wsum_equal` | 33.3% | 0.9163 | 0.04055 | 7.46 | 3.80 | 33.3% | 0.9080 | 0.04255 | 11.78 | 6.07 |
| `wsum_accuracy_light` | 20.0% | 0.9385 | 0.02701 | 13.02 | 6.74 | 20.0% | 0.9242 | 0.03281 | 15.41 | 8.03 |

Three readings, in the order a sceptic would take them. The point value moves from 0.7603 to
0.9385 on the same system, so the composite is a choice, not a measurement. The interval width
moves by a factor of 4.3, so the confidence statement is a choice too. And the cost of declaring
the cross-terms zero moves from 0.00% to 15.41% of the variance, monotonically in `g(RGA)`, so the
question this article is about has to be answered for the composite that will actually be
reported, not once for the system. Sections 4.1 to 4.7 give the intervals themselves and the
fixed-draw rows; Section 5 reads the ordering.

`geomean` and `shapley_prod` share their `var%` and `under%` columns exactly, and that is
structural rather than a copying error: the product is the cube of the geometric mean, so the two
gradients are proportional, and both diagnostics are ratios that are invariant to a positive
rescaling of the gradient. The two arms are one composite up to a monotone transform, which is
also why the study claims four structurally different composites and not seven.

### The per-composite tables

Rows are model x conditioning scheme. `CI measured` is the delta interval with the full
covariance; `CI zero` is the same with the cross-terms declared zero. `under%` is the width
understatement; `var%` is the share of the variance that declaring the cross-terms zero throws
away. `boot CI` is the paired-bootstrap percentile interval, reported alongside as the
assumption-light alternative.

### 4.1 `wsum_equal`, w = (1/3, 1/3, 1/3)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.9163 | [0.8958, 0.9368] | [0.8967, 0.9360] | 0.04101 | 0.03929 | 4.19 | 8.20 | [0.8945, 0.9357] |
| logistic / redraw | 0.9163 | [0.8958, 0.9369] | [0.8966, 0.9361] | 0.04108 | 0.03952 | 3.80 | 7.46 | [0.8957, 0.9363] |
| forest / fixed draw | 0.9080 | [0.8867, 0.9294] | [0.8880, 0.9280] | 0.04271 | 0.03999 | 6.37 | 12.33 | [0.8850, 0.9276] |
| forest / redraw | 0.9080 | [0.8865, 0.9295] | [0.8878, 0.9282] | 0.04300 | 0.04039 | 6.07 | 11.78 | [0.8883, 0.9308] |

### 4.2 `wsum_accuracy_led`, w = (0.50, 0.20, 0.30)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.8868 | [0.8570, 0.9165] | [0.8577, 0.9158] | 0.05951 | 0.05812 | 2.34 | 4.63 | [0.8548, 0.9149] |
| logistic / redraw | 0.8868 | [0.8570, 0.9165] | [0.8577, 0.9159] | 0.05951 | 0.05824 | 2.14 | 4.23 | [0.8564, 0.9153] |
| forest / fixed draw | 0.8839 | [0.8544, 0.9134] | [0.8556, 0.9122] | 0.05902 | 0.05659 | 4.10 | 8.04 | [0.8519, 0.9111] |
| forest / redraw | 0.8839 | [0.8543, 0.9135] | [0.8555, 0.9123] | 0.05915 | 0.05682 | 3.94 | 7.72 | [0.8556, 0.9136] |

### 4.3 `wsum_accuracy_light`, w = (0.20, 0.40, 0.40)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.9385 | [0.9250, 0.9521] | [0.9260, 0.9511] | 0.02711 | 0.02504 | 7.63 | 14.68 | [0.9241, 0.9512] |
| logistic / redraw | 0.9385 | [0.9248, 0.9522] | [0.9258, 0.9513] | 0.02739 | 0.02555 | 6.74 | 13.02 | [0.9254, 0.9524] |
| forest / fixed draw | 0.9242 | [0.9080, 0.9403] | [0.9094, 0.9389] | 0.03230 | 0.02952 | 8.62 | 16.49 | [0.9067, 0.9389] |
| forest / redraw | 0.9242 | [0.9077, 0.9407] | [0.9090, 0.9393] | 0.03294 | 0.03029 | 8.03 | 15.41 | [0.9103, 0.9431] |

### 4.4 `geomean` (the deposited composite)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.9127 | [0.8898, 0.9356] | [0.8906, 0.9348] | 0.04588 | 0.04426 | 3.53 | 6.94 | [0.8877, 0.9338] |
| logistic / redraw | 0.9127 | [0.8897, 0.9357] | [0.8905, 0.9349] | 0.04592 | 0.04444 | 3.22 | 6.33 | [0.8889, 0.9345] |
| forest / fixed draw | 0.9056 | [0.8827, 0.9286] | [0.8841, 0.9272] | 0.04583 | 0.04313 | 5.90 | 11.44 | [0.8805, 0.9263] |
| forest / redraw | 0.9056 | [0.8826, 0.9287] | [0.8839, 0.9274] | 0.04609 | 0.04350 | 5.63 | 10.95 | [0.8836, 0.9297] |

### 4.5 `topsis` (fixed ideal at 1, anti-ideal at 0)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.8883 | [0.8583, 0.9182] | [0.8586, 0.9179] | 0.05992 | 0.05927 | 1.09 | 2.16 | [0.8568, 0.9169] |
| logistic / redraw | 0.8883 | [0.8583, 0.9182] | [0.8586, 0.9179] | 0.05989 | 0.05930 | 0.99 | 1.97 | [0.8573, 0.9172] |
| forest / fixed draw | 0.8896 | [0.8618, 0.9175] | [0.8628, 0.9165] | 0.05570 | 0.05367 | 3.65 | 7.17 | [0.8591, 0.9148] |
| forest / redraw | 0.8896 | [0.8617, 0.9175] | [0.8627, 0.9166] | 0.05581 | 0.05386 | 3.49 | 6.87 | [0.8620, 0.9174] |

### 4.6 `shapley_min` (min game; composite = min of the three metrics)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.8052 | [0.7476, 0.8629] | [0.7476, 0.8629] | 0.11532 | 0.11532 | 0.00 | 0.00 | [0.7443, 0.8601] |
| logistic / redraw | 0.8052 | [0.7476, 0.8629] | [0.7476, 0.8629] | 0.11532 | 0.11532 | 0.00 | 0.00 | [0.7443, 0.8601] |
| forest / fixed draw | 0.8272 | [0.7726, 0.8818] | [0.7726, 0.8818] | 0.10921 | 0.10921 | 0.00 | 0.00 | [0.7697, 0.8787] |
| forest / redraw | 0.8272 | [0.7726, 0.8818] | [0.7726, 0.8818] | 0.10921 | 0.10921 | 0.00 | 0.00 | [0.7697, 0.8787] |

### 4.7 `shapley_prod` (product game; composite = product of the three metrics)

| Model / scheme | C | CI measured | CI zero | w meas | w zero | under% | var% | boot CI |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| logistic / fixed draw | 0.7603 | [0.7030, 0.8176] | [0.7050, 0.8156] | 0.11466 | 0.11061 | 3.53 | 6.94 | [0.6995, 0.8141] |
| logistic / redraw | 0.7603 | [0.7029, 0.8177] | [0.7048, 0.8158] | 0.11476 | 0.11107 | 3.22 | 6.33 | [0.7025, 0.8161] |
| forest / fixed draw | 0.7428 | [0.6864, 0.7992] | [0.6898, 0.7959] | 0.11277 | 0.10612 | 5.90 | 11.44 | [0.6825, 0.7948] |
| forest / redraw | 0.7428 | [0.6861, 0.7995] | [0.6893, 0.7963] | 0.11341 | 0.10702 | 5.63 | 10.95 | [0.6899, 0.8036] |

## 5. What the tables say

**The intervals differ, and by a lot.** Taking the logistic model under the redraw scheme, the
paired-bootstrap intervals are [0.8957, 0.9363] for the equal-weight sum, [0.8889, 0.9345] for
the geometric mean, [0.8573, 0.9172] for TOPSIS, [0.7443, 0.8601] for the min composite and
[0.7025, 0.8161] for the product composite. The widths run from 0.02701 (`wsum_accuracy_light`)
to 0.11580 (`shapley_min`), a factor of 4.3, and the point estimates run from 0.7603 to 0.9385.
A composed figure is not a property of the system alone; it is a property of the system and the
chosen composite, and so is its interval.

**The understatement is a property of the aggregator, not only of the data.** On the same
covariance matrix, declaring the cross-terms zero throws away between 1.97% and 15.41% of the
variance of the composite (logistic and forest, redraw scheme, excluding the min composite, which
reads exactly 0.00% for the structural reason given below), corresponding to width
understatements between 0.99% and 8.03%.

The ordering is intelligible, and the mechanism is worth stating exactly, because the obvious
reading of it is wrong. RGA dominates the variance on this dataset: under the redraw scheme its
bootstrap standard error is 0.02942 against 0.00095 for RGE and 0.00694 for RGR on the logistic
model, and 0.02786 against 0.00096 and 0.01335 on the forest. The scheme has to be named here
because it is the only place where it moves a quoted number: RGA and RGE are deterministic in the
resampled rows and their standard errors are identical under both schemes, while RGR picks up the
perturbation redraw and reads 0.00614 and 0.01263 under the fixed draw. A composite whose gradient
sits mostly on RGA is therefore
close to a one-metric functional and its variance is nearly all diagonal; a composite that
downweights RGA has a much smaller total variance while its cross-terms shrink far less, so the
cross-terms take a larger share. Between `wsum_equal` and `wsum_accuracy_light` on the logistic
model, the total variance falls by 55.5% (1.0982e-04 to 4.8831e-05) while the cross contribution
falls by only 22.4% (8.1981e-06 to 6.3581e-06); that ratio, not any change in the correlations, is
the whole effect. Ordered by the share of the gradient sitting on RGA, the cross-covariance share
is monotone decreasing across all seven arms and both models -- 100.0% of the gradient on RGA
gives 0.00%, 70.5% gives 1.97%, 50.0% gives 4.23%, 37.6% gives 6.33%, 33.3% gives 7.46%, 20.0%
gives 13.02% (logistic; the forest column of Table 4.0 has the same shape).

What is *not* true is that the RGE-RGR block drives this, even though its correlation is the
largest of the three (+0.430 logistic, +0.329 forest, redraw). The run now records the pairwise
decomposition of the cross contribution, and in every arm and both models it is the RGA-with-RGR
pair that carries it: 65.5% to 87.3% of the cross total on the logistic model and 76.0% to 95.1%
on the forest, against 1.2% to 14.3% for RGE-with-RGR. The largest correlation belongs to the pair
with the two smallest standard errors, and what enters the variance is the covariance, the
correlation multiplied by both standard deviations. Reading the correlation matrix alone would
have pointed at the wrong pair.

**How much can the weights move the answer.** Three hand-picked weight profiles do not bound the
effect, and a referee would ask for the bound rather than the three points. A weighted sum is
linear, so the diagnostic over the entire weight simplex is a closed-form scan on the same
covariance matrix, with no further resampling: 1326 weight vectors at a grid step of 0.02.

| Weight profile | logistic var% | logistic under% | forest var% | forest under% |
|:--|--:|--:|--:|--:|
| minimum over the closed simplex, at a vertex w = (0, 0, 1) | 0.00 | 0.00 | 0.00 | 0.00 |
| minimum with every weight at least 0.10, at w = (0.80, 0.10, 0.10) | 0.98 | 0.49 | 1.89 | 0.95 |
| `wsum_accuracy_led`, w = (0.50, 0.20, 0.30) | 4.23 | 2.14 | 7.72 | 3.94 |
| `wsum_equal`, w = (1/3, 1/3, 1/3) | 7.46 | 3.80 | 11.78 | 6.07 |
| `wsum_accuracy_light`, w = (0.20, 0.40, 0.40) | 13.02 | 6.74 | 15.41 | 8.03 |
| maximum with every weight at least 0.10 | 21.35 | 11.31 | 23.09 | 12.30 |
| maximum over the closed simplex | 36.67 | 20.42 | 33.59 | 18.51 |

The maxima with every weight at least 0.10 are attained at w = (0.10, 0.66, 0.24) for the logistic
model and w = (0.10, 0.76, 0.14) for the forest; over the closed simplex, at w = (0.02, 0.88,
0.10) and w = (0.02, 0.92, 0.06). The pattern is the mechanism above taken to its limit: loading
the weight onto RGE, whose own variance is negligible, leaves a composite whose variance is
mostly cross-terms. The number to carry into the article is that on this covariance a weighted sum
can lose up to a fifth of its interval width to the independence assumption, against the 3.8% that
equal weights happen to lose. The 6.9% and 11.4% of v0.2 are not a ceiling; they are one point on
this surface, and which point depends on a weighting decision that is made outside the statistics.

This is the cleanest available demonstration that the cross-covariance question is not answered
once for a system: it has to be answered for the composite that will actually be reported.

**The min composite is a special case worth its own sentence.** Its gradient at the point
estimate is one-hot -- the active coordinate is RGA in all four rows -- so the quadratic form
`grad' Sigma grad` annihilates every off-diagonal entry and the analytic cross-term diagnostic
reads exactly 0.00%. The dependence between the metrics has not gone away; the linearisation
simply cannot see it. Only the bootstrap can, and Section 8 shows what happens when the active
coordinate is genuinely in doubt. Any reader who takes "cross-term share = 0" from a min-type
composite as evidence that the metrics are independent will be wrong, and we say so explicitly.

**The empirical zero-cross-term construction is noisier than the analytic one.** The
independent-stream bootstrap, which gives each metric its own resample stream, produces width
understatements of 2.91% (`wsum_equal`), 1.77% (`geomean`), -0.85% (`topsis`) and -1.62%
(`shapley_min`) for the logistic model, against analytic values of 4.19%, 3.53%, 1.09% and 0.00%.
On the three composites with a nonzero analytic value, one comes out with the wrong sign
(`topsis`, -0.85% against +1.09%). The `shapley_min` row is not a sign disagreement at all: its
analytic value is 0.00%, so there is no sign to disagree with, and the -1.62% is a reading around
a true zero -- which is exactly what makes it useful, as the next paragraph explains. For the
forest the effect is larger and no composite with a nonzero analytic value disagrees in sign:
2.96%, 3.04%, 2.35% and 0.40% against 6.37%, 5.90%, 3.65% and 0.00%.

The min composite calibrates the noise floor directly, which is why we can be specific about the
cause. At the deposited operating point that composite is exactly RGA, so the two constructions
differ only in which resample stream RGA is drawn from and the true effect is exactly zero; the
-1.62% and +0.40% it nevertheless reports are pure Monte Carlo error on percentile endpoints at
B = 2000. On the logistic model every empirical-minus-analytic gap is of that order -- 1.28, 1.76,
1.94 and 1.62 points -- so noise accounts for the wrong signs there. On the forest the gaps are
3.41, 2.86, 1.30 and -0.40 points, and the two largest are not explained by the noise floor alone.
That is not a defect: the two columns do not estimate the same object. The analytic column is a
width ratio between two delta intervals built on one linearisation, while the empirical column is
a ratio between two percentile widths whose replicate distributions have different shapes, since
composing independently resampled metrics changes the skew as well as the variance. The empirical
construction is the assumption-light check on the sign and rough size of the effect, not a second
estimate of the same number, and it should be reported as such.

## 6. Continuity check against the deposited numbers

The geometric-mean arm is the continuity case. It reproduces the deposited numbers exactly. Every
residual below is a maximum absolute difference against `../results.json` (fixed draw, the v0.1
deposit, DOI 10.5281/zenodo.21516475) or `../redraw-2026-07-24/results-redraw.json` (redraw, the
figures behind v0.2).

| Quantity compared | logistic | forest |
|:--|:--|:--|
| point vector (RGA, RGE, RGR) | 0.0 | 0.0 |
| C_geomean | 0.0 | 0.0 |
| Sigma, fixed draw | 0.0 | 0.0 |
| Sigma, redraw | 0.0 | 0.0 |
| paired percentile CI for C | 0.0 | 0.0 |
| independent-stream percentile CI for C | 0.0 | 0.0 |
| delta CI for C | 0.0 | 0.0 |
| sd of C, paired | 0.0 | 0.0 |
| sd of C, independent | 0.0 | 0.0 |
| redraw percentile CI for C | 0.0 | 0.0 |

Not "agrees to four decimals": identical, to the last bit, on every quantity compared. The
regeneration recipe therefore reproduces both deposits, and the aggregation layer sits on exactly
the replicates the published numbers were computed from.

Two further reconciliations are worth recording because they were not reproducible from any
deposited script before this run. The cross-covariance share of the variance of the geometric
mean under the fixed draw comes out at 6.9357% (logistic) and 11.4429% (forest), which is the
6.9% and 11.4% quoted in v0.2; those figures had been computed off-script and no deposited code
recomputed them. The corresponding width understatements, which v0.2 does not quote at the
single-system level, are 3.53% and 5.90% -- markedly smaller than the variance shares, because
width goes as the square root. The chain-level 23.6% in the article is a width understatement and
the single-system 6.9% and 11.4% are variance shares; they are not the same kind of number and
this study reports both for every composite so that nobody has to reconstruct which is which.

One documentation defect surfaced along the way and should be fixed when the deposited script is
next touched. Its module docstring says the delta method uses a "numerical gradient of the
aggregator"; the code uses the closed form `dC/dtheta_k = C / (3 theta_k)`, and no numerical
differentiation appears anywhere in the file. The gradient is right and the numbers are right;
only the description is wrong. This study uses analytic gradients throughout and computes a
central-difference gradient alongside as a check: the largest disagreement over all seven
composites and both models is 6.8e-11.

## 7. Delta versus bootstrap in the linear case

For a linear composite the delta method is exact, and it should be exact in the strong sense that
`grad' Sigma grad` and the sample variance of the pushed-through replicates are the same number,
since Sigma is the sample covariance of those replicates. They are. Over the three weighted-sum
arms, both models and both schemes, the largest relative difference between the two variances is
5.7e-16, which is floating point and nothing else.

| Model / arm / scheme | var (delta) | var (replicates) | rel. diff | max endpoint gap |
|:--|:--|:--|:--|:--|
| logistic / `wsum_equal` / fixed draw | 1.094367e-04 | 1.094367e-04 | 1.238e-16 | 0.00135 |
| logistic / `wsum_equal` / redraw | 1.098209e-04 | 1.098209e-04 | 0.000e+00 | 0.00061 |
| logistic / `wsum_accuracy_led` / fixed draw | 2.304963e-04 | 2.304963e-04 | 1.176e-16 | 0.00220 |
| logistic / `wsum_accuracy_led` / redraw | 2.305070e-04 | 2.305070e-04 | 3.528e-16 | 0.00124 |
| logistic / `wsum_accuracy_light` / fixed draw | 4.782505e-05 | 4.782505e-05 | 5.668e-16 | 0.00092 |
| logistic / `wsum_accuracy_light` / redraw | 4.883116e-05 | 4.883116e-05 | 2.775e-16 | 0.00052 |
| forest / `wsum_equal` / fixed draw | 1.186954e-04 | 1.186954e-04 | 0.000e+00 | 0.00181 |
| forest / `wsum_equal` / redraw | 1.203254e-04 | 1.203254e-04 | 1.126e-16 | 0.00174 |
| forest / `wsum_accuracy_led` / fixed draw | 2.266594e-04 | 2.266594e-04 | 1.196e-16 | 0.00250 |
| forest / `wsum_accuracy_led` / redraw | 2.277173e-04 | 2.277173e-04 | 1.190e-16 | 0.00129 |
| forest / `wsum_accuracy_light` / fixed draw | 6.790562e-05 | 6.790562e-05 | 1.996e-16 | 0.00140 |
| forest / `wsum_accuracy_light` / redraw | 7.059707e-05 | 7.059707e-05 | 1.920e-16 | 0.00260 |

The two variance columns are printed to seven significant figures and agree in every digit shown;
`results-aggregation.json` records them to full precision under `linear_exactness_check`, where
the relative difference is the informative quantity.

The last column is the honest qualification. The *variances* agree to machine precision, but the
delta *interval* and the percentile *interval* still differ, by up to 0.0026 at an endpoint, and
this is not a defect in either. The delta interval is symmetric around the point estimate by
construction; the replicate distribution is not symmetric, because the components sit close to
their upper bound of 1. The gap is the skew, and it is the same effect v0.2 already documents for
the geometric mean, where the naive delta interval misses the percentile bounds by about 0.002
and the logit-scale and BCa repairs close it. Exactness of the delta method as a *variance*
calculation does not make the resulting *interval* right for a bounded composite. That
distinction is worth carrying into the article.

## 8. Non-smoothness: where the delta method breaks

The outline anticipates that "bootstrap is preferred at TOPSIS ties and Shapley kinks". Two
things need to be said about that, one negative and one positive, and both are demonstrated
rather than asserted.

**TOPSIS as instantiated here has no tie problem.** With the ideal fixed at 1 and the anti-ideal
at 0, the closeness coefficient contains no comparison between alternatives and no rank
operation. It does have two singular points, and both have to be cleared rather than one. The
closeness coefficient is `d_minus / (d_minus + d_plus)` with `d_minus = ||v||` and
`d_plus = ||v - 1||`; a Euclidean norm has a cone point where its argument vanishes, so the
composite is non-differentiable at the anti-ideal (0,0,0) *and* at the ideal (1,1,1), and
differentiable everywhere else. On metrics bounded above by 1 and sitting near that bound it is
the ideal, not the anti-ideal, that the replicate cloud approaches. Both margins are recorded in
`replicate_diagnostics`: over the six replicate matrices the closest approach to the ideal is
0.1109 and to the anti-ideal is 1.5087, and the largest single component observed anywhere is
0.9925, so no replicate reaches either point. The binding margin is thus the one to the ideal,
and it is a factor of 13.6 smaller than the margin to the anti-ideal; quoting only the
distance to the origin would have left the nearer singularity unmeasured. Having cleared both,
the delta and bootstrap standard deviations for the TOPSIS arm agree to three decimals in all
four rows (for instance 0.01529 against 0.01524, logistic / fixed draw). The kink the outline
anticipates belongs to the data-dependent-ideal version, which is out of scope for the reason
given in Section 3. We state this rather than implying a tie problem the chosen instantiation
does not have.

This needs to be carried back into the article text, not only into the outline. Section 7 of v0.2
lists "non-smooth aggregators (TOPSIS at rank ties)" among the limitations, without the
qualification. On the fixed-ideal instantiation there are no rank ties to have, so the sentence is
true only of the data-dependent-ideal version and should say which version it is talking about
when it is carried into Section 5 of the joint article.

**The min-type Shapley composite does have a kink, and at the deposited operating point it is
frozen.** RGA is the minimum in all 2000 replicates in four of the six replicate matrices, in 1999
of 2000 in the forest redraw matrix and in 1997 of 2000 in the forest independent-stream matrix;
the counts are recorded per matrix under `replicate_diagnostics`. The composite is therefore
locally a function
of RGA alone, and the delta method matches the bootstrap: the sd ratio is exactly 1.000 at noise
level 0.5 for both models, and the plug-in minus bootstrap-mean bias is -0.00015 (logistic) and
+0.00031 (forest). Reporting "we could not make the delta method fail at the deposited point"
would have been a legitimate result. We were able to do better than concede it.

**Locating the kink.** RGR falls monotonically in the perturbation scale over the whole range
swept below, and the scale is fixed at 0.5 in both deposited scripts, while RGA does not depend on
it at all. Raising the scale
therefore pushes RGR down through RGA and manufactures a genuine argmin crossing on real
substrate output, with no synthetic data. The family is exact at the deposited point: because
`normal(0, s, size)` equals `s * standard_normal(size)` on the same generator state, we draw the
standard normal matrix once, from the deposited global stream in the deposited order, and scale
it. At s = 0.5 the perturbation is bit-identical to the deposited one, which is why row one of
the sweep below reproduces the deposited numbers.

Coarse sweep, fixed draw, showing the fraction of replicates in which RGR falls below RGA, and
the two standard deviations of the min composite:

| s | RGA (point) | RGR (point) | argmin at point | frac(RGR < RGA) | sd delta | sd bootstrap | ratio |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 0.50 | 0.8052 | 0.9542 | RGA | 0.0000 | 0.02942 | 0.02942 | 1.000 |
| 0.75 | 0.8052 | 0.9141 | RGA | 0.0000 | 0.02942 | 0.02942 | 1.000 |
| 1.00 | 0.8052 | 0.8792 | RGA | 0.0055 | 0.02942 | 0.02932 | 1.003 |
| 1.25 | 0.8052 | 0.8474 | RGA | 0.1020 | 0.02942 | 0.02776 | 1.060 |
| 1.50 | 0.8052 | 0.8191 | RGA | 0.3495 | 0.02942 | 0.02343 | 1.256 |
| 1.75 | 0.8052 | 0.7944 | RGR | 0.6380 | 0.01894 | 0.02018 | 0.939 |
| 2.00 | 0.8052 | 0.7731 | RGR | 0.8365 | 0.02013 | 0.01980 | 1.016 |
| 2.50 | 0.8052 | 0.7396 | RGR | 0.9655 | 0.02188 | 0.02156 | 1.015 |
| 3.00 | 0.8052 | 0.7143 | RGR | 0.9935 | 0.02309 | 0.02302 | 1.003 |
| 4.00 | 0.8052 | 0.6779 | RGR | 1.0000 | 0.02492 | 0.02492 | 1.000 |
| 6.00 | 0.8052 | 0.6371 | RGR | 1.0000 | 0.02645 | 0.02645 | 1.000 |
| 8.00 | 0.8052 | 0.6148 | RGR | 1.0000 | 0.02738 | 0.02738 | 1.000 |

(Logistic model. The forest sweep is in `results-aggregation.json`; it crosses earlier, between
s = 0.75 and s = 1.00, with the same shape.)

Read the two sd columns down the table. The delta standard deviation is constant at 0.02942 while
RGA is the argmin at the point, then jumps discontinuously to 0.01894 when the argmin switches to
RGR at s = 1.75 -- a 36% drop caused by a change in the point vector of 0.025 in one coordinate --
and then climbs again. The bootstrap standard deviation moves continuously through the same
region: 0.02942, 0.02932, 0.02776, 0.02343, 0.02018, 0.01980. The delta method is not merely
inaccurate at the kink; it is discontinuous in the data, which is the precise sense in which it
misbehaves and the paired bootstrap does not.

**At the crossing.** Bisecting on the flip fraction locates s = 1.6091 for the logistic model,
where 1000 of 2000 replicates have RGA as the minimum and 1000 have RGR, and s = 0.9324 for the
forest, where the split is 1003 to 997. There:

| Quantity | logistic, s = 1.6091 | forest, s = 0.9324 |
|:--|:--|:--|
| point (RGA, RGR) | (0.80522, 0.80726) | (0.82721, 0.82733) |
| replicates whose argmin differs from the point's | 50.00% | 49.85% |
| plug-in C = min(point) | 0.80522 | 0.82721 |
| bootstrap mean of C | 0.79280 | 0.81419 |
| plug-in minus bootstrap mean | +0.01241 | +0.01303 |
| sd delta | 0.02942 | 0.02786 |
| sd bootstrap | 0.02152 | 0.02148 |
| sd ratio (delta / bootstrap) | 1.367 | 1.297 |
| 95% CI, delta | [0.7476, 0.8629] | [0.7726, 0.8818] |
| 95% CI, paired bootstrap | [0.7443, 0.8286] | [0.7683, 0.8520] |
| width, delta | 0.11532 | 0.10921 |
| width, bootstrap | 0.08429 | 0.08372 |
| analytic cross-covariance share | 0.00% | 0.00% |
| geometric-mean control, sd ratio at the same s | 0.997 | 0.997 |

The sharpest way to state the failure: the delta interval at the crossing is
**numerically identical** to the delta interval at the deposited operating point --
[0.7476, 0.8629] for the logistic model at both s = 0.5 and s = 1.6091, to every printed digit --
because RGA is still the argmin at the point estimate and the variance of RGA does not depend on
the perturbation scale. The delta method has not registered that anything changed, while half the
replicates now have a different active coordinate. Over the same move the bootstrap interval
narrows from width 0.11580 to 0.08429, and the plug-in point estimate acquires a +0.012 upward
bias relative to the bootstrap mean, because the expectation of a minimum is below the minimum of
the expectations.

The bootstrap interval loses that width entirely from the top. Its lower endpoint on the logistic
model is 0.7443038803389271 at s = 0.5 and 0.7443038803389271 at the crossing, unchanged to the
last bit, because the lower tail of the minimum is still made of the same low-RGA replicates,
while the upper endpoint falls from 0.8601 to 0.8286. On the forest the lower endpoint moves only
from 0.7697 to 0.7683 and the upper from 0.8787 to 0.8520. It is the optimistic end of the
interval that the kink destroys, which is the end a compliance argument leans on.

Note the direction, which is not the one a reader might expect. At the kink the delta method
*overstates* the width -- by 37% on the logistic model and 30% on the forest -- so it is
conservative on width; but it is centred 0.012 too high, and its upper endpoint is 0.034 above the
bootstrap's on the logistic model and 0.030 above on the forest. For a composite compared against
a compliance threshold near the top of the scale, being 0.034 too high at the upper end is the
dangerous error, not the width. We report the direction rather than implying that the naive
method always errs one way.

Two controls keep this honest. The geometric mean, evaluated at exactly the same noise level and
on exactly the same replicates, shows a delta-to-bootstrap sd ratio of 0.997 for both models and
a lower-bound gap of 0.0013 to 0.0015: the discrepancy is specific to the non-smooth composite
and is not an artefact of the raised noise level. And the caveat that matters most: changing the
perturbation scale changes the estimand. These numbers are a methodological demonstration of what
a kink does to a linearisation. They are not a correction to any deposited figure and must not be
mixed into the headline tables of Section 4, which are all at the deposited scale of 0.5.

The sweep was run under the fixed-draw scheme only. Under the redraw scheme each noise level
would cost about five minutes of forest predictions and would not change the geometry being
demonstrated, since the kink is a property of the composite and the location of the argmin, not
of the perturbation conditioning.

## 9. What this does not settle

Pavia's actual composite is not known. Kolesnikov is away until mid-August and Babaei until the
end of August, and the aggregation function is not in the public package. This study does not
guess at it. What it does is make the guess unnecessary: the interval construction is written
once and applied to seven composites, four of them structurally different, and adding an eighth
is a matter of supplying the composite function and, optionally, its gradient. If Pavia's
composite turns out to be a weighted sum, arms 4.1 to 4.3 give the construction and the scan of
Section 5 already bounds the cross-term diagnostic over every weight vector, though the interval
endpoints themselves would have to be recomputed at their weights; a geometric mean is
4.4; a TOPSIS composite with a fixed ideal is 4.5, and with a data-dependent ideal it needs the
across-model covariance discussed in Section 3; a Shapley-based composite is 4.6 or 4.7 with the
efficiency caveat of Section 3 attached.

Two things this study deliberately does not do. It takes no position on which composite anyone
should report: the seven arms are targets for one interval construction, not seven candidates
being compared on merit, and the choice of composite belongs to the group that defines the
integrated score. And it covers only composites whose weights are fixed in advance. If Pavia's
composite derives its weights from the same test sample -- Shapley values used as weights rather
than summed, or any other data-driven weighting -- then the weights are random and correlated with
the metrics they weight, and the delta step needs the joint covariance of the weights and the
metric vector, not just of the metric vector. That is the same difficulty as the data-dependent
TOPSIS ideal point of Section 3, arriving through a different door, and it is not covered here.
The paired bootstrap would still work unchanged, provided the weights are recomputed inside each
replicate.

Four limits are worth stating so they are not discovered by a referee. First, one dataset, one
split, n = 300: the cross-covariance shares of the seven reported composites are modest because
RGA's variance dominates this particular covariance matrix, and nothing in this study establishes
finite-sample coverage for any of the intervals. Second, the weight scan of Section 5 is a scan
over composites on one fixed covariance matrix, not over datasets; it bounds what a weighting
choice can do here, and says nothing about how the covariance itself would look on other data.
Third, the independent-stream construction was run under the fixed draw only, so its comparison
with the analytic zero-cross-term calculation is a fixed-draw comparison. Fourth, the redraw arm
reuses the fixed-draw point vector as the linearisation centre, following the deposited redraw
run, which stores only `point_fixed_draw`; the perturbation redraw shifts the replicate
distribution slightly, which is why the redraw bootstrap intervals sit a little off the redraw
delta intervals. Over the six perturbation-sensitive arms and both models, the plug-in minus
bootstrap-mean gap runs from -0.00791 to -0.00051 under the redraw against -0.00008 to +0.00047
under the fixed draw. The min composite is the exception: -0.00015 on the logistic model under
both schemes, and +0.00031 against +0.00032 on the forest, because its value is RGA, which does
not depend on the perturbation at all -- the forest pair differs in the last digit only because
one of the 2000 redraw replicates has RGR rather than RGA as its minimum. That gap is a
conditioning artefact and is reported rather than absorbed.

## 10. Files, runtime, reproduction

| File | Contents |
|:--|:--|
| `aggregation_experiment.py` | the study; regenerates the replicates, runs seven composites, both schemes, both models, and the noise sweep |
| `results-aggregation.json` | every number in this summary, to full precision |
| `run-aggregation.log` | the console trace of the run these numbers come from |
| `.cache/` | regenerated replicate matrices, gitignored; delete to force a full recomputation |

Measured runtime of the run behind this summary, a cold one with no cache present: 228.7 s
reported by the script. The itemised costs are 1.0 s precompute; 1.9 s and 1.8 s for the two
fixed-draw paired passes; 5.5 s and 5.7 s for the two independent-stream passes; 0.6 s for the
logistic redraw pass and 205.4 s for the forest redraw pass, which is 2000 sequential 300-tree
`predict_proba` calls, one per replicate; and 2.1 s and 3.9 s for the two noise sweeps of 22
levels each. Those items account for 227.9 s of the 228.7 s total. The remaining 0.8 s covers data
loading, model fitting, JSON serialisation and the entire aggregation layer: seven composites, two
schemes, two models, every gradient, every interval, every cross-term decomposition, the 1326-point
weight scan per model and the exact Shapley values over 2000 replicates. The cost of this study is
the cost of regenerating replicates the deposited runs did not persist; the aggregation question
itself is free.

The wall-clock figures are the only quantities in this summary that a rerun will not reproduce:
they depend on how loaded the machine is, and the forest redraw pass in particular has been
observed between 169 s and 205 s across reruns of identical code on identical data (the 169 s
reading is from an earlier rerun, whose log the run of record has overwritten; it is quoted here
as a range, not as a figure of this run). Every
statistical number here is deterministic given the seeds and does reproduce exactly, which is what
Section 6 measures. A reader comparing timings against `run-aggregation.log` should expect the
proportions to hold and the absolute seconds not to.

To reproduce: run `python3 aggregation_experiment.py` from this directory with the vendored
`safeai` clone and stubs present alongside the 24 July redraw run. `AGG_B` overrides the replicate
count for a fast smoke run. The cache holds the replicate matrices only, and everything else,
including the perturbation matrix and the index stream the noise sweep needs, is recomputed on
every run: a cached rerun of this study was checked leaf by leaf against the cold run behind this
summary and reproduces every number in `results-aggregation.json`, differing only in the recorded
runtime, the timestamp and the `replicates_loaded_from_cache` flag. It cost 10.6 s, again a figure
from that check's own run rather than from the log of record. Python 3.12.3,
numpy 2.5.0, scikit-learn 1.9.0; no dependency outside numpy and scikit-learn is used. The OpenML
dataset is served from the local scikit-learn cache, so no network access is required.
