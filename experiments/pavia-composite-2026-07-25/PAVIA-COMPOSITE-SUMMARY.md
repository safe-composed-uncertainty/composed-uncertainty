# An interval for the Pavia Compliance Score, on the construction that defines it

Study of 25 July 2026. Anton Sokolov, Tyche Institute, Tallinn.
Feeds Section 5 of the joint article with Paolo Giudici and Vasily Kolesnikov.

Every number below comes from the single run recorded in `results-pavia-composite.json` and
`run-pavia-composite.log`, produced by `pavia_composite_experiment.py` in this directory,
with four exceptions, all labelled where they appear. Two are published sources: the
Giudici-Kolesnikov Table 5 rows, which the run also records and checks, and the Kolesnikov
thesis Table 3.12 rows, which it does not. Two are our own earlier artifacts, quoted only in
Section 5 for the contrast: the four cross-covariance shares and four width understatements of
the companion aggregation study's summary table, and the 6.9%, 11.4% and 23.6% of the deposited
article. Section 5a is the one section with a second provenance: its numbers come from the
adversarial verification run recorded in `verify-cross-terms.log`, produced by
`verify_cross_terms.py` in this directory. Nothing is quoted from memory or from a fact sheet.

If you read one thing, read Section 5. The cross-covariance between the three curve summaries
is small and mixed in sign, and on the random forest it is negative in all five analytic arms
by 2.2 to 3.7 Monte Carlo standard errors, which means declaring the cross-terms zero makes
the interval *wider*, not narrower. That is the opposite of what the scalar experiment found,
and it constrains what our Section 5 may claim. Section 5a reports the adversarial verification
of that reversal: it is real, not a bug and not an anchor artifact, and the estimator-by-estimator
bridge shows the tail-swap perturbation family -- not the sweep, not the averaging -- is what
flips the sign.

## 1. Why this study exists, and how it differs from the aggregation study

The companion aggregation study of the same morning built a family of composites and gave each
one an interval. It composed three *scalars*. The Pavia Compliance Score does not consume three
scalars. Giudici and Kolesnikov, *Integrating Safe AI Metrics* (SSRN 10.2139/ssrn.5362574;
journal version *Machine Learning with Applications* 23:100821), equations (13) to (17), and
Kolesnikov's MSc thesis, equations (2.13) to (2.17), define the score on the three *vectors*
from which the RGA, RGE and RGR curves are built. For every triplet of indices `(i, j, k)` a
mean of `(RGA[i], RGE[j], RGR[k])` is taken -- arithmetic (13), geometric (14), root mean
square (15) -- filling an `L x L x L` tensor `M` (16), and the Compliance Score is the average
height of that tensor (17):

    V = (1 / L^3) * sum_i sum_j sum_k M(i, j, k)

This study redoes the interval layer on that construction. The aggregator machinery of the
aggregation study transfers unchanged; what changes is the object being composed, and, as
Section 5 shows, that change moves the answer.

## 2. What was run

Substrate, data, seed and split are inherited from the deposited runs so that every number here
is commensurable with the published ones.

- Substrate: `safeai` at pinned commit `39768fcd5264c881f7174268bbffda52b298ae89`, tabular code
  paths only, reusing the vendored clone and the inert import stubs of the 24 July redraw run.
  This study never clones, never writes into the substrate, and never touches the deposited
  experiment directories.
- Data: Statlog German credit (OpenML `credit-g` v1), 700 train / 300 test, d = 20 predictors,
  seed 20260723, stratified split. numpy 2.5.0, Python 3.12.3.
- Models: a standardised logistic regression and a 300-tree random forest, fixed after training.
- Curve resolution: the common normalised severity grid `s_t = t / d`, `t = 0 .. d`, hence
  L = d + 1 = 21 knots. Section 3 explains why 21 and not 20.
- Replicates: B = 2000 paired bootstrap replicates of the three curves, one row resample per
  replicate, all three curves recomputed on that same resample; plus B = 2000
  independent-stream replicates, one index stream per curve, which are the assumption-light
  empirical counterpart of declaring the cross-terms zero. Index-stream recipes are the
  deposited ones.
- Conditioning check: replicates in which the greedy removal order is re-searched and the
  perturbation quantiles and masking column means are recomputed on the resampled rows, at
  B = 300 for the logistic model and B = 30 for the forest, each with a measured Monte Carlo
  band on the spread ratio it reports. Section 10 explains the asymmetry.

Total runtime 898.8 s, that is 15.0 minutes, on one workstation. Section 10 breaks it down and
explains the choice of B.

## 3. The three curves, and the one reconciliation the data force

The three axes do not naturally share a length. RGA resolution is free: `rga_curve` with
`curve_method='partial'` returns `n_segments + 1` values for any `n_segments`. RGR resolution is
free: the article fixes the interval `p` in `[0, 0.5]` but never the increment. RGE resolution is
pinned by the data at d + 1 = 21, because the greedy removal search runs out of features at
step 21 and the complete curve is the anchor plus one value per predictor.

We put all three on the same severity grid `s_t = t/d` and accept L = 21:

- RGA at `n_segments = 20`: index t = t top-ranked segments of the test data removed;
- RGE at `n_steps = 20`, greedy, masking to the column mean: index t = t predictors removed,
  including the all-removed endpoint;
- RGR at `p_t = 0.5 * t/20`: the article's own tail-swap perturbation at fraction t/20 of the
  maximum severity the article allows.

Every index then means the same fraction of that dimension's own maximum severity, which is what
makes the diagonal reading of Section 7 well defined at all.

Our L = 21 is one longer than the article's parenthetical "vectors of equal length n
(corresponding to the number of features)". That parenthetical also sits one short of the
vectors the article's own equations write, and it is worth being explicit about why rather than
asserting it. Equation (6) writes the accuracy vector as `[RGA, RGA - RGA_1, ..., 0]`, which is
the zero-severity anchor plus one entry per removed segment; the text under eq. (12) says the
flipped explainability curve "starts at 1" at p = 0 and "ends at the random-baseline value of
0.5" at p = 1, which on a d-feature dataset is again d + 1 knots; and the text around eq. (9)
says RGR "starts at 1 when p=0", though there the increment is never fixed, so that vector's
length stays a free choice. This is an observation about a parenthetical, not a defect in the
construction, and we do not hide the cost of our own choice. The literal length-n reading is
reached by dropping one end knot, and the text does not say which: dropping the zero-severity
anchor and dropping the terminal knot are both length 20, they move every composite in opposite
directions, and the spread between the two readings is 0.62 to 0.85 of an interval width.
Section 9 reports both truncations and that spread; the anchor-dropped one alone moves every
composite by 0.012 to 0.017, between 0.23 and 0.28 of an interval width.

Point curves, both models, L = 21, from the run:

| Model | Curve | knots 0..4 | knots 16..20 |
|:--|:--|:--|:--|
| logistic | RGA | 0.8052 0.7974 0.7974 0.7896 0.7752 | 0.3520 0.3269 0.2518 0.1335 0.0000 |
| logistic | RGE | 1.0000 0.9999 0.9998 0.9996 0.9994 | 0.9394 0.9185 0.8998 0.8367 0.5083 |
| logistic | RGR | 1.0000 0.8654 0.7600 0.6762 0.5963 | 0.1740 0.1671 0.1633 0.1588 0.1597 |
| forest | RGA | 0.8272 0.8272 0.8272 0.8028 0.8028 | 0.4300 0.3439 0.2502 0.1216 0.0000 |
| forest | RGE | 1.0000 0.9999 0.9988 0.9975 0.9961 | 0.9166 0.9000 0.8731 0.7938 0.5058 |
| forest | RGR | 1.0000 0.9161 0.8359 0.7636 0.6893 | 0.2927 0.2819 0.2777 0.2719 0.2723 |

The RGE curves end at 0.5083 and 0.5058, which is the 0.5 random baseline the article and the
thesis both predict. The RGA anchor is the model's own RGA and it reproduces the deposited point
estimate exactly: 0.8052154195011337 and 0.8272108843537415, absolute gap 0.0 for both models
against the deposited `results.json`, which this run opens read-only. The curve construction sits
on top of the deposited point estimate rather than beside it.

### 3.1 Two substrate choices that had to be settled by measurement, not by preference

**The perturbation.** The pinned package ships no tail swap: `swap`, `quantile` and `percentile`
do not occur anywhere in its source. Its tabular RGR curve builder offers a Gaussian-noise sweep
and an adversarial sweep, and neither is the article's perturbation. The article defines that
perturbation in the clause that fixes p in eq. (9) -- for each p in [0, 0.5] the lowest
p-quantile of observations is swapped with the highest p-quantile -- so we rebuilt it on top of
the package's published `rgr_score`. Section 9 reports what the Gaussian arm would have given
instead, and the answer is that the perturbation choice moves the composite by 0.12 to 0.22,
which is 2.4 to 3.3 times the width of that composite's own interval. This is the largest single
free choice in the whole construction and it is not a statistical one.

**The scoring convention.** The package's own curve builders resolve a class order from the fitted
model and score full probability matrices; `rge_score` and `rgr_score` called on a bare
one-dimensional probability vector do not. `rge_curve` gives a reference curve, so the question is
settled by measurement: our fast path reproduces `rge_curve` to 0.00e+00 (logistic) and 2.22e-16
(forest) when scored on the probability matrix, and diverges by up to 1.87e-02 (logistic) and
1.16e-02 (forest) when scored on the bare vector. We therefore score all three curves on the
probability matrix. For RGR the two conventions differ by up to 0.0336 (logistic) and 0.0233
(forest) over the curve, and at p = 0.5 they read 0.1597 against 0.1267 and 0.2723 against 0.2701.
Under the bare-vector convention, applied to every convention-sensitive curve at once, the three
composites for the logistic model would be 0.62854 / 0.52502 / 0.69974 instead of
0.63100 / 0.53471 / 0.69885. The run records both. Anyone splicing
these numbers into the article must not mix the conventions between curves.

## 4. The composite, and the factorisation

The triple sum factorises for two of the three variants:

- arithmetic: `V = (mean(RGA) + mean(RGE) + mean(RGR)) / 3`
- geometric: `V = mean(RGA^(1/3)) * mean(RGE^(1/3)) * mean(RGR^(1/3))`
- root mean square: no factorisation; the square root of a sum does not separate.

We re-verified this on the real curves, against a brute-force triple sum written with three
explicit Python loops and no array broadcasting, so that the check is against the definition as
written and not against a rearrangement of it. Residuals, absolute:

| Model | Variant | brute force | vectorised tensor | closed form | tensor - brute | closed - brute | closed - tensor |
|:--|:--|--:|--:|--:|--:|--:|--:|
| logistic | arithmetic | 0.631002815389 | 0.631002815389 | 0.631002815389 | 4.774e-15 | 4.774e-15 | 0.000e+00 |
| logistic | geometric | 0.534706260611 | 0.534706260611 | 0.534706260611 | 2.776e-15 | 2.887e-15 | 1.110e-16 |
| logistic | rms | 0.698847909330 | 0.698847909330 | none | 4.441e-15 | n/a | n/a |
| forest | arithmetic | 0.672086493118 | 0.672086493118 | 0.672086493118 | 7.772e-16 | 7.772e-16 | 0.000e+00 |
| forest | geometric | 0.600733353066 | 0.600733353066 | 0.600733353066 | 8.882e-16 | 9.992e-16 | 1.110e-16 |
| forest | rms | 0.721070073530 | 0.721070073530 | none | 1.110e-15 | n/a | n/a |

**The factorisation residual is at most 4.774e-15 in absolute value, over all six model-variant
cells.** The residual against the vectorised tensor is smaller still, 0.000e+00 for the arithmetic
variant and 1.110e-16 for the geometric, because the same floating-point sums are being formed;
the 1e-15 figures are the price of the different summation order that the explicit loops impose.
At L = 21 the closed forms replace 9261 evaluations with three averages.

The root mean square variant was evaluated as the full tensor. Its gradient with respect to each
curve entry is available in closed form, and because `M` is homogeneous of degree one in
`(RGA, RGE, RGR)` jointly the projections onto that gradient must reconstruct V exactly. They do:
`g_A . RGA + g_E . RGE + g_R . RGR` equals V to 1.110e-16 (logistic) and 0.000e+00 (forest).

## 5. The headline table, and the article's throughline

`V` is the point value of the composite. `boot 95%` is the paired-bootstrap percentile interval
and `w` its width. `delta w` is the width of the delta interval built on the measured covariance
between the three curve summaries; `zero w` is the same with the cross-terms declared zero;
`under%` is the resulting understatement of the width and `var%` the share of the variance
thrown away, both with the Monte Carlo standard error from 400 second-level resamples of the
B replicate rows. `emp under%` is the assumption-light counterpart: the width of the
independent-stream bootstrap against the width of the paired one.

### Logistic regression

| Composite | V | boot 95% | w | delta w | zero w | under% (se) | var% (se) | emp under% (se) |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| arithmetic | 0.63100 | [0.60343, 0.66098] | 0.05755 | 0.05639 | 0.05690 | -0.91 (0.73) | -1.84 (1.48) | +0.89 (3.01) |
| geometric | 0.53471 | [0.49248, 0.56699] | 0.07452 | 0.07224 | 0.07358 | -1.86 (0.98) | -3.76 (2.00) | +1.45 (3.58) |
| root mean square | 0.69885 | [0.67837, 0.72480] | 0.04643 | 0.04665 | 0.04606 | +1.26 (0.54) | +2.51 (1.07) | -1.41 (2.55) |
| TOPSIS (thesis) | 0.38025 | [0.35749, 0.40591] | 0.04842 | 0.04910 | 0.04984 | -1.50 (0.63) | -3.02 (1.29) | -3.10 (2.83) |
| arithmetic, weighted (sensitivity) | 0.58575 | [0.54689, 0.62863] | 0.08174 | 0.08158 | 0.08218 | -0.74 (0.46) | -1.48 (0.94) | -2.00 (2.97) |

### Random forest

| Composite | V | boot 95% | w | delta w | zero w | under% (se) | var% (se) | emp under% (se) |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| arithmetic | 0.67209 | [0.64631, 0.70013] | 0.05382 | 0.05451 | 0.05618 | -3.08 (0.89) | -6.25 (1.84) | -6.49 (3.47) |
| geometric | 0.60073 | [0.56661, 0.63248] | 0.06587 | 0.06626 | 0.06831 | -3.09 (1.02) | -6.27 (2.10) | -5.21 (3.07) |
| root mean square | 0.72107 | [0.70082, 0.74615] | 0.04533 | 0.04578 | 0.04647 | -1.50 (0.69) | -3.02 (1.39) | -4.86 (3.03) |
| TOPSIS (thesis) | 0.41533 | [0.39390, 0.43807] | 0.04417 | 0.04539 | 0.04680 | -3.11 (0.84) | -6.32 (1.73) | -7.56 (3.19) |
| arithmetic, weighted (sensitivity) | 0.63005 | [0.59332, 0.66982] | 0.07651 | 0.07775 | 0.07923 | -1.90 (0.58) | -3.84 (1.19) | -6.81 (2.78) |

**The intervals exist, they are the right size, and the two constructions agree on the width.**
Widths run from 0.04417 to 0.08174. The delta widths sit within 3.1 per cent of the paired
bootstrap widths in every row, including the two arms where the composite is not an exact
function of three summaries and the delta interval is a first-order device. So the linearisation
is adequate for the width at this operating point. The endpoints are a separate question and the
agreement there is weaker: the delta and percentile endpoints differ by up to 8.20 per cent of a
width, at the lower end of the logistic geometric arm. Since our own deposited article argues
about where an interval's lower endpoint falls relative to a decision threshold, the two
constructions are interchangeable for the width and should not be treated as interchangeable for
a threshold comparison.

**The throughline does not hold on the curve construction, and we must say so.** On the scalar
construction, the two arms closest to the two factorising variants here were the equal-weight sum
and the geometric mean; the companion aggregation study's summary records their cross-covariance
shares as 7.46% and 6.33% of the variance on the logistic model and 11.78% and 10.95% on the
forest, with width understatements of 3.80%, 3.22%, 6.07% and 5.63%. Here the same diagnostic
reads between -6.32% and +2.51% of the variance, and the understatement between -3.11% and +1.26%.
Three of the ten analytic rows sit within two Monte Carlo standard errors of zero -- the logistic
arithmetic, geometric and weighted arms, at 1.2, 1.9 and 1.6 standard errors. All five
forest rows are *negative*, at 2.2 to 3.7 standard errors, which means the independence assumption
makes the interval **wider** on that model, not narrower. The honest statement for Section 5 is
that on this data and this construction the cross-covariance between the three curve summaries is
worth a few per cent of the variance at most, that its sign is not stable across variants or
models, and that the single-system 6.9% and 11.4% and the chain-level 23.6% quoted in the
deposited article are properties of the scalar estimators used there, not of composition as such.

**Where the change comes from, measured rather than asserted.** The run records the correlation of
the three curve summaries and, separately, the correlation of the three curves at matched severity
index. Both are small, so the effect is not an artefact of averaging along the severity axis:

| Quantity | RGA-RGE | RGA-RGR | RGE-RGR |
|:--|--:|--:|--:|
| logistic, correlation of the three curve means | +0.1996 | -0.0583 | -0.1372 |
| logistic, mean correlation at matched severity index | +0.1089 | -0.0259 | -0.1082 |
| logistic, deposited scalar metrics (read from the deposit) | +0.2858 | +0.1596 | +0.5967 |
| forest, correlation of the three curve means | +0.1574 | -0.0800 | -0.3102 |
| forest, mean correlation at matched severity index | +0.0533 | -0.0330 | -0.1524 |
| forest, deposited scalar metrics (read from the deposit) | +0.2922 | +0.1530 | +0.3415 |

The RGE-with-RGR pair, which carried the largest positive correlation in the deposit, has changed
sign. Two estimators changed at once -- RGE moved from an average over d single-feature masks to
the nested greedy curve, and RGR from a single fixed Gaussian draw to the tail-swap sweep -- and
the run of record does not separate them. The verification run of Section 5a does: it is the RGR
change, and specifically the tail-swap perturbation family, that flips the sign.

The variance remains dominated by RGA: the bootstrap standard errors of the three curve means are
0.041022, 0.002769 and 0.014351 on the logistic model and 0.038937, 0.003067 and 0.017978 on the
forest. A composite of three quantities of which one has thirteen to fifteen times the standard
error of another has little room for cross-terms to matter, whatever their sign.

**The empirical column is noisier than the analytic one, and the run now says by how much.** The
Monte Carlo standard errors of the independent-stream understatement are 2.5 to 3.6 percentage
points, against 0.5 to 1.0 for the analytic understatement in the same rows, and 0.9 to 2.1 for
the analytic variance share. Every logistic-model sign disagreement
between the two columns is inside that noise. The forest rows agree in sign and roughly in size.
This reproduces, with an explicit error bar this time, what the aggregation study observed
qualitatively.

## 5a. The sign reversal survives an adversarial verification, and the cause is isolated

Because the reversed sign is the one result that changes what our article may claim, we attacked
it before accepting it. `verify_cross_terms.py` in this directory is an independent rewrite --
its own tail-swap implementation, its own replicate loop, its own delta arithmetic -- that
regenerates the paired replicates from the documented recipe and recomputes everything the claim
rests on. Its run of record is `verify-cross-terms.log`, and every number in this section comes
from that log or from `results-pavia-composite.json` where marked.

**It is not a bug.** The regenerated covariance matrix of the three curve summaries matches the
stored one with a maximum absolute gap of 0.00e+00 on both models, and the point curves match the
stored curves to 0.00e+00. The delta arithmetic recomputed by hand from that matrix reproduces the
understatement to three decimals on every arm checked: -0.914 / -1.862 / +1.264 (logistic,
arithmetic / geometric / rms) and -3.076 / -3.086 / -1.498 (forest). The independent-stream
widths, a construction that shares no code with the delta block, reproduce as well: +0.89 / +1.45
(logistic) and -6.49 / -5.21 (forest). The factorisation was re-verified by brute-force triple
loops at 4.8e-15 (arithmetic) and 2.9e-15 (geometric), and the nearest natural closed-form
candidate for the root mean square variant misses by at least 2.2e-02, so the tensor there is
genuinely necessary. The sign conventions read the way the tables use them: a negative
understatement means the declared-zero interval is the wider one.

**The deterministic anchors are not the mechanism, by algebra and then by measurement.** Each
curve carries exactly one deterministic knot (the RGE and RGR anchors at 1, the terminal RGA knot
at 0; replicate standard deviations 0.00e+00, 0.00e+00 and 1.1e-16). A curve mean is therefore an
affine function of its twenty random knots, and deleting the deterministic knot can change no
correlation and no cross-covariance share. Measured: deleting them changes the correlations by at
most 8.3e-16 and leaves the arithmetic cross share at -1.836% (logistic) and -6.246% (forest) to
the printed precision. The "anchors shrink the covariance" reading is excluded.

**The decorrelation is real, and the bridge says exactly where it comes from.** The deposited
experiment draws its paired replicates with the same recipe and seed as this study, so its three
scalar metrics and our three curve summaries live on identical row resamples and their
correlations are directly commensurable; the verification recomputed the deposited scalars on
those streams and reproduced the deposit's correlation matrix with a maximum gap of 0.00e+00.
Walking the RGE-RGR pair from the deposit's estimators to ours, one change at a time:

| RGE estimator x RGR estimator | logistic | forest |
|:--|--:|--:|
| single-mask mean x fixed Gaussian draw (the deposit pair) | +0.5967 | +0.3415 |
| same, both scored on the probability matrix (convention only) | +0.6144 | +0.3437 |
| greedy curve mean x fixed Gaussian draw (RGE estimator swapped) | +0.4249 | +0.2879 |
| single-mask mean x tail-swap curve mean (RGR estimator swapped) | -0.2083 | -0.1529 |
| greedy curve mean x Gaussian-noise sweep mean (sweep, same family) | +0.5354 | +0.4711 |
| greedy curve mean x tail-swap curve mean (the study pair) | -0.1372 | -0.3102 |

The scoring convention is worth nothing. Swapping the RGE estimator attenuates the correlation
but leaves it clearly positive. Swapping the RGR estimator alone flips it negative on both
models. And the fifth row is the control that kills the tidy explanation we ourselves expected:
a full severity sweep of the *Gaussian-noise* family, averaged exactly like the tail-swap curve,
stays as positively correlated as the scalars. Sweeping and averaging do not decorrelate
anything. The tail-swap perturbation family does. The per-severity-index correlations in the
results file say the same thing from inside the primary run: the RGE-RGR correlation starts
positive at low severity and falls to -0.31 (logistic) and -0.51 (forest) deep in the sweep. The
same family swap also drags RGA-RGR from +0.16 to -0.06 (logistic) and from +0.15 to -0.08
(forest). Why deep tail-swaps anti-correlate with the other two curves on a shared resample is a
question this verification does not answer, and we flag it as open rather than decorate it.

**The one positive cell has the same explanation.** The logistic root-mean-square arm is the only
analytic cell where declaring the cross-terms zero still narrows the interval (+1.26%, mc se
0.54). The rms gradient weights each knot by its value, which shifts weight toward low severity,
where the pairwise correlations are still positive: under the rms projections the RGE-RGR
correlation relaxes from -0.1372 to -0.0252 and RGA-RGE holds at +0.21, so the surviving positive
RGA-RGE block wins. The same reweighting on the forest is not enough against its stronger
negative RGE-RGR block, which is why that cell stays negative.

**What this study now licenses, in two sentences the article can carry.** Where a composite
consumes metrics estimated by small perturbations of one prediction vector on one shared test
sample -- our scalar composites, and a fortiori the two-link chain -- the cross-metric covariance
is strongly positive: declaring it zero discards 6.9 per cent (logistic) and 11.4 per cent
(forest) of the composite's variance in the deposited single-system case, and understates the
deposited two-link chain interval by 23.6 per cent of its width. On the volume composite of Giudici and Kolesnikov on German
credit the same declaration moves the width by at most a few per cent with no stable sign,
because the tail-swap robustness sweep their article specifies happens to be nearly uncorrelated
with the accuracy and explainability curves on the shared sample; the safety of the independence
assumption is thus a property of the estimators -- of the perturbation family above all -- and
not of composition as such, which is precisely why the interval has to be measured rather than
assumed.

## 6. TOPSIS, which is Kolesnikov's and not ours

TOPSIS appears in Vasily Kolesnikov's MSc thesis (University of Pavia, supervisor Paolo Giudici,
co-supervisor Alessandra Tanda, open access in the university repository), section 2.2.2 for the
method and section 3.2.3 for the case. The word does not occur once in the published article. We
are bringing forward a piece of his work that did not reach print, and the write-up must say so
and credit him by name.

The construction is his, verbatim. Alternatives are models, criteria are the three metric vectors,
and the ideal and anti-ideal solutions are hand-constructed rather than read from the data:

- best case: RGA all ones; RGE `[1, 0, 0, ..., 0]`; RGR all ones
- worst case: RGA all zeros; RGE a linear decay `[1, 0.9737, 0.9474, ..., 0.5]`; RGR `[1, 0, ..., 0]`

The thesis states that the best-case vector yields `Ci = 1.0` and the worst-case vector `Ci = 0.0`.
Our implementation reproduces both exactly: 1.000000 and 0.000000. That is the check that the
distance and closeness definitions have been read correctly.

The printed worst-case RGE is `numpy.linspace(1, 0.5, 20)` -- the run records its first three
entries as 1.0, 0.9736842105263158, 0.9473684210526316, against the printed 1.0, 0.9737, 0.9474.
It is therefore an anchor plus nineteen steps on a twenty-predictor dataset, one step short of the
thesis's own curve. Our primary reads the same construction at the length our curves actually
have, L = 21, giving a worst-case RGE of `linspace(1, 0.5, 21)` whose head is 1.0, 0.975, 0.95.
Section 9 reports the literal length-20 reading as a sensitivity; it moves the closeness
coefficient by +0.0085 on the logistic model and +0.0082 on the forest.

Two instantiation decisions we made and will state in the article rather than bury:

1. **No column normalisation.** The thesis's generic step 2 divides each column by the square root
   of the sum of squares over alternatives. We do not apply it, because the rank-graduation metrics
   are already on a common [0,1] scale and because a data-dependent normalisation would make each
   model's score depend on the other models in the comparison set, which would require the joint
   covariance across models rather than across curves. Without normalisation the closeness
   coefficient of a model depends on that model alone, which is what lets us put an interval on it.
2. **Equal weights.** The thesis Table 3.12 is the equal-weight case. The closeness coefficient is
   invariant to a common rescaling of the weight vector, so equal weights are the no-weight case
   and the weighting decision does not enter.

Our values are 0.38025 (logistic) and 0.41533 (forest). The thesis Table 3.12 reports a 0.336 to
0.468 band across seven models and four datasets, with LR below RF in every column. Our two values
fall in that band and preserve that ordering. **This is a plausibility check on the
implementation, nothing more.** The thesis cases are New York and California mortgage data and an
employee dataset; ours is German credit. No numerical agreement is claimed.

## 7. The full product space against the matched-severity diagonal

Their volume averages over the full product space of indices: accuracy at removal step i is
combined with explainability at an unrelated step j and robustness at an unrelated noise level k.
The alternative reading averages only the diagonal i = j = k, pairing the three degradations at
matched severity. Both were computed, and the difference was bootstrapped as a paired quantity so
that the comparison carries its own interval.

| Model | Variant | product space | diagonal | diagonal - product | 95% of the difference | excludes zero | as a fraction of the composite's own interval width |
|:--|:--|--:|--:|--:|:--|:--|--:|
| logistic | arithmetic | 0.631003 | 0.631003 | +0.000000 | [-2.2e-16, +2.2e-16] | no | 0.000 |
| logistic | geometric | 0.534706 | 0.559463 | +0.024757 | [+0.021897, +0.032753] | yes | 0.332 |
| logistic | rms | 0.698848 | 0.690461 | -0.008387 | [-0.009287, -0.007652] | yes | 0.181 |
| forest | arithmetic | 0.672086 | 0.672086 | +0.000000 | [-2.2e-16, +2.2e-16] | no | 0.000 |
| forest | geometric | 0.600733 | 0.622694 | +0.021961 | [+0.019490, +0.025896] | yes | 0.333 |
| forest | rms | 0.721070 | 0.711908 | -0.009162 | [-0.010078, -0.008340] | yes | 0.202 |

**For the arithmetic variant the two readings are the same object.** This is algebra, not a
coincidence: the diagonal average of `(a_t + e_t + r_t)/3` is `(mean a + mean e + mean r)/3`,
which is V. The run confirms it numerically at every one of the 2000 replicates: the standard
deviation of the difference is 1.01e-16 for the logistic model and 1.07e-16 for the forest. So the
choice between the product space and the matched-severity diagonal is invisible to the article's
first variant.

**It is visible in the other two, with opposite signs, and it is larger than sampling noise.** The
diagonal is higher for the geometric variant by 0.0248 and 0.0220, and lower for the root mean
square by 0.0084 and 0.0092. Both differences exclude zero comfortably. The shape is what one
expects when the three curves are co-monotone in severity: the diagonal pairs high with high and
low with low, which a concave mean rewards and a convex mean penalises.

**How much it matters, stated in the only units that are honest.** The geometric difference is a
third of that composite's own 95% interval width, and the root-mean-square difference a fifth. It
is smaller than the interval, and it is not negligible against it. It is also large against the
gaps between models in the article's own Table 5, with one caveat that must travel with the
comparison: our shift is measured on German credit and their spreads on California HMDA, so this
is a comparison of magnitudes across datasets, not a statement about their table. The comparison
also has to be made column by column,
because the indexing shift is zero in the arithmetic column by the algebra just given. In their
geometric column the five models other than logistic regression and the random baseline lie
between 0.52752 and 0.53229, a spread of 0.00477, and the geometric indexing shift measured here
is 5.2 times that spread on the logistic model and 4.6 times on the forest. Their root-mean-square
column has a spread of 0.00855 over the same five models, and the root-mean-square shift is about
equal to it, 0.98 and 1.07 times. A reader comparing models on a volume would be affected by which
reading was used.

**The conceptual point, and its limit.** The composite's own construction treats the three
dimensions as independently indexed: the tensor runs over all `(i, j, k)`, and the article's
"all possible combinations of perturbation types" is exactly that. The sampling uncertainty of the
three curves, by contrast, is coupled, because all three are computed on the same test sample. The
first is a modelling choice about which severity combinations count; the second is a fact about
the estimator. This study measures both. What it does *not* support is the neat version of the
sentence, that independent indexing and coupled uncertainty pull in opposite directions by
comparable amounts: the indexing choice moves the point value by up to a third of an interval
width, while the coupling moves the width itself by a few per cent and in an unstable direction.
On this data the indexing choice is the larger effect by an order of magnitude, and the article
should say that rather than the tidier thing.

One further limit, which belongs in the same paragraph. The diagonal comparison is only defined
because our reconciliation puts the three vectors on one severity axis. Under the article's own
construction, where the three grids are chosen independently and may even have different lengths,
the diagonal has no meaning. We are not correcting them; we are pointing out that a reading their
formula does not offer becomes available once the grids are aligned, and reporting what it costs.

## 8. The ordering the power-mean inequality forces

For any non-negative triple, geometric mean <= arithmetic mean <= root mean square. The composite
inherits the ordering term by term, so it must hold for V as well.

Our implementation reproduces it. At the point estimate: 0.534706 <= 0.631003 <= 0.698848 for the
logistic model and 0.600733 <= 0.672086 <= 0.721070 for the forest. Across the bootstrap it holds
in 2000 of 2000 replicates for both inequalities and both models.

The published Table 5 shows the same ordering in every one of its seven rows, which the run checks
row by row: LR, RF, XGB, SEM, VEM, NN and Random all satisfy geometric <= arithmetic <= rms. Their
LR row is 0.59695 / 0.47965 / 0.67482 and their Random row is 0.76681 / 0.61145 / 0.84059.

**No numerical agreement is claimed or expected.** Their Table 5 is California mortgage data and
ours is German credit, their models include gradient boosting and neural networks that we do not
fit, and the perturbation increment that fixes their curve resolution is never stated. What is
checked is the ordering, and it holds on both sides.

## 9. Sensitivities

All from the same replicate matrices, so the shifts are directly comparable with the interval
widths of Section 5.

| Sensitivity | logistic: arithmetic / geometric / rms | forest: arithmetic / geometric / rms |
|:--|:--|:--|
| primary, L = 21, tail swap | 0.63100 / 0.53471 / 0.69885 | 0.67209 / 0.60073 / 0.72107 |
| zero-severity anchor dropped, L = 20 | -0.01520 / -0.01712 / -0.01306 | -0.01352 / -0.01486 / -0.01219 |
| terminal knot dropped, L = 20 | +0.02042 / +0.03813 / +0.01577 | +0.02064 / +0.04084 / +0.01573 |
| RGR as a scaled Gaussian noise sweep | +0.18771 / +0.22149 / +0.15060 | +0.14335 / +0.15758 / +0.12127 |

**The literal "length n" wording is ambiguous, and the ambiguity is not cheap.** There are two
ways to shorten a 21-knot curve to the article's stated length 20 -- drop the zero-severity
anchor, or drop the terminal knot -- and the text does not choose between them. The two readings
move every composite in opposite directions, for every variant and both models. The spread
between them is 0.036 / 0.055 / 0.029 (logistic, arithmetic / geometric / rms) and 0.034 /
0.056 / 0.028 (forest), which is 0.62 to 0.85 of that composite's own interval width -- larger
than the cost of either single reading. Reporting only one truncation would understate the
ambiguity, so the run reports both, and the anchor-dropped cost of 0.23 to 0.28 of an interval
width quoted in Section 3 is the smaller half of the story. We keep the anchors in the primary
because they carry information: the RGA anchor is the model's own RGA, and the anchors are what
let the geometric variant see the full range.

**The perturbation choice dominates everything else in this study.** Replacing the article's
tail swap with the package's scaled Gaussian sweep moves the composite by +0.121 to +0.221, which
is 2.4 to 3.3 times the width of the interval it moves, and it moves it in the same direction for
every variant and both models. No statistical refinement anywhere in this study is worth as much
as that one modelling
decision. It is also the decision the published article specifies and the public package does not
implement. Section 5 of the article should carry this sentence.

**TOPSIS at the literal thesis length.** The thesis reference vectors keep the anchor -- the
printed worst-case RGE is `[1.0, 0.9737, 0.9474, ..., 0.5]`, an anchor plus nineteen steps -- so
the length-20 reading that matches them drops the *last* knot of each curve. That reading gives
0.38871 (logistic, +0.00846) and 0.42356 (forest, +0.00823), with paired bootstrap intervals
[0.36545, 0.41465] and [0.40180, 0.44657]. Dropping the anchor instead gives 0.36925 (logistic,
-0.01100) and 0.40576 (forest, -0.00956), with intervals [0.34485, 0.39644] and
[0.38342, 0.42984]. The same truncation ambiguity as above, in the same opposite-sign shape;
the spread between the two length-20 readings is 0.0195 and 0.0178, which is 0.40 of the
interval width for both models.

## 10. Cost, the choice of B, and the conditioning check

**Why the study is affordable at all.** Evaluating the three curves naively from the fitted model
costs 0.085 s for the logistic model and 30.9 s for the forest, of which the greedy RGE search is
almost all. At that rate a paired bootstrap of B = 2000 would cost 17 hours on the forest. The
fast path freezes the greedy removal order at the point estimate, precomputes the 21 nested masked
and 21 tail-swapped prediction vectors once on the full test sample, and then a replicate is row
indexing plus rank statistics. It costs 7.6 ms a replicate on both models, because no model call
happens inside a replicate.

The fast path is not an approximation of the package. At the point estimate it reproduces the
package's own curves with maximum absolute gaps of 0.00e+00 / 0.00e+00 / 0.00e+00 (logistic,
RGA / RGE / RGR) and 0.00e+00 / 2.22e-16 / 0.00e+00 (forest).

**Runtime, measured, in seconds:**

| Stage | logistic | forest |
|:--|--:|--:|
| point curves, from the fitted model | 0.1 | 20.5 |
| precompute of the 63 prediction vectors | 0.0 | 7.7 |
| paired bootstrap, B = 2000 | 15.5 | 15.7 |
| independent-stream bootstrap, B = 2000 | 11.1 | 11.0 |
| conditioning check, B = 300 / B = 30 | 26.6 | 787.1 |

Total 898.8 s. **The conditioning check is 91% of it.** Everything else -- both models, both
bootstraps at full B = 2000, the brute-force factorisation checks, all the sensitivities and all
the Monte Carlo error bars -- takes 85 s.

**What we chose, and why.** B = 2000 at the full curve resolution L = 21, with no reduction
anywhere in the primary. There was no reason to reduce either: the fast path makes B = 2000 cost
15 s a model, and L = 21 is fixed by the data rather than chosen. The only deliberate reduction in
the study is the conditioning check, and it is reduced because it is the one arm that cannot use
the fast path. On the logistic model a full re-search costs 0.09 s a replicate, so that arm runs
at B = 300 and its Monte Carlo bands are informative. On the forest it costs 26.2 s a replicate --
B = 2000 there would have been almost fifteen hours -- so that arm stays at B = 30 and is labelled
a gross check. Both are reported as paired comparisons against frozen replicates on the same index
streams, and not as intervals.

**What the conditioning check says.** The two frozen quantities are the greedy removal order and
the perturbation quantiles and masking column means, all taken from the full test sample. The
check re-searches the order and recomputes both on each resample.

| Model | Variant | sd frozen | sd full recompute | ratio | mean absolute paired difference |
|:--|:--|--:|--:|--:|--:|
| logistic | arithmetic | 0.012295 | 0.012578 | 1.023 | 0.004332 |
| logistic | geometric | 0.013198 | 0.013766 | 1.043 | 0.007697 |
| logistic | rms | 0.010671 | 0.010633 | 0.996 | 0.002245 |
| forest | arithmetic | 0.012776 | 0.013081 | 1.024 | 0.004944 |
| forest | geometric | 0.015381 | 0.014929 | 0.971 | 0.007419 |
| forest | rms | 0.010766 | 0.011195 | 1.040 | 0.002898 |

At B = 300 on the logistic model the spread ratios sit within 1.6% of unity -- 1.005 (arithmetic,
95% Monte Carlo band [0.958, 1.054]), 1.016 (geometric, [0.954, 1.084]) and 0.997 (rms,
[0.966, 1.030]) -- so freezing is statistically indistinguishable from a full re-search at the
resolution this arm affords. The forest arm at B = 30 is consistent with the same reading
(ratios 0.971 to 1.040) but its bands are wide ([0.73, 1.27] at the worst), and we call it a
gross check accordingly. Individual replicates do move: the mean absolute paired difference is
0.0027 to 0.0092, which for the geometric variant on the logistic model is about an eighth of an
interval width. So the frozen conditioning is adequate for the interval and would not be adequate
for a claim about any single replicate. We report it that way rather than calling the two schemes
equivalent.

## 11. Weights

We carry exactly one weighted arm and it is marked in the table as a sensitivity check. The
authors have taken a position in print, immediately after eq. (17): "We remark that our
perspective is agnostic, that is, we do not assign weights to the different three dimensions,
reflecting different risk appetites. Rather, we consider all possible combinations of perturbation
types and take simple averages the performance of each model in the three dimensions." A weighted
composite next to their work is therefore not a neutral illustration; it contradicts something
they have stated.

The arm exists only to show that the interval construction is indifferent to weights, and the
profile w = (0.50, 0.20, 0.30) is carried over from the companion aggregation study for
continuity and for no other reason. In the article this belongs in a sentence, not in a table row.
The numbers are in Section 5 for completeness: 0.58575 with interval [0.54689, 0.62863] on the
logistic model and 0.63005 with [0.59332, 0.66982] on the forest.

## 12. What surprised us

**The throughline reversed.** We expected the curve construction to strengthen the cross-covariance
argument, on the reasoning that three curves computed on one test sample are more entangled than
three scalars. The measurement says the opposite: the correlations are smaller, the RGE-RGR pair
has changed sign, and on the forest the independence assumption is conservative rather than
anti-conservative. Section 5 of the article has to be rewritten around this rather than around the
deposited percentages.

**The arithmetic variant cannot see the diagonal question at all.** The identity is exact and it
holds at every replicate to 1e-16. Any discussion of independent versus matched indexing has to be
carried by the other two variants, and if the authors elect the arithmetic mean the whole question
becomes moot for them.

**The perturbation choice outweighs every statistical decision in the study by an order of
magnitude.** Between 2.4 and 3.3 interval widths, in one direction, for a choice the article
specifies and the public package does not implement. The verification of Section 5a then showed
the same choice controls the *sign of the cross-metric covariance* as well: one modelling
decision owns both the level of the composite and the direction in which the independence
assumption errs.

**Freezing the greedy order is nearly free.** A factor of about 3300 in cost on the forest (26.2 s a replicate against 0.008 s on the fast path) for a
spread change that is statistically indistinguishable from zero where we can measure it well
(logistic, B = 300: ratios within 1.6% of unity, inside their Monte Carlo bands).

## 13. Files

| File | Contents |
|:--|:--|
| `pavia_composite_experiment.py` | the whole study, one script, no arguments; `PAVIA_B`, `PAVIA_B_FULL_LOGIT` and `PAVIA_B_FULL_RF` override the bootstrap sizes |
| `results-pavia-composite.json` | every number above, plus the full point curves, the covariance and correlation matrices of the curve summaries, the per-severity-index correlations, the pairwise cross-term decompositions, and the TOPSIS reference vectors |
| `run-pavia-composite.log` | the run of record |
| `verify_cross_terms.py` | adversarial verification of the cross-term sign reversal: independent rewrite, regenerates the replicates, recomputes the covariance and the deposited scalars on the same index streams, and bridges the two constructions estimator by estimator (Section 5a) |
| `verify-cross-terms.log` | the verification run of record |
| `PAVIA-COMPOSITE-SUMMARY.md` | this file |

The study reuses, read-only, the vendored substrate and import stubs of the 24 July redraw run,
and opens the deposited `results.json` read-only for the continuity anchors. It writes nothing
outside this directory.
