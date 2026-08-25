# The second dataset: UCI Default of Credit Card Clients (Taiwan)

Item A11 of `UPGRADE-PLAN-2026-07-27.md`. Run of 28 July 2026.
Pre-registration: `PRE-REGISTRATION.md`, written and timestamped before any
composite, correlation or interval was computed on this dataset.
Anton Sokolov, Tyche Institute, Tallinn.

**Revised 28 July 2026 after an adversarial verification of this study.** No
estimate changed. What changed: two misquoted floating-point residues in section 2
were corrected against the primary run; two design departures that this file had
described as declared in advance are relabelled as not pre-registered, and the
sentence in the pre-registration that made one of them necessary is quoted and
corrected; the claim that the pre-registration prescribed a fallback for the
verdict gap is withdrawn as false and the label is relabelled a post-hoc choice
(sections 1 and 10); the stated reason for the coding invariance in section 9 was
wrong and is replaced with the measured one; a new section 10a tests whether the
design departures manufactured the one result that did not reproduce; a third
recomputation that imports no part of the pinned package was written and is
reported in section 14b; and section 15 drafts the sentences the article should
carry. Every number in this file is now traced to a deposited artefact by
`audit_every_number_in_the_summary.py`.

---

## 1. The result

**The pre-registered verdict is INCONCLUSIVE, and the reason is not that the
mechanism failed.** The mechanism reproduced. On both models, on a dataset with
thirty times the test sample and three more predictors, every rung of the
estimator bridge carries the same sign it carries on German credit: swapping only
the robustness estimator to the tail swap turns the correlation negative
(rung 4: -0.3789 logit, -0.3174 forest), keeping the sweep and changing only the
perturbation family back to Gaussian keeps it positive (rung 5: +0.6062, +0.7530),
and the study pair stays negative (rung 6: -0.5423, -0.2266). The weakest of
those six is 10.7 Monte Carlo standard errors from zero and the strongest is 76.
The two-link chain reproduced almost exactly: the article's 23.6 per cent understatement comes back
as 23.77 per cent under the redraw scheme and 25.08 under the fixed draw, with a
positive cross-link correlation both times.

What did not carry over is the *corollary*. Pre-registered condition (d) required
the delta-method width understatement of the three volume variants to stay within
6 percentage points of zero on both models, which is the article's "declaring the
cross-terms zero barely moves it" claim. On the random forest it holds
comfortably (-4.18, -4.12, -1.52 per cent). On the logistic regression two of
the three breach it: -9.83 (arithmetic) and -13.40 (geometric), with the rms
variant at -5.17 still inside the band. Because the pre-registration made
GENERALISES a conjunction of five conditions, one failing condition removes that
verdict; because the understatement is *negative* -- the zero-cross-term interval
is wider, not narrower, than the measured one -- none of the four failure
triggers fires either, and none of the three named INCONCLUSIVE triggers fires.

**The trichotomy in section 7 of the pre-registration is not exhaustive, it
prescribes no fallback for the gap, and this run landed in the gap.** We say that
plainly rather than dress it up. The label itself is not a post-hoc choice: the
fall-through that emits it was already in `second_dataset_experiment.py`, whose
last edit is timestamped 19 seconds before the run began and 55 minutes before
the deciding numbers existed. What is post hoc is the reading placed on it in
this document -- adopting the INCONCLUSIVE consequence for the magnitude
corollary while writing the mechanism up at the strength conditions (a), (b)
and (c) support. What section 7 actually attaches to the INCONCLUSIVE label is a three-item
trigger list and this consequence -- "We then report every number, decline the
generalisation claim in both directions, and state that the mechanism is bounded
to the conditions under which it was measured." We adopt that consequence for the
*corollary* -- the magnitude claim about the volume composite -- and we do not
adopt it for the *mechanism*, whose three pre-registered conditions (a), (b) and
(c) all held on both models at 10.7 to 76 Monte Carlo standard errors. That
separation is itself post hoc: the pre-registration bundled a mechanism claim and
a magnitude corollary into a single conjunction, and section 8 forbids promoting
anything outside section 6 to decisive but says nothing about splitting a
conjunction after the fact. The reader is entitled to know that the split was
made after the numbers were in, and that under the letter of section 7 nothing at
all may be concluded. Section 10 states which sentence rests on which authority.

Stated without the machinery: **the anti-correlation is a property of the
tail-swap family and it survives a change of dataset, of sample size and of class
balance. The claim that this anti-correlation is numerically immaterial for the
volume composite does not survive as a magnitude claim: on the forest it is
essentially unchanged, but on the logistic model it is five to eleven times larger
than on German credit on the three arms that kept their sign (arithmetic 10.8x,
geometric 7.2x, TOPSIS 5.3x), and the one arm that was positive on German credit,
rms at +1.26 per cent, is -5.17 here.** Its direction, which is the conservative
one, is unchanged everywhere and is now uniform. The chain claim -- the one the
article actually leads with -- survives intact.

---

## 2. What was run

| | German credit | Taiwan |
|:--|--:|--:|
| dataset | OpenML `credit-g` v1 | OpenML id 42477, `default-of-credit-card-clients` v1 |
| n | 1,000 | 30,000 |
| n_train / n_test (stratified 70/30, seed 20260723) | 700 / 300 | 21,000 / 9,000 |
| d | 20 | 23 |
| positive class | y = 1 is "good", the majority, 70.0 per cent | y = 1 is default, the minority, 22.1 per cent |
| positives in test | 210 | 1,991 |
| curve length L = d + 1 | 21 | 24 |
| composite tensor cells | 9,261 | 13,824 |
| tail-swap severity spacing 0.5/d | 0.02500 | 0.02174 |
| B paired / independent | 2000 / 2000 | 2000 / 2000 |
| B_full, conditioning check (logit / rf) | 300 / 30 | 300 / 30 |

Two models, untuned and unchanged: `StandardScaler` + `LogisticRegression(max_iter=2000)`
and `RandomForestClassifier(n_estimators=300)`. Retuning on 21,000 training rows
would have confounded "different dataset" with "different model" and destroyed
the only comparison this exercise exists to make.

**Design departures. Items 1 to 3 were declared in advance; items 4 to 6 were not,
and are declared here.** The pre-registration's section 4 lists three changes and
closes with "4. **Nothing else.**", so anything below item 3 is a departure from
the pre-registration as well as from the German-credit study, and is labelled as
one.

1. L = 24 and GRID = 23 instead of 21 and 20. The rule L = d + 1 is what the
   pinned study fixed, not the number: the greedy removal curve has exactly one
   knot per feature plus the zero-removal anchor, so L follows d and is not free.
2. n_test = 9,000 instead of 300. That is the point of the exercise.
3. The positive class is a minority. Declared, with a coding sensitivity that ran
   unconditionally (section 9) precisely so that it could not serve as a
   post-hoc rescue.
4. **Not pre-registered.** A Monte Carlo standard error for the correlations and
   for the chain understatement, by the same second-level resampling block (400
   resamples of the B replicate rows) that the pinned driver applies to the width
   shares. The pre-registration's section 7 says a sign is called against "twice
   its Monte Carlo standard error, computed by the second-level resampling blocks
   the pinned drivers already carry". That sentence is wrong about the pinned
   drivers: they carry the block for the delta-method width share
   (`delta_share_mc_se`), for the empirical width share (`empirical_under_mc_se`)
   and for the conditioning band, but not for any correlation. The block had to be
   written for this run, so the verdict rule could not have been applied as
   pre-registered without it. It qualifies estimates; it changes none.
5. **Not pre-registered.** The estimator bridge is computed inside the same paired
   loop as the composite rather than in a second script. The index stream is
   identical (`default_rng(SEED + 1)`), so the numbers are the ones a separate
   script produces, and two independent recomputations confirm this (section 14).
6. **Found at run time.** The terminal knot of the accumulated partial RGA curve is
   a structural zero that floating point delivers as a residue, and on Taiwan that
   residue is **negative** on both models: -1.110e-16 for the logistic model and
   -1.110e-16 for the forest. On German credit it was +2.220e-16 (logistic) and
   exactly 0.0 (forest), so the pinned expression evaluated there. The pinned
   geometric brute-force check writes `(a*e*r) ** (1/3)`, and Python's `**` returns
   nan for a negative base at a fractional exponent, so that one check cannot
   evaluate here. The vectorised tensor and the closed form both use numpy's
   `cbrt`, which returns the real cube root, and are unaffected. Both brute-force
   forms are run and both are reported. (The forest residue in the tiny-B preflight
   was -2.220e-16, not -1.110e-16; the two differ by one unit in the last place
   because of the substrate fact of section 11. The driver's stored
   `design_departures` string quotes the preflight value and is left as it ran; the
   value for the primary run is the one above, and it is the one in
   `results-second-dataset.json`.)

Everything else is verbatim: the substrate (`koleso500/safeai` pinned at
`39768fcd5264c881f7174268bbffda52b298ae89`, vendored read-only with the inert
import stubs), SEED = 20260723 and every stream derived from it, alpha = 0.05,
z = 1.959963984540054, the tail swap on the same p-range [0, 0.5], the fixed
Gaussian draw at 0.5 column standard deviations and the Gaussian sweep at
sigma_t = t/d, the three volume variants as the mean of the L x L x L tensor, the
matched-severity diagonal readings, TOPSIS with the thesis ideal and anti-ideal
vectors read at the run's L, the weighted arm at w = (0.50, 0.20, 0.30) still
declared a departure from the source authors, the chain h = C_A * C_B on the
scalar construction, both conditioning senses, and `class_order = [0, 1]`
wherever the pinned drivers pass it.

**The German-credit anchors were re-read from the deposits at run time.** All 42
values frozen in section 6 of the pre-registration reproduced from
`results-pavia-composite.json`, `results-redraw.json` and `verify-cross-terms.log`.
The run aborts before computing anything if they do not.

**Cost.** 3,321.8 s = 55.4 minutes serial, against a projection of 55.8 minutes
plus about 7 for the coding sensitivity. The coding sensitivity (561.6 s) in fact
ran inside the total. No pre-authorised reduction was invoked: B stayed at 2000
and the conditioning check stayed at B_full = 300 and 30. Two things moved in
opposite directions: the forest's conditioning check overran, at 949 s against a
projected 747, while the estimator bridge, which the projection priced as a
separate 1,005-second pass over both models, was absorbed into the paired loop it
shares an index stream with and cost only the two scalar triples.

---

## 3. The estimator bridge: the mechanism, rung by rung

The six rungs walk from the deposited scalar pair to the study pair, one
estimator change at a time. This is the table that isolates the mechanism, and it
is the one to read first.

| rung | construction | German logit | Taiwan logit | German rf | Taiwan rf |
|--:|:--|--:|--:|--:|--:|
| 1 | scalar RGE (bare) x scalar RGR (bare), the deposit pair | +0.5967 | **+0.5028** (mc se 0.0168) | +0.3415 | **+0.4688** (0.0169) |
| 2 | the same with the class-order convention | +0.6144 | **+0.4952** (0.0168) | +0.3437 | **+0.4417** (0.0175) |
| 3 | curve-mean RGE x scalar RGR | +0.4249 | **+0.4490** (0.0189) | +0.2879 | **+0.4930** (0.0164) |
| 4 | scalar RGE x curve-mean tail-swap RGR | -0.2083 | **-0.3789** (0.0186) | -0.1529 | **-0.3174** (0.0200) |
| 5 | curve-mean RGE x Gaussian-sweep RGR mean | +0.5354 | **+0.6062** (0.0144) | +0.4711 | **+0.7530** (0.0099) |
| 6 | curve-mean RGE x curve-mean tail-swap RGR, the study pair | -0.1372 | **-0.5423** (0.0161) | -0.3102 | **-0.2266** (0.0212) |

Twelve of twelve rungs agree in sign with German credit. The attribution argument
survives unchanged and is stronger:

- Rung 3 keeps the RGR estimator and swaps only the RGE estimator to the curve
  mean. The correlation stays positive, on both models and on both datasets.
- Rung 4 keeps the RGE estimator and swaps only the RGR estimator to the
  tail-swap curve mean. The correlation goes negative, on both models and on both
  datasets, and on Taiwan it is roughly twice as negative.
- Rung 5 keeps the sweep structure of rung 6 and changes only the perturbation
  family back to Gaussian. The correlation is positive, on both models and on
  both datasets, and on Taiwan it is larger.

So the sign is attributable to the tail-swap *family*, not to the curve
construction and not to the fact that a sweep is being averaged. That was the
article's claim, and it now rests on two datasets, four fitted models and
n_test = 300 and 9,000.

The two RGA endpoints move the same way. RGA-RGE stays positive through the
change of construction (+0.2093 to +0.2805 on the logistic model, +0.2191 to
+0.2477 on the forest; German credit: +0.2858 to +0.1996 and +0.2922 to +0.1574).
RGA-RGR flips sign as the construction changes, on both models
(+0.2224 to -0.2540 and +0.1315 to -0.1078; German credit: +0.1596 to -0.0583 and
+0.1530 to -0.0800).

---

## 4. Where in the sweep the anti-correlation lives

The article's phrasing is that the anti-correlation appears *deep in the severity
sweep*. That is exactly what the per-severity-index correlations show again.

| index | German logit | Taiwan logit | German rf | Taiwan rf |
|--:|--:|--:|--:|--:|
| 0 | n/a | n/a | n/a | n/a |
| 1 | +0.1418 | -0.1848 | +0.0280 | +0.1255 |
| 2 | +0.1122 | -0.0826 | +0.0434 | +0.1131 |
| 3 | +0.1004 | -0.1564 | -0.0074 | +0.0971 |
| 4 | -0.0097 | -0.0929 | +0.0044 | +0.0168 |
| 5 | -0.0272 | -0.0498 | -0.0337 | +0.0438 |
| 6 | -0.0769 | -0.0884 | -0.0241 | -0.0575 |
| 7 | -0.1039 | -0.0935 | -0.0948 | -0.0567 |
| 8 | -0.1259 | -0.1670 | -0.1432 | -0.1461 |
| 9 | -0.1711 | -0.2207 | -0.1926 | -0.1836 |
| 10 | -0.1436 | -0.2400 | -0.1592 | -0.1443 |
| 11 | -0.1338 | -0.3012 | -0.1147 | -0.1156 |
| 12 | -0.1616 | -0.2579 | -0.1733 | -0.1507 |
| 13 | -0.1408 | -0.2132 | -0.2028 | -0.2078 |
| 14 | -0.1835 | -0.2809 | -0.2127 | -0.1725 |
| 15 | -0.1863 | -0.2895 | -0.2971 | -0.1710 |
| 16 | -0.2221 | -0.2962 | -0.3041 | -0.2204 |
| 17 | -0.2506 | -0.2959 | -0.2536 | -0.3857 |
| 18 | -0.2248 | -0.3638 | -0.5088 | -0.2117 |
| 19 | -0.3132 | -0.3294 | -0.3667 | -0.2088 |
| 20 | -0.0432 | -0.4015 | -0.0347 | -0.1612 |
| 21 | -- | -0.3434 | -- | -0.4074 |
| 22 | -- | -0.4748 | -- | -0.2298 |
| 23 | -- | +0.0111 | -- | -0.0047 |

Index 0 is the zero-severity anchor, where RGE and RGR are the constant 1 and the
correlation is undefined; the run records it as undefined rather than computing
one. The German column ends at 20 because L = 21 there.

Splitting the grid in half:

| | first half | second half |
|:--|--:|--:|
| Taiwan logit, RGE-RGR | -0.1525 | -0.2946 |
| Taiwan rf, RGE-RGR | -0.0280 | -0.2110 |

The forest reproduces the German pattern in its purest form: positive at indices
1 to 5, crossing zero around index 6, and deepening monotonically thereafter. The
logistic model on Taiwan is negative from index 1 onwards and still deepens by a
factor of two across the grid. The mean over the grid is -0.2266 (logit) and
-0.1235 (forest) against German credit's -0.1082 and -0.1524.

---

## 5. The three curves

Both models: the RGA curve anchor equals the scalar `rga_score` to 0.0e+00, and
the accumulated partial curve ends at the structural zero (a residue of order
1e-16, whose replicate standard deviation is 1.3e-16, i.e. it carries no sampling
variation). The RGE curve starts at 1 and ends at 0.5006 (logit) and 0.5006
(forest), the random baseline the construction predicts. The RGR tail-swap curve
starts at 1 and ends at 0.0912 (logit) and 0.3645 (forest).

| | logit | rf |
|:--|--:|--:|
| RGA anchor = scalar RGA | 0.751134 | 0.765361 |
| RGE terminal knot | 0.500642 | 0.500629 |
| RGR terminal knot, p = 0.5 | 0.091162 | 0.364493 |
| first five greedily removed features | x11, x15, x9, x23, x22 | x10, x2, x4, x3, x11 |

Under the UCI dictionary that probe 1 asserted against the measured column
statistics (the ARFF ships anonymous names, x_k being column k - 1), the logistic
model removes PAY_6, BILL_AMT4, PAY_4, PAY_AMT6 and PAY_AMT5 first, and the
forest removes PAY_5, SEX, MARRIAGE, EDUCATION and PAY_6. The removal orders
differ between the two models, as they do on German credit.

The fast bootstrap path reproduces the package curves at the point estimate to
0.0e+00 on the logistic model. On the forest the gap is 1.5e-7; section 11
explains why and shows it is a floating-point floor of the substrate at this
sample size, four orders of magnitude below the narrowest interval reported here.

---

## 6. The volume composite, three variants, with intervals

Point value, the paired bootstrap interval, and the delta-method interval with
the measured covariance against the same interval with the cross-terms declared
zero. The German-credit point values are **not** compared -- different data and
separately fitted models, so no numerical agreement is expected or claimed.

| arm | model | V | paired bootstrap 95% | width | delta 95%, measured covariance | delta 95%, cross-terms zero |
|:--|:--|--:|:--|--:|:--|:--|
| arithmetic | logit | 0.541986 | [0.537812, 0.544829] | 0.007017 | [0.538428, 0.545543] | [0.538078, 0.545893] |
| geometric | logit | 0.394666 | [0.389651, 0.398415] | 0.008764 | [0.390288, 0.399044] | [0.389702, 0.399631] |
| rms | logit | 0.650805 | [0.647634, 0.652922] | 0.005288 | [0.648144, 0.653465] | [0.648007, 0.653603] |
| TOPSIS | logit | 0.304668 | [0.300413, 0.307439] | 0.007026 | [0.301098, 0.308237] | [0.300813, 0.308523] |
| arithmetic | rf | 0.614617 | [0.610384, 0.618848] | 0.008465 | [0.610425, 0.618808] | [0.610250, 0.618983] |
| geometric | rf | 0.530703 | [0.525756, 0.537056] | 0.011299 | [0.525062, 0.536344] | [0.524830, 0.536577] |
| rms | rf | 0.674230 | [0.670909, 0.677246] | 0.006337 | [0.671062, 0.677397] | [0.671014, 0.677445] |
| TOPSIS | rf | 0.370052 | [0.366319, 0.373521] | 0.007202 | [0.366422, 0.373683] | [0.366215, 0.373890] |

The power-mean ordering geometric <= arithmetic <= rms holds at the point
estimate and at 2000 of 2000 replicates for both models.

The intervals are 5.8 to 8.8 times narrower than German credit's, against the
5.48 that the square root of the sample-size ratio would predict; the composite
is a functional of curves, not of a single mean, so no exact rate is expected,
and this is recorded as a description rather than a claim.

---

## 7. The cross-terms

`understatement_of_width_pct` = 100 (1 - w_declared_zero / w_measured). A positive
value means that declaring the cross-terms zero produces a **narrower** interval,
which is the direction that matters for a certificate. A negative value means it
produces a **wider** one.

| arm | model | German delta % | Taiwan delta % (mc se) | German empirical % | Taiwan empirical % (mc se) |
|:--|:--|--:|--:|--:|--:|
| arithmetic | logit | -0.91 | **-9.83** (1.01) | +0.89 | -11.96 (4.27) |
| geometric | logit | -1.86 | **-13.40** (1.60) | +1.45 | -11.05 (3.87) |
| rms | logit | +1.26 | **-5.17** (0.66) | -1.41 | -8.47 (3.56) |
| TOPSIS | logit | -1.50 | **-8.00** (0.85) | -3.10 | -12.29 (3.83) |
| arithmetic | rf | -3.08 | **-4.18** (1.16) | -6.49 | -4.60 (3.04) |
| geometric | rf | -3.09 | **-4.12** (1.10) | -5.21 | -5.27 (3.45) |
| rms | rf | -1.50 | **-1.52** (1.08) | -4.86 | -1.43 (3.38) |
| TOPSIS | rf | -3.11 | **-5.70** (1.15) | -7.56 | -5.53 (3.23) |

Reading it honestly:

- **The direction is unchanged and is now uniform.** On German credit seven of
  eight delta-method arms were negative; on Taiwan all eight are. Declaring the
  cross-terms zero never made a volume interval too narrow on either dataset.
  The volume composite's cross-terms are conservative, and that is a two-dataset
  finding.
- **The magnitude is not portable.** The forest tracks German credit closely
  (-4.18 against -3.08, -4.12 against -3.09, -1.52 against -1.50). The logistic
  model does not: -9.83 and -13.40 against -0.91 and -1.86. The article's word
  "barely" is a magnitude claim, and on the logistic model at n = 9,000 it is
  wrong by an order of magnitude.
- The empirical (independent-stream) column agrees in sign with the delta-method
  column on every Taiwan arm. All four forest values sit inside two Monte Carlo
  standard errors of zero and are reported as unresolved; all four logistic
  values are resolved. The empirical column is a reported quantity, not a
  deciding one.
- The gradient decomposition puts almost all of the cross-covariance on the
  RGA-RGR pair: on the logistic arithmetic arm the three contributions
  2 g_i g_j Sigma_ij are +9.49e-08 (RGA-RGE), -7.00e-07 (RGA-RGR) and -7.42e-08
  (RGE-RGR). The cross share of the variance is -20.63 per cent (mc se 2.23).

---

## 8. The two-link chain

Links A = logistic regression and B = random forest, h = C_A * C_B with C the
geometric mean of the three **scalar** metrics, first-order propagation, shared
index stream `default_rng(SEED + 99)`, exactly as German credit did it.

| quantity | scheme | German | Taiwan | Taiwan mc se |
|:--|:--|--:|--:|--:|
| cross-link correlation | fixed draw | +0.7222 | **+0.7836** | 0.0084 |
| cross-link correlation | redraw | +0.7120 | **+0.7253** | 0.0102 |
| width, measured covariance | fixed draw | 0.079804 | 0.018824 | |
| width, cross-terms zero | fixed draw | 0.060817 | 0.014103 | |
| understatement % | fixed draw | 23.79 | **25.08** | 0.18 |
| width, measured covariance | redraw | 0.079954 | 0.018968 | |
| width, cross-terms zero | redraw | 0.061108 | 0.014460 | |
| understatement % | redraw | 23.57 | **23.77** | 0.22 |

h = 0.769589, from link point values C_A = 0.891076 and C_B = 0.863662.

This is the strongest replication in the study. The article quotes 23.6 per cent;
a different dataset, a different class balance, thirty times the test sample and
separately fitted models return 23.77 per cent under the same conditioning
scheme, with a Monte Carlo standard error of 0.22. The contrast the article is
built on -- cross-terms immaterial for a volume composite, and a quarter of the
interval for a chain -- holds in its chain half without qualification.

The per-model scalar correlations behave as they do on German credit: the redraw
scheme leaves RGA-RGE untouched (it is a deterministic function of the resampled
rows) and pulls RGE-RGR down slightly, from +0.5028 to +0.4149 on the logistic
model and from +0.4688 to +0.4189 on the forest.

---

## 9. The coding sensitivity, run unconditionally

Pre-registered in section 4.3 because Taiwan's positive class is a minority where
German credit's is a majority, and registered to run whatever the primary result
looked like so that it could not function as a rescue. Both models were refitted
on y' = 1 - y and the point curves, the curve-summary covariance and one paired
pass at B = 2000 were recomputed (561.6 s).

| model | primary rung 6 | flipped rung 6 | primary rung 4 | flipped rung 4 | primary RGA-RGR | flipped RGA-RGR |
|:--|--:|--:|--:|--:|--:|--:|
| logit | -0.5423 | **-0.5423** | -0.3789 | **-0.3789** | -0.2540 | -0.2069 |
| rf | -0.2266 | **-0.2266** | -0.3174 | **-0.3174** | -0.1078 | -0.1013 |

Rungs 4 and 6 are invariant to the coding: on the logistic model rung 6 agrees to
1.7e-14 and rung 4 to 9.1e-15, and on the forest rung 6 agrees to 2.75e-07 and
rung 4 to 7.41e-06 -- the substrate floor of section 11, not a residual dependence
on the coding. That is not an accident, but the reason is narrower than it first
looks and the article must state it exactly.

RGE and RGR do not consume y. Relabelling maps each fitted model's predicted
probability to its complement, p -> 1 - p. That alone does **not** make a
CvM-over-Gini functional invariant: the Lorenz curve and the Gini of 1 - p are not
those of p, and on a synthetic 500-point pair the bare one-dimensional score moves
by 4.5e-04 under the complement. What makes rungs 4 and 6 invariant is the
convention the pinned drivers pass, `class_order = [0, 1]`. Under it the score is
the mean of the one-versus-rest scores on the two probability columns
[1 - p, p], and relabelling swaps those two columns, so the mean is unchanged to
machine precision. The invariance is therefore a property of the two-column
convention, not of rank functionals in general: rungs 1 and 3, which use the bare
one-dimensional convention, carry no such guarantee, and the study does not claim
one for them. Only the RGA-involving correlations move, and they move by two to
five hundredths without changing sign. The pre-registered INCONCLUSIVE trigger
"the coding sensitivity flips the sign" therefore does not fire, and the
minority-class concern is answered by measurement rather than by argument.

**One gap in the sensitivity, declared.** The pre-registration's section 4.3
promised "the point curves, the curve-summary covariance and one paired pass"
under the flipped coding. The deposit records the correlation matrix of the curve
summaries but not their covariance, and the flipped replicate matrices were not
kept, so the one pre-registered quantity that actually failed -- the delta-method
width understatement of condition (d) -- is the one quantity for which no
flipped-coding recomputation exists. What can be said from what was kept: applying
the flipped correlation matrix to the primary run's summary standard deviations
gives -8.08 per cent (logistic, arithmetic) and -3.75 per cent (forest), against
-9.83 and -4.18 under the primary coding. The breach of condition (d) is therefore
not an artefact of the class coding. That is an indication built from a stored
correlation matrix and a borrowed standard-deviation mix; it is not the paired
recomputation section 4.3 asked for, and it is not treated as one.

The composite point values do move, as they must, because RGA does consume y:
the arithmetic volume goes from 0.541986 to 0.565204 (logit) and from 0.614617 to
0.651621 (forest).

---

## 10. The pre-registered verdict, applied

Rules copied from section 7 of the pre-registration and evaluated in code, not by
hand. A sign is called only where its absolute value exceeds twice its Monte
Carlo standard error.

**GENERALISES requires all five:**

| | condition | holds? |
|:--|:--|:--|
| (a) | bridge rung 4 negative on both models | **yes** |
| (b) | bridge rung 5 positive on both models | **yes** |
| (c) | bridge rung 6 negative on both models | **yes** |
| (d) | delta-method understatement of the three volume variants within 6 pp of zero on both models | **no** (logit: -9.83, -13.40, -5.17) |
| (e) | chain understatement >= 10 pp with positive cross-link correlation under both schemes | **yes** |

**FAILS TO GENERALISE requires any one of:**

| | condition | holds? |
|:--|:--|:--|
| (a') | rung 4 positive on both models | no |
| (b') | rung 6 positive on both models | no |
| (c') | volume delta understatement exceeds +15 pp on both models | no |
| (d') | chain understatement below 5 pp, or sign-changed, under both schemes | no |

**INCONCLUSIVE triggers:**

| trigger | fires? |
|:--|:--|
| a deciding quantity inside two Monte Carlo standard errors of zero | no -- the list is empty |
| the two models disagree in sign on rung 4 | no |
| the two models disagree in sign on rung 6 | no |
| the coding sensitivity flips rung 6 | no |
| the coding sensitivity flips rung 4 | no |

Neither the GENERALISES conjunction nor any FAILS disjunct nor any named
INCONCLUSIVE trigger holds. **The recorded verdict is INCONCLUSIVE, and the label
is a post-hoc conservative choice, not a pre-registered one.** Section 7 defines
INCONCLUSIVE by three triggers and none of them fired; it provides no fallback for
an outcome that satisfies no branch. We chose the label that declines the claim
rather than the one that flatters the article, and we record that the choice was
made after the numbers were seen. We do not add a third dataset to break the tie;
that is plan item B4 and it belongs to the sibling.

On reading (c'), "the volume-composite delta understatement exceeds +15
percentage points on both models, that is, the cross-terms become material where
they were immaterial": the run evaluates it as the largest of the three variants
exceeding +15 on each model. Two other readings are available and both are also
false here. Under "all three variants exceed +15 on each model" it is false a
fortiori. Under a magnitude reading -- absolute value above 15, which is what the
gloss "become material" suggests -- the worst Taiwan arm is the logistic geometric
at -13.40, which is 1.6 percentage points short of the trigger, and the worst
forest arm is -5.70, so the conjunction over both models fails on the forest by a
wide margin. The choice of reading does not affect the verdict, but the magnitude
reading is the one that came closest to firing and it is recorded here so that a
reader does not have to reconstruct it.

What is now safe to write, and what is not:

- **Safe.** The tail-swap family anti-correlates the robustness curve with the
  explainability curve; this is a property of the perturbation family and not of
  the curve construction or of the sweep; it reproduces on two datasets, four
  fitted models, n_test = 300 and 9,000, a majority and a minority positive
  class, and 20 and 23 predictors; it deepens with severity; and the two-link
  chain's interval is understated by roughly a quarter when the cross-terms are
  declared zero, on both datasets and under both conditioning schemes.
- **Not safe.** That the cross-terms are numerically negligible for the volume
  composite. Their direction is stable and conservative on both datasets, but
  their magnitude ranges from 0.9 to 13.4 per cent across two datasets and two
  models, and the article's current single-dataset phrasing understates that
  range by an order of magnitude.
- **Which sentence rests on which authority.** The "safe" bullet is the
  pre-registered GENERALISES sentence for conditions (a), (b) and (c), all of
  which held. It is written under a verdict label that is not GENERALISES, so it
  must be attributed to the three conditions that held and not to the verdict. The
  "not safe" bullet is the pre-registered INCONCLUSIVE consequence applied to
  condition (d), which failed. No sentence in this study is authorised by the
  verdict label itself.

---

## 10a. Is the comparison fair? Three probes on the design differences

The pre-registration named two structural differences (the minority positive class
and the column-type mix) as the leading candidate explanations if the mechanism did
not reproduce. It did reproduce, so the burden moved: the difference that has to be
explained is the *corollary*, the eightfold to sevenfold jump in the volume
composite's cross-term cost on the logistic model. Three probes ask whether a
design choice manufactured that jump. All three are recomputed from the deposited
replicate matrices, so none of them needs a rerun.

**Probe A, the grid length.** L = 24 instead of 21 is the largest declared
departure. Recomputing the Taiwan curve means from the first 21 knots only, and
separately from a 21-knot subgrid spanning the whole severity range, moves the
arithmetic understatement from -9.83 to -9.63 and -9.67 on the logistic model, and
from -4.18 to -3.54 and -4.28 on the forest. The grid length moves it by at most
0.6 percentage points, against a 9-point gap from German credit. **L is not the
explanation.**

**Probe B, correlations against the standard-deviation mix.** The understatement
is scale-free: it depends only on the correlation matrix of the three curve
summaries and on their relative standard deviations. Crossing the two datasets:

| arithmetic arm | logistic | forest |
|:--|--:|--:|
| German correlations, German sd mix (as reported) | -0.91 | -3.08 |
| Taiwan correlations, Taiwan sd mix (as reported) | -9.83 | -4.18 |
| **Taiwan correlations on the German sd mix** | **-8.27** | -3.31 |
| **German correlations on the Taiwan sd mix** | **-1.44** | -4.85 |

On the logistic model the correlations carry the whole move and the standard
deviations carry almost none. **The corollary failed because the mechanism got
stronger, not because a design knob changed.** On the forest the two channels are
comparable, and the forest's understatement barely moved anyway.

**Probe C, why "barely" was true on German credit.** The three cross contributions
2 g_i g_j Sigma_ij do not have the same sign. On German credit's logistic arm the
RGA-RGE contribution is +5.04e-06 and the other two sum to -8.84e-06, so the net is
27 per cent of the gross: the smallness is a *near-cancellation*. On Taiwan's
logistic arm the positive contribution nearly vanishes and the net is 78 per cent
of the gross. The article's "barely moves it" was never a statement that the
cross-terms are small; it was a statement that on one dataset they happened to
cancel. That is the single most important thing this second dataset established
about the volume half of the claim, and it should be in the article.

One difference the probes cannot dispose of: the models are fitted on 700 rows on
German credit and on 21,000 here, and the test sets differ by a factor of thirty.
Both are inherent to "a second, larger dataset" and neither can be varied without
abandoning the comparison the exercise exists to make.

---

## 11. Checks, identities and one substrate fact

**Numerical identities.** The vectorised L x L x L tensor equals the explicit
triple sum to 1.8e-15 (arithmetic), 6.7e-16 (geometric, real cube root) and
1.4e-15 (rms) on the logistic model, and comparably on the forest. The arithmetic
and geometric closed forms reproduce the tensor; the rms variant admits neither
of the two natural candidate closed forms, off by more than 1e-4 in every case.
The Euler identity for the rms gradient, ga.a + ge.e + gr.r = V, closes to
1.1e-16 (logit) and 0.0 (forest).

**The deterministic knots are not the mechanism.** Deleting the one deterministic
knot from each curve -- RGA's terminal zero, RGE's and RGR's unit anchors --
changes every curve-summary correlation by at most 9.4e-15 (logit) and 4.7e-15
(forest), and leaves the arithmetic cross share at -20.6288 per cent either way.
The replicate standard deviations of those knots are 1.3e-16, 0.0 and 0.0.

**Conditioning, sense (b): freezing the greedy removal order.** The paired
bootstrap re-searches nothing; the check re-searches the greedy order and rebuilds
the perturbation on the resampled rows, at B_full = 300 (logit, 1.06 s a
replicate) and 30 (forest, 31.6 s a replicate, and labelled a gross check).

| model | arm | sd frozen | sd full recompute | ratio | 95% Monte Carlo band | German ratio |
|:--|:--|--:|--:|--:|:--|--:|
| logit | arithmetic | 0.001956 | 0.001979 | 1.012 | [0.966, 1.057] | 1.005 |
| logit | geometric | 0.002389 | 0.002454 | 1.027 | [0.954, 1.106] | 1.016 |
| logit | rms | 0.001454 | 0.001553 | 1.068 | [1.029, 1.104] | 0.997 |
| rf | arithmetic | 0.002183 | 0.002442 | 1.119 | [1.009, 1.255] | 1.024 |
| rf | geometric | 0.003028 | 0.003221 | 1.064 | [0.974, 1.170] | 0.971 |
| rf | rms | 0.001626 | 0.001923 | 1.183 | [1.056, 1.317] | 1.040 |

The ratios are above 1 in every arm, and the band excludes 1 on the logistic rms
arm and on two of the three forest arms. Freezing the greedy order is therefore *mildly
anti-conservative* at n = 9,000 -- it understates the standard deviation by
roughly 1 to 18 per cent -- where on German credit the ratios straddled 1. This
is a new observation, it is not one of the pre-registered comparisons, and it
should be reported as a limitation of the fast path rather than folded into any
headline.

**Conditioning, sense (a): the fixed perturbation draw against a per-replicate
redraw.** Reported in section 8 for the chain, where it moves the understatement
from 25.08 to 23.77 per cent, and per model for the scalar correlations.

**Truncation and indexing sensitivities.** The two length-d readings shift the
composite in opposite directions on both models, exactly as on German credit; the
spread between them is 3.3 to 5.0 paired-bootstrap interval widths (German
credit's spread was of the same character but the widths were larger). The
matched-severity diagonal differs from the product space by +0.0277 (geometric,
logit) and -0.0053 (rms, logit), both excluding zero; the arithmetic diagonal
equals the product space at every replicate, to 2.2e-16. TOPSIS at the two
length-d readings spreads by 0.0197 (logit) and 0.0171 (forest), 2.8 and 2.4
interval widths.

**The Gaussian-sweep RGR arm** raises the composite substantially, by +0.231
(arithmetic, logit) and +0.118 (arithmetic, forest), against German credit's
+0.188 and +0.143 -- the tail swap is much the harsher perturbation on both
datasets.

**TOPSIS column normalisation** (the thesis's step 2) shifts the closeness
coefficient by -0.0183 (logit) and -0.0018 (forest) on the decision matrix of two
models plus the two reference vectors, and destroys it entirely on a matrix of
models alone, where the structurally zero terminal RGA column has no meaningful
norm. That degeneracy is the same one German credit exhibits.

**One substrate fact, measured rather than assumed
(`probe_05_rf_determinism.py`, log in `probe-05-rf-determinism.log`).** The
300-tree forest's `predict_proba` is not bitwise reproducible: two calls on one
fitted forest differ by 2.220e-16, and so, to within the same order, do a second
fit with the same `random_state` (2.220e-16) and a switch between `n_jobs=1` and
`n_jobs=-1` (1.110e-16 to 2.220e-16 across repetitions of the probe). At
n = 9,000 the forest's probabilities are extremely
tie-dense -- 442 distinct values among 9,000 rows, and 8,558 of the 8,999
adjacent sorted pairs closer than 1e-12 -- so a 2.2e-16 perturbation reorders
many near-tied pairs and moves a rank functional by of the order of 1e-7. That is
precisely the size of the forest's fast-path gap (1.5e-7) and of the gap between
the primary run and the independent recomputation (1.4e-7). It is a floor of the
substrate at this sample size, not a defect of either script, and it is four
orders of magnitude below the narrowest interval width reported here (5.3e-3).
The logistic model, whose probabilities are continuous, reproduces exactly.

---

## 12. What this changes in the article

1. **Keep the mechanism claim and widen its base; keep a caveat, but a different
   one.** The "bounded to German credit" caveat may be dropped for the
   anti-correlation itself and must be replaced, not deleted: two datasets, four
   fitted models, n_test = 300 and 9,000, d = 20 and 23, a 70 per cent majority
   positive class and a 22 per cent minority one, the same sign on all twelve
   bridge rungs -- and still one perturbation family, one domain (tabular consumer
   credit), two model classes and no third dataset. The caveat the article should
   now carry is that it has not been tested outside tabular credit data or outside
   the tail-swap family, not that it rests on one dataset.
2. **Keep the chain number and add the second one.** 23.6 per cent on German
   credit and 23.77 per cent on Taiwan, both under the redraw scheme, both with a
   positive cross-link correlation.
3. **Rewrite the volume-composite sentence.** "Barely moves it" is a
   single-dataset, single-model magnitude claim. The defensible statement is that
   the cross-terms move the volume interval in the conservative direction on both
   datasets and in all sixteen arms measured, by between 0.9 and 13.4 per cent,
   and that the size of the effect is not portable across datasets or models
   while its direction is.
4. **Report the verdict as pre-registered, and report the gap as a gap.** State
   that four of the five GENERALISES conditions held, name the one that did not,
   and say that the pre-registered trichotomy was not exhaustive, prescribed no
   fallback, and that INCONCLUSIVE was chosen after the fact as the conservative
   label. Do not present it as a rule that was followed.
5. **Add the coding-invariance observation, with its actual reason.** Rungs 4 and
   6 are invariant to relabelling to machine precision on the logistic model and
   to the substrate floor on the forest. The reason is the two-column
   `class_order = [0, 1]` convention -- relabelling swaps the two probability
   columns and the mean over them is unchanged -- and not the more general claim
   that a rank functional is blind to the complement, which is false: the bare
   one-dimensional score moves under p -> 1 - p. Stated correctly this answers the
   "but your positive class flipped" objection; stated loosely it invites a
   referee to falsify it in one line.
6. **Add the frozen-order caveat, with its band.** At n = 9,000 the frozen greedy
   order gives a standard-deviation ratio above 1 in all six arms, and the Monte
   Carlo band excludes 1 in three of the six (logistic rms, forest arithmetic,
   forest rms). The point estimates are consistently anti-conservative; the
   evidence is resolved in half the arms and unresolved in the other half. The
   fast path is a convenience, not an identity.
7. **Say why "barely" was true on German credit.** It was a near-cancellation
   between a positive RGA-RGE cross term and two negative ones (section 10a, probe
   C), not a smallness of the cross-terms. This is the most transportable thing
   the second dataset established about the volume half of the claim.
8. **Attribute the volume-composite move to the correlations, not to the design.**
   Probe A shows the grid length moves it by at most 0.6 percentage points; probe B
   shows that on the logistic model the Taiwan correlation matrix on the German
   standard-deviation mix already produces -8.27 per cent. Condition (d) failed
   because conditions (a) to (c) succeeded more strongly, and the article should
   say so rather than leave the two facts in separate paragraphs.

---

## 13. Reproduction

| file | what it is |
|:--|:--|
| `PRE-REGISTRATION.md` | the commitments, written before any result |
| `second_dataset_experiment.py` | the primary driver |
| `results-second-dataset.json` | every number this summary quotes |
| `run-second-dataset.log`, `main-run-stdout.txt` | the primary run log |
| `verify_second_dataset.py` | the second recomputation: shares no code with the driver, but imports the same pinned package for the estimators |
| `verify-second-dataset.json`, `verify-second-dataset.log`, `verify-run-stdout.txt` | its results |
| `verify-run-stdout-attempt1-aborted.txt` | the aborted first verification attempt, kept as required |
| `verify2_from_definition_curves.py`, `verify2-from-definition-curves.log`, `verify2-from-definition-curves.json` | the third recomputation, logistic curve pass: imports no safeai, rebuilds every estimator from the definition |
| `verify2_from_definition_chain.py`, `verify2-from-definition-chain.log`, `verify2-from-definition-chain.json` | the third recomputation, the chain, both schemes |
| `probe_06_comparability_of_the_two_studies.py`, `probe-06-comparability-of-the-two-studies.log` | the comparability probes of section 10a and the two convention checks of section 9, recomputed from the deposited replicate matrices |
| `audit_every_number_in_the_summary.py`, `audit-every-number-in-the-summary.log` | the fabrication screen: every numeric literal in this file traced back to a deposited artefact |
| `replicates-second-dataset.npz` | the replicate matrices, so error bars can be recomputed without a rerun |
| `make_tables.py`, `tables.md` | the tables above, rendered from the results file |
| `probe_01_data.py` .. `probe-04-comparability.log` | the four design probes |
| `probe_05_rf_determinism.py`, `probe-05-rf-determinism.log` | the forest reproducibility measurement of section 11 |
| `prereg-number-audit.log` | the machine audit of the frozen German-credit anchors |
| `smoke-run-stdout-preflight.txt`, `results-second-dataset-SMOKE.json`, `run-second-dataset-SMOKE.log` | the tiny-B preflight that exercised every code path and found the negative-residue nan |

Substrate: `koleso500/safeai` at `39768fcd5264c881f7174268bbffda52b298ae89`,
vendored read-only at `../redraw-2026-07-24/safeai-src` with the inert stubs
alongside. Interpreter `/srv/tyche/repos/safeai-fork/.venv-test/bin/python`:
Python 3.12.3, numpy 2.5.1, scikit-learn 1.9.0, on zeus1, 24 threads. The data
load requires no network -- probe 3 fetched it with the socket layer deliberately
disabled, from the local scikit-learn OpenML cache.

Nothing under `../redraw-2026-07-24/` or `../pavia-composite-2026-07-25/` was
written to. Both were opened read-only, once, to read back the German-credit
anchors; all 42 frozen values reproduced.

---

## 14. Independent recomputation

There are two of them, and the second was added because the first has a blind
spot. `verify_second_dataset.py` shares no code with the driver, but it imports
the same pinned `safeai` package for RGA, RGE and RGR, so it can only verify how
the estimators were orchestrated, not what they compute. Section 14b closes that
gap.

### 14a. The second recomputation, through the same package

`verify_second_dataset.py` regenerates the index streams from the documented
recipe, rebuilds the tail swap from the definition with a different
implementation (an index permutation rather than a fancy-index assignment),
recomputes the point curves from the package, regenerates all 2000 paired and
2000 independent-stream replicates, recomputes the covariance of the three curve
summaries and each delta-method understatement by hand, recomputes the deposited
scalar estimators on the same streams, rebuilds the six-rung bridge, re-derives
the per-severity-index correlations, and re-runs the chain under both
conditioning schemes.

**All 48 recomputed quantities agree with the primary run inside their declared
tolerance.** Eighteen agree exactly. Every logistic-model quantity agrees to
2.3e-14 or better, including all six bridge rungs, all three curve-summary
correlations, all three delta-method understatements, both empirical
understatements, the full 24-knot per-severity-index vector (gap 0.0e+00) and the
covariance matrix of the curve summaries (gap below 1e-14). The forest's
quantities agree to between 0.0e+00 and 1.6e-05; the largest gaps are

| quantity | independent recomputation | primary run | gap |
|:--|--:|--:|--:|
| P4 empirical understatement, geometric, rf | -5.26763239 % | -5.26764821 % | 1.6e-05 pp |
| per-severity-index RGE-RGR, rf, worst knot | | | 1.6e-05 |
| P5 chain understatement, redraw | 23.76866979 % | 23.76868459 % | 1.5e-05 pp |
| P6 bridge rung 4, rf | -0.31743663 | -0.31743589 | 7.4e-07 |

and every one of them is the floating-point floor of section 11 propagated
through 2000 replicates, not a disagreement about anything. The chain, the six
rungs, the curve-summary correlations and the delta-method understatements are
reproduced by a second implementation that shares no code with the driver: it
rebuilds the tail swap from the definition as an index permutation rather than a
fancy-index assignment, redraws every index stream from the seed recipe, refits
both models, and re-derives each interval by hand.

The verification tolerances were fixed from the measured substrate property of
section 11 -- exact for the logistic model, 1e-6 on a forest curve knot, 1e-3 on
a forest correlation and 1e-2 of a percentage point on a forest width share --
and not from the gaps the script then found. Every gap is stored in
`verify-second-dataset.json` regardless of the pass flag, so the tolerance decides
a label and not a claim.

### 14b. The third recomputation, from the definitions, importing no safeai

`verify2_from_definition_curves.py` and `verify2_from_definition_chain.py` were
written during the adversarial verification of this study and import no part of
the pinned package. They rebuild the Lorenz-Gini coefficient, the weighted
Cramer-von Mises distance, the three scores in both the bare and the
`class_order = [0, 1]` conventions, the RGA partial curve, the RGE greedy removal
search and the tail swap -- the last as a source-index gather rather than a
fancy-index assignment -- from the published definitions, redraw every index
stream from the seed recipe, and refit both models.

**Curve pass, logistic model: 34 of 34 checks agree, and 33 of them agree at
0.0e+00.** That includes the three point curves, the greedy removal order, all six
replicate matrices element by element (`Ab`, `Eb`, `Rb`, `Rb_noise`, `SC`,
`SC_co`, each 2000 x 24 or 2000 x 3, maximum absolute gap 0.0e+00), the full 3 x 3
curve-summary correlation matrix and its covariance, all six bridge rungs, both
RGA endpoints, the whole 24-knot per-severity-index vector, and all four
delta-method understatements with their cross shares. The single non-zero gap is
2.3e-14 on the geometric understatement, from the order of two floating-point
sums.

**Chain, both schemes: 19 of 19 checks agree.** Every logistic-model quantity is
exact; the chain quantities, which involve the forest, agree to 1.1e-08 (fixed
draw rho), 2.5e-07 (redraw rho) and 4.8e-06 percentage points (redraw
understatement). Those are the substrate floor of section 11, four orders of
magnitude below the 0.22-point Monte Carlo standard error of the estimate itself.
The redraw understatement is 23.7686845882 in the primary run and 23.7686797684
here, on the same machine with the same seeds. That is a direct demonstration of
probe 5's finding: the forest is not bitwise reproducible, so any digit beyond the
seventh on a forest-involving quantity belongs to the machine and not to the study,
and the article must not quote one. The study's own reported figure, 23.77 per
cent with a Monte Carlo standard error of 0.22, is four orders of magnitude
coarser than that floor and is unaffected.

The headline numbers of this study have therefore been computed three times: once
by the driver, once by an independent orchestration over the same package, and once
by an implementation that shares neither code nor package with either.

The first verification attempt aborted on an assertion that held the forest to
exact agreement. That abort is what prompted `probe-05-rf-determinism.log`, which
established that the forest's `predict_proba` is not bitwise reproducible at all
and that its probabilities at n = 9,000 are tie-dense enough to turn a 2.2e-16
perturbation into a 1e-7 move in a rank functional. The aborted log is kept as
`verify-run-stdout-attempt1-aborted.txt`, as the pre-registration's section 8
requires. No estimate was changed; only the tolerance that decides whether two
runs are called equal.


---

## 15. The sentences the article should carry

Drafted at exactly the strength two datasets support, and no further. The first
three are for the generality passage; the fourth replaces the caveat.

> The anti-correlation that the tail-swap family induces between the robustness
> curve and the accuracy and explainability curves is not a property of German
> credit. It reproduces on the UCI Default of Credit Card Clients data at thirty
> times the test sample and three more predictors, with the same sign on all
> twelve rungs of the estimator bridge across the two datasets and the two model
> classes, and with the same attribution: swapping only the robustness estimator to
> the tail swap turns the correlation negative on both datasets and both models,
> while keeping the sweep and restoring the Gaussian family leaves it positive.
> The consequence this article draws for a chain reproduces almost exactly --
> declaring the cross-terms zero understates a two-link interval by 23.57 per cent
> on German credit and by 23.77 per cent on the second dataset under the same
> conditioning scheme, with a positive cross-link correlation in both.
> What does not transport is the size of the corollary. The cross-terms move a
> volume composite's interval in the conservative direction in all sixteen arms we
> measure across the two datasets, but by between 0.9 and 13.4 per cent; the
> near-zero figure on German credit turns out to be a cancellation between cross
> contributions of opposite sign rather than a smallness of the cross-terms, so we
> report the direction as a two-dataset finding and the magnitude as specific to a
> dataset and a model.
> These results were obtained on two tabular consumer-credit datasets, two model
> classes and one perturbation family; we have not tested another domain, another
> perturbation family, or a third dataset.

**May the article drop its "bounded to German credit" caveat?** For the
anti-correlation itself and for the chain number, yes: those are the two claims
that reproduced, and the caveat as written is now false. For the volume composite,
no -- and the caveat there has to get stronger, not weaker, because the word
"barely" is a magnitude claim that a second dataset falsified by an order of
magnitude on one of the two models. The scope sentence must not simply be deleted:
"bounded to German credit" should be replaced by "two tabular consumer-credit
datasets, two model classes, one perturbation family", which is what was actually
tested. Deleting it outright would be the one move these two datasets do not
license.
