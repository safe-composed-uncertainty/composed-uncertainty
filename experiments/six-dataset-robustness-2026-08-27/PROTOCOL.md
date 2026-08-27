# Protocol: six-dataset credit robustness extension

**Frozen:** 27 August 2026, before any RGA, RGE, RGR, correlation,
composite, interval, or model-performance result was computed on the six
datasets below.

**Status:** a prospectively specified extension to six new datasets, but a
post hoc extension relative to the German-credit and Taiwan studies. It is not
part of the Taiwan pre-registration. The Taiwan protocol explicitly assigns a
third-dataset factorial to a sibling article; nothing here changes that record.

Only source metadata, target semantics, dimensions, class counts, missingness,
duplicate counts, file checksums, data types, and download/runtime feasibility
were inspected before this freeze. No SAFE metric or model outcome was viewed.

## 1. Question and claim boundary

The primary question is whether the contrast between the article's tail-swap
robustness family and its Gaussian-noise control has the same direction across
six additional public tabular consumer-credit cohorts.

The dataset is the replication unit. Logistic regression and random forest are
two sensitivity arms within each dataset, not twelve independent replications.
Results will support only a finite-set statement about these six snapshots,
under this representation, split, seed, model pair, and perturbation design.
They will not establish generality across credit data, interval calibration,
legal or compliance validity, or comparability of absolute compliance-score
levels across datasets.

## 2. Locked intended set

The intended set is fixed in this order. OpenML data ids and versions identify
the exact machine-readable snapshots. Raw files are fetched at run time and are
not redistributed.

1. **Statlog Australian Credit Approval**, OpenML 40981 v4, corresponding to
   UCI 143 (DOI `10.24432/C59012`): 690 rows, 14 predictors. The released
   target is approval; it is inverted so `y=1` means denial/adverse outcome.
2. **SAS HMEQ Home Equity**, OpenML 43337 v2: 5,960 rows, 12 predictors;
   `BAD=1` is default or serious delinquency.
3. **FICO HELOC cleaned**, OpenML 45554 v2: 9,871 rows, 23 predictors;
   `RiskPerformance=Bad` is adverse. The snapshot is publicly fetchable, but
   its redistribution licence is custom/unknown, so no raw copy is deposited.
4. **Give Me Some Credit**, OpenML 45577 v1: 150,000 rows, 10 predictors;
   `SeriousDlqin2yrs=1` is 90-days-past-due or worse within two years. It is
   publicly fetchable but has no express open redistribution licence in the
   OpenML record, so no raw copy is deposited.
5. **Lending Club Loan Data**, OpenML 43729 v1: 9,578 rows, 13 predictors;
   `not.fully.paid=1` is adverse.
6. **Credit Risk Dataset**, OpenML 43454 v1: 32,581 rows, 11 predictors;
   `loan_status=1` is default. OpenML marks the snapshot CC0, but its upstream
   provenance is weak; it is described only as a public benchmark cohort, not
   as verified bank data.

These are six distinct cohorts. UCI Credit Approval ids 27/28 and other
Australian variants are excluded because they reproduce the same 690 outcomes.
South German Credit is excluded because it is a corrected representation of
the German cohort already used. Taiwan is excluded because it is the existing
second-dataset study. Credit-card fraud and bank-marketing datasets are excluded
because fraud and term-deposit subscription are not credit approval or repayment
outcomes. Corporate-bankruptcy datasets are outside the primary consumer-credit
stratum.

### Fallbacks

Fallbacks may be used only for a download or schema failure discovered before
any SAFE metric is computed for the intended set, and in this order:

1. Polish Companies Bankruptcy, UCI 365 / OpenML 46950 v1, explicitly labelled
   a separate corporate-distress stratum.
2. SBA Loans Case, OpenML 43539 v1, only after removing the predeclared leakage
   fields `MIS_Status`, `ChgOffDate`, `ChgOffPrinGr`, and `BalanceGross`, plus
   identifiers.

Once any RGA/RGE/RGR result exists, there are no substitutions. A failed,
timed-out, degenerate, or non-finite run remains in the intended-set denominator
and is reported as such.

## 3. Sampling, split, and preprocessing

- Raw rows and duplicates are preserved. Counts are reported.
- To bound the computation without outcome-dependent choices, datasets above
  30,000 rows are stratified-sampled without replacement to exactly 30,000.
  Dataset `k` in the locked order uses `default_rng(20260723 + 5000 + k)`.
- The retained rows are split 70/30 with `train_test_split`, stratified by the
  adverse target and `random_state=20260723`, as in the existing studies.
- The split occurs before preprocessing. All fitted preprocessing uses training
  rows only.
- Numeric missing values receive the training median. Categorical missing
  values receive the training mode, and categorical variables are one-hot
  encoded with all training levels retained and unseen test levels ignored.
  This standard adapter is fixed for all six datasets.
- Transformed columns with zero variance on the training split are removed.
  Raw and encoded dimensions, categories, imputation counts, unseen-category
  counts, and processed train/test hashes are recorded.
- Identifiers and the asserted target column are excluded. No predictor is
  removed using its association with the target.

One-hot encoding changes the feature count, the `d+1` grid, and the semantic
unit of greedy removal for mixed-data datasets. Those runs are therefore an
adapted mixed-data stratum, not an unchanged-estimand pooled replication of the
numeric German/Taiwan runs. This limitation is reported with the results.

## 4. Frozen computation

The substrate is `koleso500/safeai` commit
`39768fcd5264c881f7174268bbffda52b298ae89`.

- Master seed: `20260723`.
- Models: `StandardScaler` plus
  `LogisticRegression(max_iter=2000, random_state=20260723)` and
  `RandomForestClassifier(n_estimators=300, random_state=20260723,
  n_jobs=1)`. Single-threaded forests remove the tie-sensitive parallel
  nondeterminism observed in the Taiwan audit.
- Curve grid: `t/d`, `t=0,...,d`; curve length `L=d+1` after preprocessing.
- RGA: accumulated partial curve, with the package settings used in the paper.
- RGE: greedy masking, mean baseline, and package-selected removal order.
- Tail-swap RGR: the article's columnwise tail swap at
  `p_t=0.5*t/d`.
- Gaussian-control RGR: one deterministic sweep per model at
  `sigma_t=t/d` times the test-column standard deviation, generated by
  `default_rng(20260723)` as in the Taiwan driver.
- Paired nonparametric bootstrap: `B=2000`, the same resampled test-row indices
  for every curve and both perturbation families, drawn from
  `default_rng(20260724)` separately within each dataset/model.
- Interval level: 95%, percentile endpoints 2.5 and 97.5%; delta calculations
  use `z=1.959963984540054`.
- The greedy removal order and perturbation predictions are frozen at the
  observed test set in the primary bootstrap. Intervals are conditional on
  those fitted/frozen quantities.
- A full-research conditioning check recomputes the greedy order and tail swap
  on resampled rows for the first 100 logistic and first 30 forest replicates
  per dataset. This is a sensitivity check, not a replacement primary interval.
- The adverse-label complement is an unconditional sensitivity calculation;
  it cannot replace the primary adverse=`1` result.

The Python environment is locked in `requirements-six-dataset.txt`. Every run
records Python and library versions, the safeai commit, input checksums, seeds,
configuration, start/end UTC times, and failures.

## 5. Primary endpoint fixed before outcomes

For bootstrap replicate `b`, let `E_b` be the mean of the RGE curve, `T_b` the
mean of the tail-swap RGR curve, and `G_b` the mean of the Gaussian-control RGR
curve. For each dataset/model arm compute

`Delta = atanh(corr(E,T)) - atanh(corr(E,G))`,

with correlations clipped to `[-1+1e-12, 1-1e-12]` only for the Fisher
transform. The prespecified direction is `Delta < 0`. The two correlations and
Delta use identical bootstrap rows.

A dataset is **directionally concordant** only if both model arms have
`Delta < 0`; **mixed** if exactly one does; and **contrary** if neither does.
The report gives both the dataset count out of six and the model-arm count out
of twelve. No magnitude threshold, significance label, or dataset exclusion is
used to decide the direction.

## 6. Secondary, descriptive quantities

For each dataset/model the run also reports:

- point curves, marginal bootstrap intervals, curve-mean correlations, and the
  tail and Gaussian correlations entering the primary contrast;
- arithmetic, geometric, root-mean-square, and TOPSIS composite point values
  and paired percentile intervals;
- delta-method widths under the measured joint covariance and after setting all
  cross-covariances to zero, including the signed percentage change;
- matched-severity diagonal readings, deterministic-knot checks, label
  complement sensitivity, and the frozen-versus-full conditioning check.

These are descriptive. The second-level resampling of bootstrap rows, if shown,
measures Monte Carlo stability of the numerical bootstrap calculation; it is not
population-sampling uncertainty and is never called statistical significance.
Absolute score levels are not compared across datasets.

## 7. Failure, verification, and reporting rules

- Every intended dataset gets a log and a terminal status. Downloads, schema
  failures, missing classes, convergence warnings, non-finite metrics, timeouts,
  and adaptive changes are retained in `PROTOCOL-DEVIATIONS.md`.
- A defect-triggered rerun keeps the first output and states the defect.
- Replicate matrices are deposited for every completed run, including the
  Gaussian control and label sensitivity needed to audit the primary endpoint.
- An independent verifier recomputes the primary endpoint and interval-width
  calculations from the replicate deposits without importing the runner.
- The public summary reports all six intended datasets, outcome counts, the
  full range of the cross-covariance effect, and every failure. It does not say
  merely that six datasets were run.
- Manuscript wording, if used, must identify this as separate from the original
  Taiwan pre-registration and cite an immutable release/tag rather than mutable
  `main`.

