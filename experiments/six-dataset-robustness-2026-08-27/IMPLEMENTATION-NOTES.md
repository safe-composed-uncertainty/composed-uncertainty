# Outcome-blind implementation notes

These notes were fixed on 27 August 2026 before the first RGA, RGE, RGR,
correlation, composite, interval, or predictive-performance result on the six
datasets. They resolve implementation details inside the already frozen
protocol; they do not change the intended set, endpoint, direction, model pair,
or reporting rule. The runner hard-asserts the published protocol SHA-256
`791769816ef3dbe99a7f67b4abb07deafefdbe45f5aa9415e05708f017bd488d`
and manifest SHA-256
`b86e0e173e66f0b67f52046065b991e26097c4f98b1eabc7155ec81869042b7b`.

## Source and schema adapter

- `fetch_openml` is called by integer data id with `as_frame=True` and
  `parser="auto"`. The returned id, version, name, MD5, raw dimensions, target
  vocabulary, and adverse count must equal the manifest before any model call.
- The target is popped by its asserted name. The exact predictor names and
  dtypes are retained in `schema-preflight-audit.json`. Inspection found no row
  identifier field among the selected snapshots' predictor columns, so the
  identifier-exclusion list is empty for all six; no outcome-associated feature
  filtering is performed.
- The SAS HMEQ checksum in the manifest is a provenance cross-pointer only. The
  selected machine-readable input is the exact OpenML 43337 v2 snapshot; no
  row-equivalence claim between the SAS CSV and OpenML ARFF is used by the run.
- Numeric columns are pandas numeric dtypes other than bool. Category, object,
  string, and bool columns use the categorical branch. Categorical values are
  converted to their string representation before training-mode imputation and
  one-hot encoding.
- Raw data, raw labels, and source row indices are not deposited. The public NPZ
  files contain only derived curves, conditioning arrays, severity grids, and
  perturbation grids. Split membership and processed matrices are represented by
  shape-and-dtype-bound SHA-256 records produced by the runner, not by row values.

## Cap, split, and streams

- For a capped dataset, each class first receives
  `floor(class_count * 30000 / n)` rows. Remaining places go in descending order
  of fractional remainder, with stable class-value order for a tie. One
  `default_rng(20260723 + 5000 + dataset_order)` stream samples without
  replacement within the classes; the selected source indices are then sorted.
- The stratified 70/30 split follows the cap and uses sklearn's
  `train_test_split(..., random_state=20260723)`. All imputation, category maps,
  and zero-variance decisions are fit on its training rows only.
- `default_rng(20260724)` is reinitialised for each model, so the logistic and
  forest arms use the same row-index stream within a dataset. The label
  complement reuses that model arm's identical indices.
- The Gaussian sweep reinitialises `default_rng(20260723)` for each fitted model
  and for its complement. Thus corresponding arms see the same standard-normal
  draw matrix at each severity, scaled by that test matrix's column standard
  deviations.

## Sensitivity constructions

- Label complement means a full refit of the identical model specification on
  `1-y_train`, followed by evaluation on `1-y_test`. It is not an algebraic
  relabelling of the primary predictions.
- TOPSIS concatenates the three length-`L` curves. Its positive ideal is
  `[ones(L), 1 followed by zeros, ones(L)]`; its negative ideal is
  `[zeros(L), linspace(1, 0.5, L), 1 followed by zeros]`. Every concatenated
  coordinate has weight `1/3`, matching the earlier study. The reported score is
  distance-to-negative divided by the sum of distances to both ideals.
- The full-conditioning check recomputes the greedy RGE order and tail-swap
  predictions on each resampled test set. It uses the first 100 paired replicates
  for logistic regression and first 30 for random forest.
- The process constrains BLAS/OpenMP thread pools to one thread during each
  dataset run, in addition to the forest's frozen `n_jobs=1`; effective threadpool
  metadata are recorded.

## Preflight and fallback boundary

The runner loads, target-checks, caps, splits, and preprocesses all six intended
snapshots before the first SAFE metric, even when only one dataset is requested
for a smoke run. It compares the same hashes again immediately before each model
call. A preflight failure is retained with all six statuses and ends the process
without a metric. The exact six passed this preflight on 27 August 2026, so no
fallback was invoked. Once the first smoke metric exists, the fallback branch in
the protocol is permanently closed.
