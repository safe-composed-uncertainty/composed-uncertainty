# Independent verifier contract

This file records the interface consumed by verify_six_dataset.py. It was
written before the verifier inspected any six-dataset metric output. The
verifier imports neither the primary runner nor safeai.

The authoritative experimental design is the already pushed PROTOCOL.md at
commit c9476e7, with the outcome-blind HMEQ manifest correction at 6129978. In
particular, preprocessing is training-only one-hot encoding, encoded_d may
differ from raw_d, and the full-research conditioning budgets are 100 logistic
and 30 random-forest replicates.

The verifier hard-anchors the frozen files rather than merely comparing a
result with whichever files happen to sit beside it:

    PROTOCOL.md SHA-256:
      791769816ef3dbe99a7f67b4abb07deafefdbe45f5aa9415e05708f017bd488d
    dataset-manifest.json SHA-256:
      b86e0e173e66f0b67f52046065b991e26097c4f98b1eabc7155ec81869042b7b
    requirements-six-dataset.txt SHA-256:
      0f82ff87831ea010b09b9bdce3467ebdbc608d9166c69bcfa753e517061c489f
    data-source-audit.json SHA-256:
      42c5ebafd911315a002d1bd259c8b7ee61772b66e32fb4bc482f07bd1d796199
    schema-preflight-audit.json SHA-256:
      7dfb10655a7c9999d1535c40e5f8349067a858ebe93bc5d46a1e621be9aeac65

## Files and invocation

The study directory contains one JSON and one NPZ per intended dataset:

    results-<slug>.json
    replicates-<slug>.npz

The slug must match the frozen order in dataset-manifest.json. Directory mode
requires all six intended datasets and is the default:

    python3 verify_six_dataset.py
    python3 verify_six_dataset.py --results-dir /path/to/study

One result can be diagnosed without claiming that the whole set passed:

    python3 verify_six_dataset.py --results results-<slug>.json

The default reports are verify-six-dataset.json for directory mode and
verify-<slug>.json for single-result mode. Any mismatch writes FAIL and exits
non-zero. Formula-only testing reads no outcomes:

    python3 verify_six_dataset.py --self-test

## Per-dataset JSON

The result root must contain:

    schema_version: non-empty string
    status: complete, completed, ok, pass or success
    dataset: object containing slug, or the slug string
    config: run configuration
    provenance: source and freeze provenance
    preprocessing: fitted training-only adapter record
    models: exactly logit and rf
    replicates_file: exactly replicates-<slug>.npz
    replicates_sha256: SHA-256 of that archive
    n_retained, n_train, n_test: split counts
    adverse_count_retained/train/test: target counts
    sampling.retained_source_indices_sha256: retained-row index hash
    dataset_manifest_entry: exact frozen manifest entry for the slug
    manifest_entry_sha256: canonical-JSON SHA-256 of that entry

The config object must contain:

    B: 2000
    alpha: 0.05
    B_conditioning_check: {logit: 100, rf: 30}
    d or encoded_d: transformed feature count
    curve_length_L: d + 1

The environment block must record Python 3.12.3 and the exact six package
versions in `requirements-six-dataset.txt`; every recorded BLAS/OpenMP pool must
show one effective thread.
    seed: 20260723
    bootstrap_seed: 20260724
    test_size: 0.30
    row_cap: 30000
    z: 1.959963984540054

The environment object records the exact NumPy, SciPy, scikit-learn, pandas,
joblib and threadpoolctl versions in requirements-six-dataset.txt and safeai
commit 39768fcd5264c881f7174268bbffda52b298ae89. Duplicate root/configuration
fields, when emitted, must agree. Every recorded BLAS/OpenMP pool must report
one effective thread.

The result also binds runner_sha256, requirements_sha256,
source_audit_sha256, schema_audit_sha256 and repository_commit. The verifier
compares the runner with the local public runner, hard-anchors the other three
artifacts to their pre-outcome digests, and requires a full Git object id.
For each slug it then links raw/processed hashes, sample and class counts,
column partitions, encoded names, imputation/unseen-level counts, removed
constants, duplicate counts and the all-six run-time preflight record to the
hard-anchored outcome-blind schema audit.

The result, config or provenance object must record protocol_sha256 and
dataset_spec_sha256. The verifier compares them byte-for-byte with PROTOCOL.md
and dataset-manifest.json in the result directory and compares those files with
the hard-anchored digests above. Directory mode obtains the six slugs from the
manifest and requires results-<slug>.json and replicates-<slug>.npz for each.

The public source record lives at provenance.openml and contains:

    id: positive integer
    name: non-empty string
    version: positive integer
    md5_checksum: 32 hexadecimal characters
    url: HTTP(S) URL
    license: non-empty string, or unknown when the source omits it
    manifest_download_sha256: frozen raw-download SHA-256
    raw_frame_sha256: SHA-256 of the loaded frame/target hash record

The preprocessing record contains:

    adapter_version: non-empty string
    raw_d: positive integer
    encoded_d: positive integer equal to config d
    numeric_columns: raw numeric feature names
    categorical_columns: raw categorical feature names
    numeric_fill_values: object keyed by every numeric feature
    categorical_fill_values: object keyed by every categorical feature
    category_maps: training levels keyed by every categorical feature
    encoded_columns: transformed names after zero-variance removal
    train_imputed_cells: non-negative integer
    test_imputed_cells: non-negative integer
    unseen_test_levels: non-negative integer
    processed_train_sha256: SHA-256
    processed_test_sha256: SHA-256

The raw numeric and categorical lists must each be duplicate-free, be disjoint,
and together cover raw_d. Fill values and category maps cover their
corresponding lists exactly; numeric fills are finite and every categorical map
is a non-empty list of unique strings. The verifier requires encoded_d = d but
deliberately does not require encoded_d = raw_d: one-hot expansion is part of
the frozen mixed-data adapter. encoded_columns contains exactly encoded_d
unique names and is also used to verify each full-conditioning removal order.

Each model entry must contain point curves. The canonical names, inherited
from the Taiwan result, are:

    curves_point:
      RGA_partial: length-L array
      RGE_greedy_mean_baseline: length-L array
      RGR_tail_swap: length-L array
      RGR_scaled_gaussian_noise: length-L array

The compact aliases RGA, RGE, RGR_tail and RGR_gaussian are also accepted.

The primary endpoint is stored as:

    primary_endpoint:
      corr_E_tail: number
      corr_E_gaussian: number
      delta_fisher_z: number
      prespecified_direction_delta_lt_zero: boolean

For compatibility during runner integration, primary_family_contrast and the
longer correlation/contrast key names are accepted aliases. The canonical
public result should use primary_endpoint.

Each model also contains conditioning_check with B_requested, B_completed and
B_full (or the accepted alias replicates/n_replicates) equal to 100 for logit
and 30 for rf, its B_full greedy orders, all four composite comparisons, and
the frozen and full-order primary-endpoint blocks. The verifier recomputes
those quantities from mandatory conditioning matrices in the NPZ.

The model's label_complement object has the same curves_point,
primary_endpoint and composites_tail structure as the primary model result.
The verifier links it to the mandatory flip point/replicate arrays and
independently recomputes its primary endpoint and four composite arms. It
remains an unconditional sensitivity and is never substituted for the primary
adverse=1 result.

The primary tail-swap composite object is:

    composites_tail:
      arithmetic: arm object
      geometric: arm object
      rms: arm object
      topsis: arm object

An arm object follows the established result shape:

    V_point: number
    paired_bootstrap:
      ci95_percentile: [lower, upper]
      width: number
      mean: optional number, verified when present
      sd: optional number, verified when present
    delta_method:
      summary_point: length-3 array
      gradient_at_point: length-3 array
      Sigma_of_summaries: 3 by 3 array
      width_delta_measured: number
      width_delta_cross_covariances_zero: number
      signed_width_change_pct_zero_vs_measured: number

Corr_of_summaries, var_measured_covariance and
var_cross_covariances_zero are optional but are independently checked whenever
present.

The verifier accepts the older Taiwan aliases
var_cross_terms_declared_zero, width_delta_declared_zero and
understatement_of_width_pct. The signed percentage convention is fixed as:

    100 * (1 - width_cross_covariances_zero / width_measured)

Positive therefore means that setting cross-covariances to zero narrows the
interval. Full-precision values are required; the verifier uses absolute and
relative tolerances of 5e-10.

## Per-dataset NPZ

For each of logit and rf, the archive must contain four finite float matrices of
shape (2000, L):

    <tag>_Ab
    <tag>_Eb
    <tag>_Rb_tail
    <tag>_Rb_gaussian

It also contains the four finite length-L point vectors used to link the JSON
point curves to the deposit:

    <tag>_a0
    <tag>_e0
    <tag>_r0_tail
    <tag>_r0_gaussian

The RGA and RGE matrices must be the same paired-bootstrap curves used with
both robustness families. The two RGR matrices must use those identical row
resamples. Additional audit arrays are allowed. When train_indices/test_indices
or y_test are present, the verifier checks their shape, integrity and agreement
with JSON counts and hashes. They are not required because the restricted-source
deposits intentionally contain derived curves rather than row-level source
data. The mandatory severity_t_over_d and tail_p vectors must equal t/d and
0.5*t/d at all L knots.

The unconditional label-complement matrices are also mandatory and must have
the same (2000, L) shape:

    <tag>_flip_Ab
    <tag>_flip_Eb
    <tag>_flip_Rb_tail
    <tag>_flip_Rb_gaussian

The corresponding point vectors are mandatory:

    <tag>_flip_a0
    <tag>_flip_e0
    <tag>_flip_r0_tail
    <tag>_flip_r0_gaussian

Finally, full-conditioning matrices are mandatory with shape (100, L) for
logit and (30, L) for rf:

    <tag>_conditioning_Ab
    <tag>_conditioning_Eb
    <tag>_conditioning_Rb_tail

The conditioning RGA matrix must equal the first B_full primary RGA rows. The
other matrices independently reproduce the conditioning composite statistics
and the full-order-tail endpoint recorded in JSON.

## Independent formulas

For each model, the verifier reduces Eb, Rb_tail and Rb_gaussian to one curve
mean per bootstrap replicate. It calculates Pearson correlation from centered
dot products and then calculates:

    Delta = atanh(r_tail) - atanh(r_gaussian)

The primary volume arm uses tail-swap robustness. Arithmetic and geometric
volume values use their exact factorisations. RMS uses the full L-cubed product
tensor in bounded-memory chunks and is checked by the outcome-free self-test
against explicit triple loops. TOPSIS is independently evaluated from the
fixed positive and negative ideals and weighted Euclidean distances.

For delta widths, arithmetic uses the three curve means and gradient
(1/3, 1/3, 1/3). Geometric uses the three cube-root curve means and the product
gradient. RMS independently derives the pointwise gradient of the full
product-space functional from the point curves and projects every replicate
curve onto it. TOPSIS independently derives the gradient of its closeness
coefficient and uses the same projection construction. The outcome-free
self-test checks both analytic gradients against central finite differences.
The declared-zero variance retains only the diagonal of the reconstructed 3 by
3 covariance matrix.

When old and new aliases coexist in a result, every emitted spelling is
checked; a correct canonical field cannot mask a contradictory legacy field.
Directory mode also derives, from verified values, the number of directionally
concordant datasets out of six, negative model arms out of twelve, dataset
classifications, and the minimum/maximum signed cross-covariance effect over
all 48 dataset/model/composite cells.

Each computed per-dataset report also carries verified n_retained, n_train,
n_test, raw_d, encoded_d, curve_length_L and cap_applied fields. The verifier
derives the expected retained/split counts from the manifest's source rows, the
30,000-row cap and the 70/30 split rule; it checks the cap flag/seed and checks
uncapped retained adverse counts against the manifest.
