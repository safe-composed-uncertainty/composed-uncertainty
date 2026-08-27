# Composed uncertainty certificate profile 0.3

This profile turns the certificate proposal into an executable example. It
separates a link certificate, which carries a component covariance matrix,
from a chain certificate, which additionally carries the measured cross-link
covariance or an explicit zero declaration.

Profile 0.3 supersedes 0.2. It adds two required fields, `interval_method`
and `uncertainty_scope`, and is therefore not backwards compatible: a 0.2
envelope fails validation against the 0.3 schema.

## Files

- `certificate-profile-0.3.schema.json` — JSON Schema 2020-12 profile.
- `build_certificate_examples.py` — deterministic builder, validator and
  signature verifier.
- `certificate-examples/` — two link certificates and one chain certificate
  for each worked example.

## Signed payload

Only the `certificate` object is signed. The canonical byte string is UTF-8
JSON with keys sorted, separators reduced to comma and colon, and
`ensure_ascii=false`. This is a deliberately small deterministic profile; it
does not claim full RFC 8785 conformance.

The examples use Ed25519. Their deterministic private key is embedded in the
builder solely to make the publication examples reproducible. The key is
labelled `PUBLICATION EXAMPLE KEY — NEVER OPERATIONAL` in every envelope.
Production issuers must use an independently managed signing key. The key seed
is not versioned with the profile, so the 0.3 examples are signed under a
different public key from the 0.2 examples they replace.

## Interval method (new in 0.3)

`composed_score.interval_method` is required on both certificate classes and
takes one of `paired_bootstrap_percentile`, `first_order_delta` or
`logit_delta`. It exists because the two classes do not carry the same kind of
interval, and without the field a verifier cannot tell which recomputation to
attempt:

- a **link** record carries `paired_bootstrap_percentile` — the empirical
  2.5th and 97.5th percentiles of the composed score across the paired
  replicates. These endpoints are a functional of the replicate distribution,
  not of `component_covariance`, so they are **not** recomputable from the
  covariance matrix the same record carries. A first-order recomputation from
  `component_covariance` misses the recorded endpoints by 1.11 to 6.08 per cent
  of the recorded width across the four example links: the skewness of a
  composite of near-one components, not an error;
- a **chain** record carries `first_order_delta` — point estimate plus and
  minus `z * sqrt(g' Sigma g)` for the gradient `g` of the chain functional.
  This *is* an exact function of `link_point_values` and `link_covariance`, and
  both example chains reproduce their own interval, their own
  `zero_cross_term_sensitivity` interval and their own `width_change_pct` to
  0.0 from the recorded numbers alone.

`build_certificate_examples.py` performs exactly this check on every envelope
it verifies, raising if a chain record fails to reproduce and printing the
measured link gaps rather than asserting they are zero.

## Uncertainty scope (new in 0.3)

`estimation.uncertainty_scope` is required and takes `fixed_artifact_evaluation`
or `training_procedure`. All six examples are
`fixed_artifact_evaluation`: the interval covers evaluation sampling for one
frozen trained artifact on one named sample, and does not cover
training-split, seed or hyperparameter variability. A retrain is a listed
recalibration trigger rather than a source integrated into the interval, so the
two scopes must never be added or compared without restating the estimand.

## Validity

The example horizon is 90 days. Any model retrain, substrate change,
evaluation-sample change, perturbation-family change or detected distribution
shift triggers earlier recalibration. A chain expires at the earliest expiry
of its input link certificates.

## Chain covariance

`link_covariance` is the measured 2-by-2 covariance matrix of the link
estimators on one paired bootstrap stream. `cross_link_covariance` identifies
the off-diagonal value, its correlation, provenance and the digest of the
shared sample. When a chain genuinely uses separate evaluation samples, the
status can be `declared_zero`; that declaration remains visible in the signed
payload.

## Verification

Run:

```text
python3 build_certificate_examples.py
python3 build_certificate_examples.py --verify-only
```

Both commands validate every output against the JSON Schema, verify the
payload digest and Ed25519 signature, and run the interval self-consistency
check described above.
