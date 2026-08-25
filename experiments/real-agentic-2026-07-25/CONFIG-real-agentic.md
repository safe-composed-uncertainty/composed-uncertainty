# CONFIG freeze: two-stage worked example from the live agentic pipeline

```
INTERNAL WORKING ARTIFACT. This file names repositories, hosts, and project
codenames for reproducibility. None of those names may appear in the
manuscript or in any article-facing text. In the article the substrate is
described only as "a live agentic research pipeline operated by the first
author". Do not commit this file or this directory without an explicit
go-ahead.
```

Frozen 2026-07-27, AFTER the pilot run (N=100, episode_seed=20261282,
artifacts under pilot/, never to be quoted) and BEFORE the production run.
No mixture, feature, or seed value below may change after the production
run is frozen. Driver: `harness.py` in this directory (the constants block
at the top of that file is the machine-readable copy of this note).

## Production run parameters

- N = 1,000 episodes, train 700 / eval 300, split stratified on the
  (yA, yB) cell, `train_test_split(random_state=20260727)`.
- SEED = 20260727. Episode rng: `default_rng([20260727, i])` per episode.
- B = 2,000 paired bootstrap replicates, alpha = 0.05.
- RGR noise: N(0, 0.5) * col_std added to every feature column (mirrors
  redraw-2026-07-24 on German credit, where coded categoricals were
  perturbed alongside continuous columns). Fixed-draw arm drawn from
  global default_rng(SEED) in the order (A,logit),(A,rf),(B,logit),(B,rf);
  redraw arm re-drawn per replicate with seed REDRAW_BASE[(stage,model)]+b,
  REDRAW_BASE = A/logit 20360727, B/logit 20460727, A/rf 20560727,
  B/rf 20660727.
- Index streams: paired pass default_rng(SEED+1) per (stage, model) arm;
  chain and joint-covariance pass default_rng(SEED+99) per model arm.
  Identical conventions to redraw-2026-07-24.
- Scorers: logit = StandardScaler + LogisticRegression(max_iter=2000),
  primary; rf = RandomForestClassifier(n_estimators=300), sensitivity arm.
  Both random_state = SEED. Mirrors the logit/rf pair of the German-credit
  study.
- Chain: h = C_A * C_B, C = geometric mean of (RGA, RGE, RGR), identical
  chain_block arithmetic to redraw_experiment.py.
- safeai pinned 39768fcd5264c881f7174268bbffda52b298ae89 (vendored clone of
  redraw-2026-07-24/safeai-src, read-only, with its inert torch/art stubs).
- Substrate: the ten pipeline modules vendored in ./pipeline-src (SHA-256
  of each recorded in results-real-agentic.json under
  pipeline_module_sha256), run unmodified. Python 3.12 system interpreter,
  numpy 2.5.0, scikit-learn 1.9.0, PyJWT 2.13.0, cryptography 49.0.0.

## Frozen episode mixture (probabilities sum to 1.000)

| Channel | p | Expected cell (yA,yB) | Source vector ids |
|---|--:|---|---|
| clean | .500 | (0,0) | happy path, demo aaa_demo.py M1 |
| over_scope_amount | .070 | (1,0) | mm-ablation over-limit L4; demo-attacks #2 |
| method_not_allowed | .050 | (1,0) | mm-ablation prompt-injected L4 |
| expired_mandate | .040 | (1,1) | demo-attacks #4 |
| untrusted_issuer | .030 | (1,1) | eatf-vec untrusted-issuer; mm-ablation L3 |
| wrong_holder_key | .025 | (1,1) | demo-attacks #5 |
| forged_credential_sig | .025 | (1,1) | mm-ablation L1; eatf-vec bad-signature-classical |
| payee_swap_action_hash | .030 | (1,1) | mm-ablation payee-swap L4; demo-attacks #1; demo-m2 confused-deputy |
| stale_attestation | .025 | (1,1) | mm-ablation L2 |
| nonce_replay_cross | .010 | (1,1) | demo-attacks #3, cross-session variant |
| nonce_replay_same | .010 | (1,0) | demo-attacks #3, same-session variant (gate check 5) |
| dtbs_field_tamper | .050 | (0,1) | demo-m2 fuzz (7 fields); eatf-vec tampered-metadata, tampered-overt-receipt |
| forged_seal | .030 | (0,1) | demo-attacks #6; eatf-vec bad-signature-classical |
| tampered_ear | .030 | (0,1) | demo-attacks #7; eatf-vec tampered-canonical-bin |
| outcome_mutation | .030 | (0,1) | eatf-vec tampered-canonical-bin (content mutated) |
| untrusted_sam | .030 | (0,1) | demo-attacks #8; eatf-vec untrusted-issuer (sealer analogue) |
| missing_component | .015 | (0,1) | eatf-vec missing-canonical-bin |

Expected prevalences: stage A DENY 0.315, stage B REJECT 0.370; all four
(A-verdict x B-verdict) cells >= 30 episodes in expectation on the
300-episode eval split. Tamper families by construction: A-only (scope
faults the offline verifier does not re-check, plus the same-session
replay), B-only (post-gate package tampers), both (credential, issuer,
freshness and binding faults).

## Frozen feature sets (pre-decision observables only; no signature-validity
or hash-recomputation bits)

- Scorer A (9): tool_code, log10_amount, payee_unknown, n_methods_allowed,
  log10_cap, headroom_frac, method_in_scope, expiry_margin_s,
  issuer_on_list.
- Scorer B (9): tool_code, log10_amount, payee_unknown, headroom_frac,
  expiry_margin_s, issuer_on_list, n_package_fields, nonce_match,
  ear_nonce_match.

One recorded deviation from DESIGN.md Section 5: the design listed
"freshness age" as a candidate scorer-A feature. The substrate's freshness
check is nonce-based, not age-based; no honest age observable exists, so
the stale-attestation channel is instead one of the tabular-invisible
faults for scorer A (it is visible to scorer B through ear_nonce_match,
which the design's "replay counter" field covers). Decided at pilot,
before the freeze.

Tabular-invisible channels (the honest overlap that keeps RGA away from
degeneracy): for scorer A - forged_credential_sig, wrong_holder_key,
stale_attestation, and all six post-gate tamper channels (which look clean
pre-authorisation); payee_swap only partially visible through the
payee_unknown proxy. For scorer B - dtbs_field_tamper on six of seven
fields, forged_seal, tampered_ear, outcome_mutation, untrusted_sam,
wrong_holder_key, forged_credential_sig, payee_swap; the scope channels
(which the offline verifier does not re-check) additionally mislead the
metadata view in the ALLOW direction.

## Pilot gate outcomes (recorded, not quoted as results)

- All 100 executed verdicts matched their channel's expected cell.
- Held-out AUC: A_logit 0.670, A_rf 0.688, B_logit 0.824, B_rf 0.731 -
  all below the 0.99 pre-registered failure threshold; no widening needed,
  mixture frozen as designed.
- Two pilot invocations reproduced every printed statistic bit-for-bit.

## Discipline

- Production artifacts: results-real-agentic.json,
  episodes-real-agentic.csv, run-real-agentic.log, REAL-AGENTIC-SUMMARY.md.
- Pilot artifacts stay in pilot/ and are never quoted.
- Second independent recomputation (same tag, fresh process, scratchpad
  out-dir) must reproduce every label and every metric before any number
  enters the manuscript. Crypto bytes are entropy-fresh per run; the
  verdicts, features and all downstream statistics are seed-determined.
- No number may be tuned after the production run is frozen.
