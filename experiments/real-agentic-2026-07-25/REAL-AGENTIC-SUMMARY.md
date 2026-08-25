# A two-stage worked example from the live agentic pipeline: run of record

```
INTERNAL WORKING ARTIFACT. This file names repositories, hosts, and project
codenames for reproducibility. None of those names may appear in the
manuscript or in any article-facing text. In the article the substrate is
described only as "a live agentic research pipeline operated by the first
author". Do not commit this file or this directory without an explicit
go-ahead. Do not paste from this file into the manuscript; write Section
7.2 fresh through the naming filter of DESIGN.md Section 9.
```

Run 2026-07-27, zeus1. Design G1 of DESIGN.md (2026-07-25), mixture and
features frozen in CONFIG-real-agentic.md before the production run.
N = 1,000 episodes, train 700 / eval 300 (stratified on the verdict cell),
d = 9 features per stage, B = 2,000 paired bootstrap replicates,
alpha = 0.05, noise sd 0.5, seed 20260727, pinned safeai
`39768fcd5264c881f7174268bbffda52b298ae89` (the redraw study's vendored
clone, read-only). Total runtime 398.5 s. Raw output:
`results-real-agentic.json`; episode table: `episodes-real-agentic.csv`;
log: `run-real-agentic.log`; driver: `harness.py`; adversarial
verification: `verify_chain_real_agentic.py` + `verify-chain.log`. The
pilot (N = 100, separate seed) lives in `pilot/` and is never quoted.

## 1. What the substrate is, and what is real

Each episode is one run through the deployed pipeline code vendored in
`./pipeline-src` (ten modules, SHA-256 per file recorded in the results
JSON), real ECDSA P-256 throughout:

- **Stage A - authorisation.** The six-check gate (`aaam.activate`):
  issuer on the trusted list; mandate + holder binding valid; presented
  action matches the mandated action hash; action within scope; nonce
  single-use; execution attestation affirming. Outcome ACCEPT/DENY plus
  the denying check. Executed, never simulated.
- **Stage B - evidence verification.** The four-check offline relying-
  party verifier (`rp_verify.verify`): mandate, execution, sole control,
  legal effect. Outcome ALLOW/REJECT plus a diagnostic class. Executed,
  never simulated.
- **Chain.** One episode's mandate, nonce, attestation digest and sealed
  evidence package flow through both stages, so the coupling is the
  deployed chain itself (Design G1).

Real, by execution: both verifiers (run unmodified), all signatures and
seals (per-episode P-256 identities, single-use SAM sole control), every
stage label, the tamper taxonomy (16 fault channels, plus clean,
de-duplicated from the published conformance suite's 7 invalid-vector
classes, the sibling artifact's gate-ablation levers, and the
demonstrator's own adversarial and fuzz suites - source vector ids per
channel in the CONFIG and in every CSV row). Synthetic, disclosed: the episode parameter draws
(randomised replays, not recorded user traffic; ~50% clean, mixture
frozen before the run). Trained for the purpose, disclosed: the two
probabilistic scorers (logit primary, random forest sensitivity), because
the deployed gates are deterministic and a 0/1 score gives degenerate
RGA.

Two harness steps sit between the deployed components and are disclosed
as such: episodes the gate denies still get their evidence sealed (same
DTBS/R composition, same SAM code), modelling an adversarial agent that
proceeds despite the gate - without this the stage B outcome would not
exist on gate-denied episodes and the 2x2 verdict table could not be
estimated; and a stage B exception on a structurally broken package
(missing component) is recorded as REJECT.

## 2. Ground truth and its verification

Every episode's expected verdict cell is fixed by its channel at draw
time; the recorded label is the executed verdict. The harness aborts on
any disagreement. **All 1,000 executed verdicts matched their channel's
expected cell.** Realised mixture: 508 clean, 492 tampered across 16
fault channels (counts per channel in the results JSON). Eval split:
DENY prevalence 0.293, REJECT prevalence 0.377, verdict cells
(A-verdict x B-verdict) = 153 / 59 / 34 / 54 - all four at or above the
30-episode floor the design required.

Three verification layers, all passed:

1. **Label tripwire** (above), all 1,000 episodes.
2. **Second independent recomputation**: a fresh process wrote its
   results to a scratchpad directory; the results JSON matched the run
   of record with zero differences outside `runtime_seconds`, and the
   1,001-line episode CSV was byte-identical. (Crypto bytes are
   entropy-fresh per run; verdicts, features and statistics are
   seed-determined, and were.)
3. **Adversarial verification** (`verify-chain.log`): episodes read back
   from the frozen CSV, scorers refit, and the chain quantities
   recomputed through an independently written replicate loop with
   hand-written covariance and delta arithmetic (explicit sums, no
   np.cov). Every checked quantity reproduced: AUCs and point metrics to
   0.00e+00, chain correlation / widths / understatement to <= 1.3e-13.

## 3. Scorers (trained for the purpose, disclosed)

Held-out AUC on the 300 eval episodes: stage A logit 0.820, rf 0.851;
stage B logit 0.666, rf 0.735. All far from the pre-registered 0.99
separability-trap threshold - the honest overlap comes from fault
channels invisible in the tabular view by nature (signature byte flips,
sealed-field tampers, content mutation after sealing), exactly the
position of a metadata-level risk scorer that cannot do cryptography.
No overlap was engineered and no feature was tuned after the freeze.

## 4. Per-stage results (same shape as the German-credit study)

Point estimates on the 300 shared eval episodes, fixed-draw perturbation:

| Stage/model | RGA | RGE | RGR | C (geomean) | C 95% CI (redraw, paired percentile) |
|---|--:|--:|--:|--:|:--|
| A / logit | 0.9225 | 0.9532 | 0.9352 | 0.9369 | [0.9153, 0.9545] |
| A / rf | 0.9337 | 0.9666 | 0.9247 | 0.9415 | [0.9198, 0.9599] |
| B / logit | 0.7515 | 0.9589 | 0.8887 | 0.8620 | [0.8279, 0.8966] |
| B / rf | 0.7902 | 0.9433 | 0.8384 | 0.8549 | [0.8112, 0.8822] |

Cross-metric correlations across the B = 2,000 paired replicates
(fixed-draw / redraw schemes, as in the redraw study; RGA-RGE is
identical across schemes by RGE determinism, reproduced here):

| Pair | A/logit fix | A/logit redraw | A/rf fix | A/rf redraw | B/logit fix | B/logit redraw | B/rf fix | B/rf redraw |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RGA-RGE | +0.253 | +0.253 | +0.279 | +0.279 | +0.051 | +0.051 | +0.516 | +0.516 |
| RGA-RGR | +0.358 | +0.310 | +0.320 | +0.214 | +0.508 | +0.415 | +0.465 | +0.444 |
| RGE-RGR | +0.601 | +0.539 | +0.342 | +0.242 | +0.120 | +0.094 | +0.524 | +0.448 |

## 5. The chain, which is what the section needs

Two-link chain h = C_A * C_B over the shared eval episodes (index stream
SEED+99, both stages recomputed on the same resample per replicate):

| | logit fixed-draw | logit redraw | rf fixed-draw | rf redraw |
|---|--:|--:|--:|--:|
| cross-link correlation | +0.319 | +0.314 | +0.275 | +0.240 |
| plug-in h point | 0.8076 | 0.8076 | 0.8049 | 0.8049 |
| composed 95% width, measured covariance | 0.0792 | 0.0786 | 0.0760 | 0.0805 |
| same, cross-terms declared zero | 0.0705 | 0.0700 | 0.0687 | 0.0737 |
| understatement of width | 11.0% | 11.0% | 9.7% | 8.5% |

The joint 6-metric covariance (both stages' RGA/RGE/RGR on the shared
resample) is in the results JSON under `joint`; the cross-stage block is
positive in seven of nine entries (redraw, logit: A_RGA with B_RGA
+0.280, A_RGE with B_RGR +0.334; the two negatives are A_RGA with B_RGE
-0.132 and A_RGE with B_RGE -0.011).

Reading, and the side-by-side the article can carry. On this real
two-stage pipeline the cross-link correlation is genuinely positive
(+0.24 to +0.32 across schemes and scorer families) and declaring the
cross-terms zero understates the chain interval width by 8.5 to 11 per
cent. The sign was not predicted in advance (the Pavia composite study
proved it can flip); it came out positive, smaller than the German-credit
chain (+0.712 redraw, understatement 23.6%), and for an identifiable
reason: there the two links were two models scoring the same predictions
on one dataset, here they are two different decision stages with
different outcome variables and different feature views, coupled only
through the shared episodes. That contrast - composition of one system's
metrics versus composition across a real decision chain - is itself
usable in Section 7.2, and the honest sentence is that the cross-link
term is smaller across a heterogeneous chain but still material: about
one tenth of the interval width on this pipeline.

## 6. Deviations from DESIGN.md, all decided before the freeze

1. "Freshness age" was dropped as a scorer-A feature: the substrate's
   freshness check is nonce-based, and inventing an age number would
   have been synthetic decoration. The stale-attestation channel is
   tabular-invisible at stage A instead (visible at stage B through
   ear_nonce_match, the design's "replay counter" field).
2. The replay channel splits into two executed sub-variants:
   cross-session replay (fresh challenge; denied at the mandate-binding
   check; rejected downstream) and same-session replay (denied at gate
   check 5, the nonce store; the once-genuine evidence itself still
   verifies offline, so it lands in the (DENY, ALLOW) cell). Both are
   real behaviours of the deployed code; the design's single "replayed"
   row did not distinguish them.
3. Scope faults land in (DENY, ALLOW) because the offline verifier does
   not re-check scope - a real property of the deployed verifier worth
   one sentence in the article, since it is why the 2x2 table has mass
   off the diagonal at all.

## 7. Files

| File | Contents |
|---|---|
| `harness.py` | the whole study: episode generator + real-chain execution + scorers + metrics + paired bootstrap + joint covariance + chain |
| `CONFIG-real-agentic.md` | frozen mixture, features, seeds; pilot gate outcomes; freeze discipline |
| `results-real-agentic.json` | every number above, plus Sigma/Corr per stage and scheme, the 6x6 joint blocks, channel counts, module hashes |
| `episodes-real-agentic.csv` | 1,000 traceable episodes: seed, channel, source vector ids, parameters, executed verdicts, denying check, diagnostics, split, features |
| `run-real-agentic.log` | run of record |
| `verify_chain_real_agentic.py`, `verify-chain.log` | adversarial verification (independent loop and arithmetic) |
| `pilot/` | pilot run, separate seed, never quoted |
| `pipeline-src/`, `DESIGN.md` | vendored substrate and the approved design |

Nothing in this directory is committed. Nothing here descends from any
quarantined lineage. The only network-touching event in the whole strand
remains the single read-only ledger GET during the 2026-07-25 recon; the
generator and this run made no network calls.
