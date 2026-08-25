# Two-stage worked example from the live agentic pipeline: design

```
INTERNAL WORKING ARTIFACT. This file names repositories, hosts, and project
codenames for reproducibility. None of those names may appear in the
manuscript or in any article-facing text. In the article the substrate is
described only as "a live agentic research pipeline operated by the first
author". Do not commit this file or this directory without an explicit
go-ahead. Do not paste from this file into the manuscript; write the
manuscript text fresh through the naming filter in Section 9.
```

Status: DESIGN ONLY. No code yet, no generated data yet. Every count and
timing cited below comes from the recon of 2026-07-25 (files read on disk,
plus three component runs executed in scratchpad copies that day) or from
the two frozen reference studies in this experiment tree. Nothing is quoted
from memory.

Purpose: insurance for Section 7.2 of the joint article (Sokolov /
Kolesnikov / Babaei TBC / Giudici), due 12 September. The planned source
(the SAFE Agents workflow) is unavailable until end of August. This design
produces a real agentic worked example from the first author's own pipeline
so the section cannot be left empty.

## 1. Feasibility verdict

**GENERATE.** The data does not exist as logs, but the pipeline's real
verifiers run offline on this host at millisecond cost with local signing
keys, so a 1,000-episode two-stage chained evaluation set with ground truth
by construction can be produced in well under an hour of wall time. BUILD
NOW fails (no logged source reaches 150 independent instances with features
and outcomes); INSTRUMENT is kept as an optional upgrade path (Section 10)
but the calendar math says organic traffic will not get there by early
September.

Audit summary behind that verdict (details in the recon of 2026-07-25):

| Source | Real instances | Why it fails as-is |
|---|---|---|
| Live playground gate ledger (public endpoint; snapshot in scratchpad, 19,234 bytes) | 80 visible entries (44 DENY / 36 ACCEPT), window-capped | Below 150; session-clustered; stage A only, no per-episode stage-B outcome |
| /srv/tyche/repos/aep-sandbox/ | 2 sample packages + 5-line samples/ledger.jsonl | Tiny as logs; but a complete offline generator (verify.py timed 0.040 s/call; private keys present locally) |
| /srv/tyche/repos/machine-mandate/ | 7-row gate-ablation table | Tiny as logs; but a four-gate verifier accepting arbitrary randomized inputs (0.083 s/call) |
| /srv/tyche/repos/accountable-agentic-action/demo/ | ~21 executed adversarial/happy-path cases (artifacts dated 2026-06-30) | Tiny as logs; but the full chained pipeline runs end-to-end in 0.059 s/call |
| eatf test-vectors + 2026-07-18 verification logs | 11 conformance vectors (4 valid + 7 invalid), dual-verifier agreement | Distinct failure modes, not a sample; usable as tamper taxonomy seed |
| Vault submission ledger + outcomes ledger | 188 records, ~73-87 with terminal outcomes | Below threshold, badly non-independent, NOT public-safe (venues, editor text) |
| EUDI bench, X-Road replay runs, watch lanes, agent atlas | 9 / 3x5 / 19 / 89 rows respectively | Too small, or no outcome variable; LIMEN/Icarus/article-forge descendants quarantined on principle |

Minimum honest size per the task brief is 150+ evaluation instances with
outcomes. The design below evaluates on 300 held-out episodes (twice the
floor) drawn from 1,000 generated ones.

## 2. The two stages

The mapping follows the deployed architecture of the pipeline itself: an
action request first passes an authorisation gate, then execution produces
a signed evidence package which an independent verifier appraises. These
are genuinely chained decisions about the same episode, which is exactly
what the chain machinery needs.

**Stage A -- authorisation.** An agent action request (tool, amount, payee)
arrives with a mandate (allowed actions, spend cap, currency, expiry) and a
runtime attestation of stated freshness from an issuer that is or is not on
the trusted list. The deployed gate checks four things in order: credential
cryptography, freshness, issuer trust-list membership, and scope plus
action-hash binding. Output: ACCEPT/DENY plus the identity of the denying
check.

**Stage B -- evidence verification.** The episode's execution produces a
signed evidence package (action record, mandate reference, hash chain,
signature, replay counter). The offline verifier appraises it and returns
ALLOW/DENY plus a diagnostic class (content mutated, signature invalid,
issuer not listed, scope violation, replayed).

**Chain.** One episode flows through both stages. Episode-level draws (an
over-scope amount, a stale attestation, an untrusted issuer, a tamper on
the package bytes) causally affect one or both stage outcomes, so the
cross-link correlation is a property of the process, not something we
engineer into the statistics.

## 3. Two close designs, and the recommendation

**Design G1 (recommended): single chained generator on the end-to-end
demonstrator.** Build the episode generator around
/srv/tyche/repos/accountable-agentic-action/demo/ (aaa_demo.py runs the
full mandate -> authorisation -> execution attestation -> bound action seal
-> composite offline verification chain in 0.059 s per process call, timed
2026-07-25 in a scratchpad copy). Each episode is ONE run of the real
chain: the stage A label is the authorisation verdict, the stage B label is
the composite evidence verdict on the very package that episode produced.
Coupling is physical: the same mandate object, the same nonce, the same
attestation digest pass through both stages.

**Design G2 (close alternative): two-verifier pairing.** Stage A =
the four-gate verifier in /srv/tyche/repos/machine-mandate/
(src/verifier_core.py, m.evaluate(SCOPE, action, mandate, replay), 0.083 s
per ablation run) and stage B = /srv/tyche/repos/aep-sandbox/verify.py
(0.040 s) on a package minted with the locally present private keys
(keys/private/agent.sk, issuer.sk, principal.sk). Coupling is by shared
exogenous draws: the same sampled action/mandate/freshness parameters feed
both components.

Recommendation: **G1 primary, G2 as the replication arm.** G1's coupling is
the deployed chain itself, which is the stronger claim for the article
("the two stages are two decision points of one executed pipeline run").
G2 couples only through shared inputs, which is weaker but uses two
independently published artifact codebases, so it is the natural robustness
check: if the cross-link correlation estimated under G1 and G2 disagree
wildly, that is itself informative and must be reported. Build G1 first; add
G2 only if time allows (it reuses the same episode sampler).

## 4. Episode generation plan (the GENERATE recipe)

Draw N = 1,000 i.i.d. episodes from a seeded generator. Per episode,
sample:

- action parameters: tool (from the demo's action set), amount_eur
  (log-uniform over a range spanning the spend cap), payee (from a pool
  including on- and off-allowlist entries);
- mandate scope: allowed_actions subset, max_spend, currency, expiry
  offset (some already expired);
- freshness age of the attestation (some beyond the freshness window);
- issuer: on or off the trusted list;
- a tamper channel drawn from the published taxonomy with probability
  ~0.5, else clean. The taxonomy union: the 7 invalid-vector classes of
  the eatf conformance suite (envelope/hash-chain/signature/timestamp
  tampers), the 7 gate-ablation attack levers of machine-mandate, and the
  7 DTBS/R fuzz fields of the demo's test_m2.py. De-duplicated into one
  labelled channel list at build time.

Then execute the real chain (G1) on the episode and record: all sampled
parameters, the stage A verdict + denying gate, the produced package's
observable attributes, and the stage B verdict + diagnostic class. Labels
are ground truth by construction of the draw; no label is ever assigned by
hand or by a model.

Mixture targets (to be frozen in a CONFIG note before the production run,
and not tuned after seeing metric values):

- roughly 50% clean episodes;
- each stage's DENY/REJECT prevalence in the 25-40% band;
- all four cells of the (A-verdict x B-verdict) 2x2 table non-empty with
  at least ~30 episodes each in the 300-episode eval split, so the
  cross-link correlation is estimable;
- tamper channels split into three families by construction: A-only
  (e.g. over-scope amount with a clean package), B-only (e.g. post-gate
  byte tamper on the package), and both (e.g. untrusted issuer, which
  fails the trust-list gate and the package appraisal).

Cost arithmetic (all timings measured 2026-07-25): 1,000 chained episodes
at 0.059 s per process call is about 60 s; even the conservative
process-call route for G2 (0.083 + 0.040 s) stays under 3 minutes. The
bootstrap layer is the same machinery as the redraw study
(/srv/tyche/repos/tyche-research-vault/papers/safe-composed-uncertainty/experiment/redraw-2026-07-24/,
total runtime 204.5 s for train 700 / test 300, d = 20, B = 2,000), so the
full study is minutes, not hours. One afternoon covers build, pilot, and a
frozen production run.

Everything runs offline on this host. No network traffic, no production
hosts, no live playground involvement. The single read-only GET against the
public ledger endpoint during recon is already noted in the recon record
and is not repeated by the generator.

## 5. What is real and what is trained for the purpose

Real, by execution:

- the two verifiers and the chained demonstrator (deployed pipeline code,
  publicly published as artifacts, run unmodified);
- the signing keys and the minted packages (real cryptographic material,
  local, demo identities only);
- every verdict/label (output of an executed verification run);
- the tamper taxonomy (published conformance suite + published ablation
  levers + executed fuzz fields);
- the coupling between stages (the deployed chain).

Synthetic, and disclosed as such: the episode parameter draws. The
episodes are randomized replays, not recorded user traffic. The article
says this plainly.

Trained for the purpose, and disclosed as such: the two probabilistic
scorers. The deployed gates are deterministic; a deterministic 0/1 gate
score gives degenerate RGA. So we train, per stage, a small scorer on the
700 training episodes and evaluate on the 300 held-out ones:

- scorer A: predicts P(stage A DENY) from pre-authorisation observables
  (tool, amount_eur, payee class, scope parameters, spend-cap headroom,
  expiry margin, freshness age, issuer trust flag);
- scorer B: predicts P(stage B REJECT) from pre-verification observables
  of the package (declared action attributes, mandate reference fields,
  replay counter value, timestamp fields, structural envelope features)
  -- explicitly NOT the signature-validity or hash-recomputation bits,
  which are the verifier's own work.

Primary scorer: logistic regression. Sensitivity arm: gradient boosting or
random forest, mirroring the logit/rf pair of the German-credit study.

**The separability trap (pre-registered failure condition).** If the
feature set fully determines the verdict, the trained scorer separates
perfectly and RGA degenerates again. The design avoids this honestly, not
cosmetically: several tamper channels are invisible in the tabular view by
nature (signature byte flips, hash-chain breaks, content mutation after
sealing change no pre-verification observable), so scorer B faces
irreducible class overlap -- exactly the position of a real metadata-level
risk scorer that cannot do cryptography. For scorer A, freshness age and
spend-cap headroom are continuous with overlapping supports across
verdicts. Check at pilot: if either held-out AUC exceeds ~0.99, widen the
overlap (more invisible-channel mass, noisier observables) BEFORE the
production run and record the change; if overlap cannot be achieved without
distorting the pipeline's semantics, declare that stage's RGA not
meaningful and say so. No silent tuning after the production run is frozen.

## 6. Metric computability, stage by stage

All three metrics use the pinned safeai code already vendored for this
paper (pin 39768fcd5264c881f7174268bbffda52b298ae89, per
REDRAW-SUMMARY.md), on the shared 300-episode eval set.

- **RGA** (needs a predicted score vs a realised binary outcome): scorer
  A's predicted probability vs the executed stage A verdict; scorer B's
  predicted probability vs the executed stage B verdict. Both binary.
  Computable per stage.
- **RGE** (needs feature-attributed predictions on tabular features): both
  scorers consume explicit tabular features listed in Section 5. Default
  greedy masking is deterministic at default settings (confirmed in the
  redraw study from the pinned code, safeai/rge.py, greedy default at
  rge.py:1635-1671), so RGE carries no auxiliary randomness. Computable
  per stage.
- **RGR** (needs the same predictions under a defined input perturbation):
  perturb the continuous features (amount_eur, freshness age, expiry
  margin, counter/timestamp fields) with the same noise scheme as the
  German-credit study (noise sd declared in config). Use the REDRAW
  scheme from day one: fresh perturbations per bootstrap replicate,
  seeded per replicate index, so the reported cross-metric correlations
  are unconditional. Computable per stage.

## 7. Chain composition and the bootstrap

Identical machinery to redraw-2026-07-24 and the composite study of
pavia-composite-2026-07-25:

- paired bootstrap over the 300 shared eval episodes, B = 2,000, alpha =
  0.05; episodes are i.i.d. by construction, so plain episode-level
  resampling is valid (no session clustering needed -- unlike the live
  ledger, which is one reason the ledger is not the substrate);
- per replicate, recompute RGA/RGE/RGR for both stages on the same
  resampled episode index set -- this yields the joint 6-metric
  covariance;
- two-link chain h = C_A * C_B where C is the per-stage composite the
  article's Section 5 settles on (scalar composite, or the tensor-mean
  Compliance Score of the Pavia construction -- follow whatever the
  article's earlier sections fix, and if that is the tensor construction,
  reuse pavia_composite_experiment.py's layer);
- report: per-stage metric values with intervals, the cross-link
  correlation, the composed interval width with measured covariance vs
  cross-terms declared zero, and the width understatement percentage --
  the same table shape as the German-credit result (there: correlation
  +0.712 redraw, understatement 23.6%). We do NOT predict the sign or
  size here; the Pavia composite study proved the sign can flip, so the
  honest posture is to measure and report whatever comes out.

Sample sizes: train 700 / eval 300 mirrors the reference studies and puts
the eval set at 2x the 150-instance honesty floor stated in the task brief.

## 8. Freeze and verification discipline

- Seeded everything: episode sampler, train/test split, scorer fits,
  bootstrap, RGR redraw base. One config block, recorded in the run log.
- Production run frozen as results-real-agentic.json +
  run-real-agentic.log in this directory; the pilot run kept separately
  and never quoted.
- Two independent recomputations before any number enters the manuscript
  (the same freeze rule the conformance corpus uses).
- Adversarial verification pass on the cross-link estimate (the
  verify_cross_terms.py pattern from the Pavia composite study) before
  the section is drafted.
- Nothing in this directory is committed without explicit go-ahead.
- Nothing descended from the LIMEN / Icarus / article-forge lineages
  feeds this example, directly or indirectly.

## 9. Privacy and codename pass

What the article MAY say (shape, not final prose):

  "As a second worked example we use a live agentic research pipeline
  operated by the first author. In this pipeline an agent's action
  request passes an authorisation gate that checks credential validity,
  attestation freshness, issuer trust, and scope; execution then produces
  a signed evidence record that an independent offline verifier
  appraises. Both components are deterministic, so we replay them over
  N = 1,000 randomised episodes and train two small probabilistic scorers
  (logistic regression; gradient boosting as a sensitivity check) on the
  executed episodes to serve as the scored stages. Every label is the
  output of an executed verification run; no outcome is simulated. The
  episode parameters are randomised draws, not recorded user traffic, and
  the fault taxonomy follows the pipeline's published conformance suite."

What the article MUST NOT contain -- banned-string list for the grep gate
(grep is necessary but not sufficient; a human read follows, since grep has
missed codenames before):

- project/repo names: accountable-agentic-action, machine-mandate,
  aep-sandbox, eatf, eatf-verifier, GATEHOUSE;
- the acronym family of the evidence format (AEP, "Action Evidence
  Package", EATF) -- write "evidence record" / "evidence package"
  generically;
- domains and hosts: any *.eatf.eu name, agent.*, uri.*, regwatch.*,
  aletheia, zeus1, zeus2;
- paths: anything under /srv/tyche, vault-relative paths, scratchpad
  paths;
- third-party stack names that fingerprint the pipeline (Veraison, RATS
  EAR wording, SCITT draft names) -- keep Section 7.2 at the generic
  architectural level;
- the live playground, its ledger, its URL, and its visitor handles --
  if the 80-entry live ledger is mentioned at all, it is one aggregate
  sentence ("the deployed gate has also adjudicated live public traffic")
  with no counts tied to a findable endpoint, and the default is to omit
  it entirely.

Data-protection notes: generated episodes contain no personal data
(identities are demo strings such as operator:alice); the live ledger's
pseudonymous handles are not exported into the study; the vault submission
ledger and the fabrication-tripwire reports are excluded from this example
entirely (confidential venue material and internal lane names
respectively).

## 10. Fallback: the INSTRUMENT path (not needed, kept for the record)

If live-traffic data were required instead of replayed episodes, the
playground backend would need to log, per episode: episode id, session id,
timestamp, full request features (tool, amount, payee, scope parameters,
freshness age, issuer), gate verdict + denying gate, the produced evidence
package's observable attributes, and a per-episode verification verdict +
diagnostic (this last field does not exist in today's ledger, which only
stores the gate side). Bootstrap would need session-level clustering.
Calendar math: the visible window held 80 entries on 2026-07-25 with an
unknown all-time rate; reaching 150+ independent SESSIONS (not entries) by
early September from organic traffic is unlikely, and promotion to force
traffic would contaminate the "live" claim. Conclusion: switching the
logging on is cheap and worth doing for a future paper, but it is not the
route to 12 September. GENERATE is.

## 11. Timeline to 12 September

- Week of 28 Jul: build the G1 episode generator against the
  scratchpad-verified components; pilot run N = 100; check the
  separability condition (Section 5) and the 2x2 cell occupancy
  (Section 4).
- By 08 Aug: freeze mixture CONFIG; production run N = 1,000; scorers,
  metrics, bootstrap, chain table; second independent recomputation;
  adversarial pass on the cross-link estimate. (Deliberately before the
  end-August SAFE Agents date so the insurance exists before we learn
  whether it is needed.)
- By 22 Aug: Section 7.2 prose drafted through the Section 9 filter;
  banned-string grep + human read.
- 31 Aug: decision point -- if the SAFE Agents workflow (Babaei) lands,
  this example moves to an appendix or a robustness paragraph; if not,
  it IS Section 7.2.
- 01-12 Sep: buffer for co-author comments (Giudici / Kolesnikov) and
  integration.

Total new compute: minutes. Total new writing: the generator script, a
CONFIG note, and the section prose. The binding constraint is review time,
not data.
