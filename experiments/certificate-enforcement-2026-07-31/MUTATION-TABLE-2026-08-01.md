# Certificate-mutation negative table (section-8 candidate)

Every row below is a RUN test in `test_certificate_policy_gate.py`
(12/12 pass, 2026-08-01; two rows added today: unaccepted interval method,
unaccepted uncertainty scope). The table is tested code turned into prose —
no row claims more than its test asserts. Verdicts come from the unmodified
gate over the recorded real-chain certificate
(interval [0.768, 0.847], point 0.808, `first_order_delta`,
`fixed_artifact_evaluation`).

## Ten mutation/boundary rows

| # | Mutation or boundary case | Check that fires | Verdict |
|---|---|---|---|
| 1 | none — lower bound 0.768 clears policy 0.75 | lower bound ≥ policy | **ALLOW** |
| 2 | policy raised to 0.79 — interval straddles | interval straddles policy bound | **ESCALATE** |
| 3 | policy raised to 0.85 — upper bound below policy | upper bound < policy | **DENY** |
| 4 | evaluated after `recalibrate_by_utc` (2026-10-26) | certificate is past recalibrate_by_utc | **DENY** |
| 5 | payload tamper: composed score set to 1.0 | SHA-256 payload digest mismatch | **DENY** |
| 6 | signature byte flipped | Ed25519 verification failure | **DENY** |
| 7 | policy expects a different system_id | certificate subject does not match the system | **DENY** |
| 8 | mandate binds a different certificate payload digest | mandate binds a different certificate payload | **DENY** |
| 9 | signer is the publication example key, no explicit opt-in | untrusted signer key | **DENY** |
| 10 | certificate declares an interval method the policy does not accept (and: in-envelope method tamper also breaks the signature first) | interval method is not accepted by policy | **DENY** |
| 11 | certificate scope `fixed_artifact_evaluation` vs policy requiring `training_procedure` | uncertainty scope is not accepted by policy | **DENY** |

Control (not a mutation): the six-check MachineMandate action gate is **not
invoked at all** unless the certificate policy returns ALLOW — asserted by
`test_action_gate_not_called_unless_policy_allows` (ESCALATE → gate never
called; ALLOW → called once). A certificate-passing but out-of-scope action
is still denied by the mandate gate afterwards (`run_demo.py`).

## Companion beat (also executable)

`more_evidence_beat.py`: same system, same fixed 0.75 policy — n=30
evaluation rows give CI [0.606, 0.972] → **ESCALATE**; n=300 give
[0.772, 0.851] → **ALLOW**. Deterministic recorded scan; the deposited chain
estimator throughout; output in `MORE-EVIDENCE-BEAT.json`. Figure:
`figures/fig-undecided-band.{svg,pdf,png}` (panel a: one certificate, three
policies; panel b: the beat).

## Paper placement note

For the manuscript, rows 1–3 are the verdict semantics, rows 4–11 the
negative table proper; the caption must say "each row is an executed unit
test over the recorded certificate", and nothing here goes into Overleaf
until Pavia's math pass concludes — this file is the staged source.
