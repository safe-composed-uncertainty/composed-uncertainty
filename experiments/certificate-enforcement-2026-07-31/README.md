# Certificate-driven enforcement demonstrator

This bounded companion demonstrator connects the paper's two previously
separate objects:

1. the executed authorization-and-evidence verifier produces evaluation
   episodes;
2. the SAFE-AI measurement layer issues the signed chain certificate; and
3. a relying-party policy gate verifies that certificate before invoking the
   ordinary six-check action gate.

The certificate gate verifies the JSON Schema, SHA-256 payload digest,
Ed25519 signature, a policy-pinned signer key, exact certificate payload
binding extracted from the cryptographically verified mandate, certificate
type, system and substrate identity,
interval method, uncertainty scope, issue time, `recalibrate_by_utc`, and the
lower confidence bound. Its result is:

- `ALLOW` when the lower bound clears policy;
- `DENY` when the upper bound is below policy or the certificate is invalid,
  untrusted, mismatched, or stale; and
- `ESCALATE` when the interval crosses the threshold.

On `ALLOW`, the normal MachineMandate checks still run. The demo places the
certificate payload digest, policy identifier, and threshold inside the
signed mandate scope; the Bound Action Seal then commits to the digest of that
mandate. A valid certificate therefore cannot authorize an out-of-scope
action.

The shipped certificate is signed by the article's deterministic publication
example key. The code rejects that key unless the caller explicitly opts into
example mode. This is a reproducibility demonstration, not an operational
trust configuration.

Run from this directory:

```text
python3 -m unittest -v test_certificate_policy_gate.py
python3 run_demo.py
```

With the recorded chain interval `[0.768, 0.847]`, threshold `0.75` allows the
system, threshold `0.79` escalates, expiry or tampering denies, and an
over-scope invoice is still denied by the ordinary action gate after the
certificate has passed.
