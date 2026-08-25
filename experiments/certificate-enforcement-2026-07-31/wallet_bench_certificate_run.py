"""One OID4VP bench run whose presentation request carries the certificate digest.

Closes the gap the asset map found: the mandate mechanics already put
`scope.uncertainty_policy` inside the signed MachineMandate (run_demo.py), and
the EUDI verifier bench already asks a live EC reference verifier for a
MachineMandate over real OpenID4VP — but no bench run had ever *requested* the
uncertainty-policy claim, so nothing tied the wallet leg to a measurement
certificate.

This script issues one live OID4VP presentation request to the local reference
verifier (dev-only, non-certified) whose DCQL query explicitly requests the
`scope.uncertainty_policy` claim path, and records:

  - the exact DCQL query sent, including the requested claim path;
  - the certificate payload digest the policy is bound to, read from the
    recorded certificate (never typed in);
  - the signed authorization-request JWT the verifier returns (header alg/typ,
    response_type/mode, nonce echo), persisted verbatim.

It does NOT claim a completed wallet round trip: no holder presents anything
here (that needs the physical wallet). What it establishes is that a real
OID4VP request can carry the certificate binding as a first-class requested
claim, with the verifier accepting and signing that request.

Requires the bench stack running (verifier backend on :18080).
Run:  python3 wallet_bench_certificate_run.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERT = HERE.parent / "certificate-examples" / "real-chain.json"
OUT = HERE / "WALLET-BENCH-CERTIFICATE-RUN.json"
VERIFIER = "http://localhost:18080"
VCT = "https://vocab.tyche.institute/vct/machine-mandate"
POLICY_ID = "research-chain-lcb-0.75"
MIN_LCB = 0.75


def post_json(url, payload, timeout=15):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, round((time.time() - t0) * 1000), json.loads(r.read())


def get(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return r.read().decode()


def jwt_parts(token):
    import base64

    def seg(s):
        return json.loads(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))
    head, payload, _ = token.split(".")
    return {"header": seg(head), "payload": seg(payload)}


def main():
    envelope = json.loads(CERT.read_text())
    digest = envelope["signature"]["payload_sha256"]
    score = envelope["certificate"]["measurement"]["composed_score"]

    nonce = "tyche-cert-" + hashlib.sha256(
        (digest + POLICY_ID).encode()).hexdigest()[:12]
    dcql = {
        "dcql_query": {
            "credentials": [{
                "id": "machine_mandate_with_uncertainty_policy",
                "format": "dc+sd-jwt",
                "meta": {"vct_values": [VCT]},
                "claims": [
                    {"path": ["principal"]},
                    {"path": ["scope"]},
                    {"path": ["action_hash"]},
                    # the new leg: the mandate must disclose the certificate binding
                    {"path": ["scope", "uncertainty_policy"]},
                ],
            }],
        },
        "nonce": nonce,
        "jar_mode": "by_reference",
        "response_mode": "direct_post.jwt",
    }

    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": VERIFIER,
        "claim_boundary": (
            "dev-only, non-certified research instrument; one live OID4VP "
            "presentation REQUEST carrying the certificate binding as a "
            "requested claim path. No holder presented a credential here: "
            "this is not a completed wallet round trip and establishes no "
            "interoperability or conformance claim."
        ),
        "uncertainty_policy_requested": {
            "policy_id": POLICY_ID,
            "certificate_payload_sha256": digest,
            "minimum_lower_bound": MIN_LCB,
        },
        "certificate": {
            "file": CERT.name,
            "composed_score": score["value"],
            "interval95": score["interval95"],
            "interval_method": score["interval_method"],
        },
        "dcql_query_sent": dcql,
    }

    try:
        code, ms, resp = post_json(f"{VERIFIER}/ui/presentations", dcql)
        req_jwt = get(resp["request_uri"])
        (HERE / "wallet-bench-request.jwt").write_text(req_jwt)
        parts = jwt_parts(req_jwt)
        requested_paths = None
        try:
            q = parts["payload"].get("dcql_query", {})
            requested_paths = [c.get("path") for c in
                               q["credentials"][0].get("claims", [])]
        except Exception:
            pass
        record["oid4vp"] = {
            "ok": code == 200,
            "http": code,
            "init_ms": ms,
            "transaction_id": str(resp.get("transaction_id", ""))[:24] + "…",
            "request_jwt_alg": parts["header"].get("alg"),
            "request_jwt_typ": parts["header"].get("typ"),
            "response_type": parts["payload"].get("response_type"),
            "response_mode": parts["payload"].get("response_mode"),
            "nonce_echoed": parts["payload"].get("nonce") == nonce,
            "requested_claim_paths_in_signed_request": requested_paths,
            "uncertainty_policy_path_requested": (
                ["scope", "uncertainty_policy"] in (requested_paths or [])),
            "request_jwt_file": "wallet-bench-request.jwt",
        }
        print(f"OID4VP request accepted: HTTP {code} in {ms} ms; "
              f"alg {parts['header'].get('alg')}; nonce echoed: "
              f"{record['oid4vp']['nonce_echoed']}")
        print(f"requested claim paths in the signed request: {requested_paths}")
    except (urllib.error.URLError, OSError) as exc:
        record["oid4vp"] = {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "hint": "start the bench stack (verifier :18080)"}
        print(f"bench verifier unreachable: {exc}", file=sys.stderr)

    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}")
    return 0 if record["oid4vp"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
