#!/usr/bin/env python3
"""Execute the certificate gate in front of the article's real action gate."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
PIPELINE = EXPERIMENT_DIR / "real-agentic-2026-07-25" / "pipeline-src"
sys.path.insert(0, str(PIPELINE))

import aaam
import attester
import mandate
from aaam import AAAMReject
from crypto import gen_ec, pub_jwk
from jcs import H, Hs
from sam_sim import SAM
from trusted_list import TrustedList

from certificate_policy_gate import (
    CertificatePolicy,
    certificate_guarded_activate,
    evaluate_certificate,
)


CERTIFICATE_PATH = EXPERIMENT_DIR / "certificate-examples" / "real-chain.json"
AS_OF = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


def load_envelope() -> dict:
    return json.loads(CERTIFICATE_PATH.read_text())


def policy(envelope: dict, threshold: float) -> CertificatePolicy:
    certificate = envelope["certificate"]
    return CertificatePolicy(
        policy_id=f"research-chain-lcb-{threshold:.2f}",
        expected_certificate_type="chain",
        expected_system_id=certificate["subject"]["system_id"],
        expected_substrate=certificate["subject"]["substrate"],
        minimum_lower_bound=threshold,
        accepted_interval_methods=("first_order_delta",),
        accepted_uncertainty_scopes=("fixed_artifact_evaluation",),
        trusted_public_keys=(envelope["signature"]["public_key_hex"],),
        allow_publication_example_key=True,
    )


def action_material(certificate_digest: str, *, amount: float, cap: float):
    issuer, holder, attestation_key = gen_ec(), gen_ec(), gen_ec()
    sam = SAM()
    trusted_list = TrustedList()
    trusted_list.add_issuer(pub_jwk(issuer))
    trusted_list.add_qtsp(sam.jwk)
    nonce = "certificate-policy-demo-nonce"
    audience = "rp://certificate-policy-demo"
    action = {
        "method": "pay_invoice",
        "args": {"invoice": "INV-CERT-001", "amount_eur": amount, "payee": "acme"},
    }
    scope = {
        "methods": ["pay_invoice"],
        "max_amount_eur": cap,
        "uncertainty_policy": {
            "policy_id": "research-chain-lcb-0.75",
            "certificate_payload_sha256": certificate_digest,
            "minimum_lower_bound": 0.75,
        },
    }
    signed_mandate = mandate.issue(
        issuer,
        "did:legal:acme-ou",
        "agent://acme/invoice-bot#1",
        scope,
        H(action),
        pub_jwk(holder),
        ttl=300,
        jti="m-cert-demo",
    )
    presentation = mandate.present(signed_mandate, holder, nonce, audience)
    outcome = {"ok": True, **action["args"]}
    evidence = attester.attest(outcome, nonce, attestation_key)
    args = (
        action,
        signed_mandate,
        presentation,
        evidence,
        pub_jwk(issuer),
        sam,
        trusted_list,
        set(),
        nonce,
        audience,
    )
    return args, signed_mandate


def digest_from_verified_mandate(args: tuple) -> str:
    claims = mandate.verify_presentation(
        args[1], args[2], args[4], args[8], args[9]
    )
    return claims["scope"]["uncertainty_policy"]["certificate_payload_sha256"]


def main() -> None:
    envelope = load_envelope()
    digest = envelope["signature"]["payload_sha256"]

    allow_policy = policy(envelope, 0.75)
    args, signed_mandate = action_material(digest, amount=100.0, cap=500.0)
    mandate_digest_binding = digest_from_verified_mandate(args)
    allow, bound_action_seal = certificate_guarded_activate(
        envelope,
        allow_policy,
        mandate_certificate_payload_sha256=mandate_digest_binding,
        as_of_utc=AS_OF,
        action_gate=aaam.activate,
        action_gate_args=args,
    )
    assert bound_action_seal is not None
    assert bound_action_seal["mandate_digest"] == Hs(signed_mandate)

    escalate = evaluate_certificate(
        envelope,
        policy(envelope, 0.79),
        mandate_certificate_payload_sha256=digest,
        as_of_utc=AS_OF,
    )
    expired = evaluate_certificate(
        envelope,
        allow_policy,
        mandate_certificate_payload_sha256=digest,
        as_of_utc=datetime(2026, 10, 26, tzinfo=timezone.utc),
    )

    tampered_envelope = copy.deepcopy(envelope)
    tampered_envelope["certificate"]["measurement"]["composed_score"]["value"] = 1.0
    tampered = evaluate_certificate(
        tampered_envelope,
        allow_policy,
        mandate_certificate_payload_sha256=digest,
        as_of_utc=AS_OF,
    )

    over_scope_args, _ = action_material(digest, amount=600.0, cap=500.0)
    over_scope_binding = digest_from_verified_mandate(over_scope_args)
    try:
        certificate_guarded_activate(
            envelope,
            allow_policy,
            mandate_certificate_payload_sha256=over_scope_binding,
            as_of_utc=AS_OF,
            action_gate=aaam.activate,
            action_gate_args=over_scope_args,
        )
        raise AssertionError("ordinary action gate unexpectedly allowed over-scope action")
    except AAAMReject as exc:
        action_gate_result = f"DENY: {exc}"

    report = {
        "publication_example_only": True,
        "certificate_policy_0.75": allow.verdict,
        "certificate_policy_0.79": escalate.verdict,
        "expired_certificate": expired.verdict,
        "tampered_certificate": tampered.verdict,
        "certificate_pass_but_action_over_scope": action_gate_result,
        "certificate_payload_bound_inside_signed_mandate": (
            bound_action_seal["mandate_digest"] == Hs(signed_mandate)
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
