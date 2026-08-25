#!/usr/bin/env python3
"""Bounded certificate-driven policy gate for the publication demonstrator.

The gate qualifies the evaluated system before the ordinary per-action
MachineMandate checks run.  It deliberately does not replace those checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jsonschema
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
SCHEMA_PATH = EXPERIMENT_DIR / "certificate-profile-0.3.schema.json"
CANONICALIZATION = "UTF-8 JSON; sorted keys; separators comma/colon; ensure_ascii=false"
EXAMPLE_KEY_USE = "PUBLICATION EXAMPLE KEY — NEVER OPERATIONAL"


class CertificatePolicyError(ValueError):
    """A certificate failed a cryptographic, binding, freshness, or policy check."""


@dataclass(frozen=True)
class CertificatePolicy:
    policy_id: str
    expected_certificate_type: str
    expected_system_id: str
    expected_substrate: dict[str, str]
    minimum_lower_bound: float
    accepted_interval_methods: tuple[str, ...]
    accepted_uncertainty_scopes: tuple[str, ...]
    trusted_public_keys: tuple[str, ...]
    allow_publication_example_key: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    verdict: str
    reason: str
    certificate_id: str | None = None
    certificate_payload_sha256: str | None = None
    interval95: tuple[float, float] | None = None


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise CertificatePolicyError("certificate time has no UTC offset")
    return parsed.astimezone(timezone.utc)


def verify_envelope(envelope: dict[str, Any], policy: CertificatePolicy) -> str:
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    try:
        validator.validate(envelope)
    except jsonschema.ValidationError as exc:
        raise CertificatePolicyError(f"schema validation failed: {exc.message}") from exc

    certificate = envelope["certificate"]
    signature = envelope["signature"]
    if signature["canonicalization"] != CANONICALIZATION:
        raise CertificatePolicyError("unsupported canonicalization")
    if signature["public_key_hex"] not in policy.trusted_public_keys:
        raise CertificatePolicyError("certificate signer is not trusted by this policy")
    if (
        signature["key_use"] == EXAMPLE_KEY_USE
        and not policy.allow_publication_example_key
    ):
        raise CertificatePolicyError("publication example key is not operationally trusted")

    payload = canonical_bytes(certificate)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != signature["payload_sha256"]:
        raise CertificatePolicyError("payload digest mismatch")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(signature["public_key_hex"])
        )
        public_key.verify(bytes.fromhex(signature["signature_hex"]), payload)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise CertificatePolicyError("certificate signature is invalid") from exc
    return digest


def evaluate_certificate(
    envelope: dict[str, Any],
    policy: CertificatePolicy,
    *,
    mandate_certificate_payload_sha256: str,
    as_of_utc: datetime,
) -> PolicyDecision:
    """Return ALLOW, ESCALATE, or DENY without invoking an action gate."""
    try:
        if as_of_utc.tzinfo is None:
            raise CertificatePolicyError("as_of_utc must be timezone-aware")
        digest = verify_envelope(envelope, policy)
        certificate = envelope["certificate"]

        if digest != mandate_certificate_payload_sha256:
            raise CertificatePolicyError("mandate binds a different certificate payload")
        if certificate["certificate_type"] != policy.expected_certificate_type:
            raise CertificatePolicyError("wrong certificate type")

        subject = certificate["subject"]
        if subject["system_id"] != policy.expected_system_id:
            raise CertificatePolicyError("certificate subject does not match the system")
        if subject["substrate"] != policy.expected_substrate:
            raise CertificatePolicyError("certificate substrate does not match the system")

        interval = certificate["measurement"]["composed_score"]["interval95"]
        method = certificate["measurement"]["composed_score"]["interval_method"]
        if method not in policy.accepted_interval_methods:
            raise CertificatePolicyError("interval method is not accepted by policy")
        scope = certificate["estimation"]["uncertainty_scope"]
        if scope not in policy.accepted_uncertainty_scopes:
            raise CertificatePolicyError("uncertainty scope is not accepted by policy")

        now = as_of_utc.astimezone(timezone.utc)
        issued = parse_utc(certificate["validity"]["issued_utc"])
        recalibrate_by = parse_utc(certificate["validity"]["recalibrate_by_utc"])
        if now < issued:
            raise CertificatePolicyError("certificate is not yet valid")
        if now > recalibrate_by:
            raise CertificatePolicyError("certificate is past recalibrate_by_utc")

        lower, upper = float(interval[0]), float(interval[1])
        common = {
            "certificate_id": certificate["certificate_id"],
            "certificate_payload_sha256": digest,
            "interval95": (lower, upper),
        }
        if lower >= policy.minimum_lower_bound:
            return PolicyDecision(
                "ALLOW",
                "certificate lower confidence bound satisfies policy",
                **common,
            )
        if upper < policy.minimum_lower_bound:
            return PolicyDecision(
                "DENY",
                "certificate upper confidence bound is below policy",
                **common,
            )
        return PolicyDecision(
            "ESCALATE",
            "confidence interval crosses the policy threshold",
            **common,
        )
    except CertificatePolicyError as exc:
        certificate = envelope.get("certificate", {})
        return PolicyDecision(
            "DENY",
            str(exc),
            certificate_id=certificate.get("certificate_id"),
        )


def certificate_guarded_activate(
    envelope: dict[str, Any],
    policy: CertificatePolicy,
    *,
    mandate_certificate_payload_sha256: str,
    as_of_utc: datetime,
    action_gate: Callable[..., Any],
    action_gate_args: tuple[Any, ...],
) -> tuple[PolicyDecision, Any | None]:
    """Run the ordinary action gate only after the certificate policy passes."""
    decision = evaluate_certificate(
        envelope,
        policy,
        mandate_certificate_payload_sha256=mandate_certificate_payload_sha256,
        as_of_utc=as_of_utc,
    )
    if decision.verdict != "ALLOW":
        return decision, None
    return decision, action_gate(*action_gate_args)
