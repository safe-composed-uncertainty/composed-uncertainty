#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from certificate_policy_gate import (
    CertificatePolicy,
    certificate_guarded_activate,
    evaluate_certificate,
)


HERE = Path(__file__).resolve().parent
CERTIFICATE_PATH = HERE.parent / "certificate-examples" / "real-chain.json"
AS_OF = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


class CertificatePolicyGateTests(unittest.TestCase):
    def setUp(self):
        self.envelope = json.loads(CERTIFICATE_PATH.read_text())
        certificate = self.envelope["certificate"]
        self.digest = self.envelope["signature"]["payload_sha256"]
        self.base = dict(
            policy_id="test-policy",
            expected_certificate_type="chain",
            expected_system_id=certificate["subject"]["system_id"],
            expected_substrate=certificate["subject"]["substrate"],
            accepted_interval_methods=("first_order_delta",),
            accepted_uncertainty_scopes=("fixed_artifact_evaluation",),
            trusted_public_keys=(self.envelope["signature"]["public_key_hex"],),
            allow_publication_example_key=True,
        )

    def make_policy(self, threshold=0.75, **changes):
        return CertificatePolicy(
            minimum_lower_bound=threshold,
            **{**self.base, **changes},
        )

    def evaluate(self, policy=None, envelope=None, digest=None, as_of=AS_OF):
        return evaluate_certificate(
            envelope or self.envelope,
            policy or self.make_policy(),
            mandate_certificate_payload_sha256=digest or self.digest,
            as_of_utc=as_of,
        )

    def test_allow_when_lower_bound_clears_policy(self):
        self.assertEqual(self.evaluate().verdict, "ALLOW")

    def test_escalate_when_interval_crosses_policy(self):
        self.assertEqual(self.evaluate(self.make_policy(0.79)).verdict, "ESCALATE")

    def test_deny_when_upper_bound_is_below_policy(self):
        self.assertEqual(self.evaluate(self.make_policy(0.85)).verdict, "DENY")

    def test_expired_certificate_is_denied(self):
        decision = self.evaluate(as_of=datetime(2026, 10, 26, tzinfo=timezone.utc))
        self.assertEqual(decision.verdict, "DENY")
        self.assertIn("recalibrate_by_utc", decision.reason)

    def test_tamper_is_denied(self):
        tampered = copy.deepcopy(self.envelope)
        tampered["certificate"]["measurement"]["composed_score"]["value"] = 1.0
        self.assertEqual(self.evaluate(envelope=tampered).verdict, "DENY")

    def test_signature_byte_tamper_is_denied(self):
        tampered = copy.deepcopy(self.envelope)
        signature = tampered["signature"]["signature_hex"]
        tampered["signature"]["signature_hex"] = (
            ("0" if signature[0] != "0" else "1") + signature[1:]
        )
        self.assertEqual(self.evaluate(envelope=tampered).verdict, "DENY")

    def test_wrong_subject_is_denied(self):
        decision = self.evaluate(
            self.make_policy(expected_system_id="another-system")
        )
        self.assertEqual(decision.verdict, "DENY")

    def test_mandate_must_bind_exact_payload(self):
        self.assertEqual(self.evaluate(digest="0" * 64).verdict, "DENY")

    def test_publication_key_requires_explicit_opt_in(self):
        decision = self.evaluate(
            self.make_policy(allow_publication_example_key=False)
        )
        self.assertEqual(decision.verdict, "DENY")

    def test_unaccepted_interval_method_is_denied(self):
        tampered = copy.deepcopy(self.envelope)
        tampered["certificate"]["measurement"]["composed_score"][
            "interval_method"
        ] = "bayesian_credible"
        decision = self.evaluate(envelope=tampered)
        self.assertEqual(decision.verdict, "DENY")
        # tampering the method also breaks the signature first; assert the
        # policy check independently with a policy that accepts nothing else
        decision = self.evaluate(
            self.make_policy(accepted_interval_methods=("paired_bootstrap_percentile",))
        )
        self.assertEqual(decision.verdict, "DENY")
        self.assertIn("interval method", decision.reason)

    def test_unaccepted_uncertainty_scope_is_denied(self):
        decision = self.evaluate(
            self.make_policy(accepted_uncertainty_scopes=("training_procedure",))
        )
        self.assertEqual(decision.verdict, "DENY")
        self.assertIn("scope", decision.reason)

    def test_action_gate_not_called_unless_policy_allows(self):
        calls = []

        def action_gate():
            calls.append(True)
            return "activated"

        decision, result = certificate_guarded_activate(
            self.envelope,
            self.make_policy(0.79),
            mandate_certificate_payload_sha256=self.digest,
            as_of_utc=AS_OF,
            action_gate=action_gate,
            action_gate_args=(),
        )
        self.assertEqual(decision.verdict, "ESCALATE")
        self.assertIsNone(result)
        self.assertEqual(calls, [])

        decision, result = certificate_guarded_activate(
            self.envelope,
            self.make_policy(0.75),
            mandate_certificate_payload_sha256=self.digest,
            as_of_utc=AS_OF,
            action_gate=action_gate,
            action_gate_args=(),
        )
        self.assertEqual(decision.verdict, "ALLOW")
        self.assertEqual(result, "activated")
        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
