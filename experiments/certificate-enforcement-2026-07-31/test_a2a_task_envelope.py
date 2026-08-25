#!/usr/bin/env python3
"""Tests for the A2A-task-shaped envelope (no crypto, no network)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from a2a_task_envelope import wrap_certificate_gated_action

TS = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
ACTION = {"type": "pay_invoice", "amount": 950.0, "currency": "EUR"}
CERT = "c" * 64
MANDATE = "m" * 64
SEAL = "s" * 64


def wrap(verdict, seal, reason="r"):
    return wrap_certificate_gated_action(
        task_id="task-1", context_id="ctx-1", action=ACTION,
        certificate_payload_sha256=CERT, mandate_sha256=MANDATE,
        policy_id="research-chain-lcb-0.75", decision_verdict=verdict,
        decision_reason=reason, bound_action_seal_sha256=seal, as_of_utc=TS)


class A2ATaskEnvelopeTests(unittest.TestCase):
    def test_allow_maps_to_completed_with_seal_artifact(self):
        env = wrap("ALLOW", SEAL).to_dict()
        self.assertEqual(env["status"]["state"], "completed")
        ids = [a["artifactId"] for a in env["artifacts"]]
        self.assertIn("bound-action-seal", ids)

    def test_escalate_maps_to_input_required_without_seal(self):
        env = wrap("ESCALATE", None).to_dict()
        self.assertEqual(env["status"]["state"], "input-required")
        ids = [a["artifactId"] for a in env["artifacts"]]
        self.assertNotIn("bound-action-seal", ids)

    def test_deny_maps_to_rejected(self):
        self.assertEqual(wrap("DENY", None).to_dict()["status"]["state"],
                         "rejected")

    def test_allow_without_seal_is_an_error(self):
        with self.assertRaises(ValueError):
            wrap("ALLOW", None)

    def test_seal_on_non_allow_is_an_error(self):
        with self.assertRaises(ValueError):
            wrap("DENY", SEAL)

    def test_history_is_submitted_working_terminal(self):
        env = wrap("ALLOW", SEAL).to_dict()
        self.assertEqual([h["state"] for h in env["history"]],
                         ["submitted", "working", "completed"])

    def test_certificate_and_mandate_digests_travel(self):
        env = wrap("ALLOW", SEAL).to_dict()
        data = {a["artifactId"]: a["parts"][0]["data"]
                for a in env["artifacts"]}
        self.assertEqual(
            data["uncertainty-certificate-binding"]
            ["certificate_payload_sha256"], CERT)
        self.assertEqual(data["machine-mandate-binding"]["mandate_sha256"],
                         MANDATE)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            wrap_certificate_gated_action(
                task_id="t", context_id="c", action=ACTION,
                certificate_payload_sha256=CERT, mandate_sha256=MANDATE,
                policy_id="p", decision_verdict="DENY", decision_reason="r",
                bound_action_seal_sha256=None,
                as_of_utc=datetime(2026, 8, 1))


if __name__ == "__main__":
    unittest.main()
