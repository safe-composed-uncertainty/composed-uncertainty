"""A2A-task-shaped envelope for the certificate-gated action (paper-own).

The 31.07 decision prefers Agent2Agent (A2A) framing for the multi-agent
branch. The demonstrator's action attempt was a direct function call; this
module wraps one certificate-gated activation into an A2A Task-shaped record
so section 8's action step reads as an A2A exchange:

    task_id / context_id / state machine (submitted -> working ->
    completed | rejected) / artifacts with digests.

Deliberately paper-own and minimal: it mirrors the A2A Task object's shape
(task, contextId, status.state, artifacts) without importing an A2A SDK, and
it does NOT reuse the separate, undeposited Task-Effect-Record work. The
envelope carries digests only — the certificate payload digest, the mandate
digest, and (on completion) the Bound Action Seal digest — never the signed
objects themselves, so it adds no new trust claims: every verification stays
in the certificate gate and the six-check MachineMandate gate.

Terminal-state mapping (fixed):
    ALLOW  + action activated  -> "completed"
    ESCALATE                   -> "input-required"  (a human must decide)
    DENY                       -> "rejected"
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


_VERDICT_TO_STATE = {
    "ALLOW": "completed",
    "ESCALATE": "input-required",
    "DENY": "rejected",
}


@dataclass
class A2ATaskEnvelope:
    task_id: str
    context_id: str
    state: str
    created_utc: str
    history: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "task",
            "id": self.task_id,
            "contextId": self.context_id,
            "status": {"state": self.state,
                       "timestamp": self.history[-1]["timestamp"]
                       if self.history else self.created_utc},
            "history": self.history,
            "artifacts": self.artifacts,
        }


def wrap_certificate_gated_action(
    *,
    task_id: str,
    context_id: str,
    action: dict,
    certificate_payload_sha256: str,
    mandate_sha256: str,
    policy_id: str,
    decision_verdict: str,
    decision_reason: str,
    bound_action_seal_sha256: str | None,
    as_of_utc: datetime,
) -> A2ATaskEnvelope:
    """Wrap one already-evaluated certificate-gated activation into an A2A
    Task-shaped envelope. All inputs are outputs of the existing gates; this
    function verifies nothing and must stay that way."""
    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    if decision_verdict not in _VERDICT_TO_STATE:
        raise ValueError(f"unknown verdict {decision_verdict!r}")
    if decision_verdict == "ALLOW" and bound_action_seal_sha256 is None:
        raise ValueError("an ALLOW that activated must carry the seal digest")
    if decision_verdict != "ALLOW" and bound_action_seal_sha256 is not None:
        raise ValueError("only an activated ALLOW carries a seal digest")

    ts = as_of_utc.astimezone(timezone.utc).isoformat()
    state = _VERDICT_TO_STATE[decision_verdict]
    history = [
        {"state": "submitted", "timestamp": ts,
         "message": {"role": "user", "parts": [
             {"kind": "data", "data": {"action_sha256": _digest(action),
                                       "policy_id": policy_id}}]}},
        {"state": "working", "timestamp": ts,
         "message": {"role": "agent", "parts": [
             {"kind": "data", "data": {
                 "certificate_payload_sha256": certificate_payload_sha256,
                 "mandate_sha256": mandate_sha256}}]}},
        {"state": state, "timestamp": ts,
         "message": {"role": "agent", "parts": [
             {"kind": "text", "text": decision_reason}]}},
    ]
    artifacts = [
        {"artifactId": "uncertainty-certificate-binding",
         "parts": [{"kind": "data", "data": {
             "certificate_payload_sha256": certificate_payload_sha256,
             "policy_id": policy_id,
             "verdict": decision_verdict}}]},
        {"artifactId": "machine-mandate-binding",
         "parts": [{"kind": "data", "data": {
             "mandate_sha256": mandate_sha256}}]},
    ]
    if bound_action_seal_sha256 is not None:
        artifacts.append(
            {"artifactId": "bound-action-seal",
             "parts": [{"kind": "data", "data": {
                 "bound_action_seal_sha256": bound_action_seal_sha256}}]})
    return A2ATaskEnvelope(task_id=task_id, context_id=context_id,
                           state=state, created_utc=ts,
                           history=history, artifacts=artifacts)
