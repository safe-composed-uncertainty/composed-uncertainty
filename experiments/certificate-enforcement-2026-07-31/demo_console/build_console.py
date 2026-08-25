"""Build the 27.08 meeting demo console (static, precomputed, honest).

The console is a single self-contained HTML page with a policy-threshold
slider and scenario buttons (valid / expired / payload tamper / signature
tamper / wrong subject / wrong mandate binding / untrusted key / wrong
scope). EVERY verdict the page can display is precomputed HERE by running
the real, unmodified certificate_policy_gate.evaluate_certificate over the
recorded real-chain certificate — the page itself contains zero
verification logic, only lookup and drawing. The full precomputed matrix is
also written to console-results.json for audit.

Run:  python3 build_console.py     -> console.html + console-results.json
Open console.html in any browser (offline; no server, no dependencies).
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_DIR = HERE.parent
sys.path.insert(0, str(GATE_DIR))

from certificate_policy_gate import (  # noqa: E402
    CertificatePolicy,
    evaluate_certificate,
)

CERTIFICATE_PATH = GATE_DIR.parent / "certificate-examples" / "real-chain.json"
AS_OF = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)      # the meeting
AS_OF_EXPIRED = datetime(2026, 10, 26, 0, 0, tzinfo=timezone.utc)

THRESHOLDS = [round(0.60 + i * 0.005, 3) for i in range(0, 61)]  # 0.600..0.900


def base_policy(envelope, threshold, **changes):
    certificate = envelope["certificate"]
    kwargs = dict(
        policy_id=f"demo-lcb-{threshold:.3f}",
        expected_certificate_type="chain",
        expected_system_id=certificate["subject"]["system_id"],
        expected_substrate=certificate["subject"]["substrate"],
        minimum_lower_bound=threshold,
        accepted_interval_methods=("first_order_delta",),
        accepted_uncertainty_scopes=("fixed_artifact_evaluation",),
        trusted_public_keys=(envelope["signature"]["public_key_hex"],),
        allow_publication_example_key=True,
    )
    kwargs.update(changes)
    return CertificatePolicy(**kwargs)


def scenario_matrix(envelope):
    digest = envelope["signature"]["payload_sha256"]

    def run(env=None, digest_override=None, as_of=AS_OF, **policy_changes):
        rows = []
        for tau in THRESHOLDS:
            decision = evaluate_certificate(
                env or envelope,
                base_policy(envelope, tau, **policy_changes),
                mandate_certificate_payload_sha256=digest_override or digest,
                as_of_utc=as_of,
            )
            rows.append({"tau": tau, "verdict": decision.verdict,
                         "reason": decision.reason})
        return rows

    tampered_payload = copy.deepcopy(envelope)
    tampered_payload["certificate"]["measurement"]["composed_score"]["value"] = 1.0

    tampered_sig = copy.deepcopy(envelope)
    sig = tampered_sig["signature"]["signature_hex"]
    tampered_sig["signature"]["signature_hex"] = (
        ("0" if sig[0] != "0" else "1") + sig[1:])

    return {
        "valid": {
            "label": "Valid certificate",
            "detail": "recorded real-chain certificate, evaluated 2026-08-27",
            "rows": run(),
        },
        "expired": {
            "label": "Past recalibrate_by",
            "detail": "same certificate evaluated 2026-10-26, after recalibrate_by_utc",
            "rows": run(as_of=AS_OF_EXPIRED),
        },
        "tamper_payload": {
            "label": "Payload tamper",
            "detail": "composed score edited to 1.0 after signing",
            "rows": run(env=tampered_payload),
        },
        "tamper_signature": {
            "label": "Signature tamper",
            "detail": "one signature byte flipped",
            "rows": run(env=tampered_sig),
        },
        "wrong_subject": {
            "label": "Wrong subject",
            "detail": "policy expects a different system_id",
            "rows": run(expected_system_id="another-system"),
        },
        "wrong_mandate": {
            "label": "Wrong mandate binding",
            "detail": "mandate binds a different certificate payload digest",
            "rows": run(digest_override="0" * 64),
        },
        "untrusted_key": {
            "label": "Untrusted signer",
            "detail": "publication example key without explicit opt-in",
            "rows": run(allow_publication_example_key=False),
        },
        "wrong_scope": {
            "label": "Wrong uncertainty scope",
            "detail": "policy accepts only training_procedure certificates",
            "rows": run(accepted_uncertainty_scopes=("training_procedure",)),
        },
    }


HTML = """<!DOCTYPE html>
<meta charset="utf-8">
<title>Uncertainty-certificate policy gate — demo console</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;
      padding:0 1rem;color:#000;background:#fff}
 h1{font-size:1.15rem} .muted{color:#444;font-size:.85rem}
 .scen{margin:.15rem;padding:.45rem .7rem;border:1px solid #000;
       background:#fff;cursor:pointer;font-size:.85rem}
 .scen.active{background:#000;color:#fff}
 #verdict{font-size:2.1rem;font-weight:700;margin:.4rem 0}
 #reason{font-size:.9rem;min-height:2.2em}
 svg{width:100%;height:110px;margin-top:.6rem}
 input[type=range]{width:100%}
 .foot{border-top:1px solid #000;margin-top:1.4rem;padding-top:.6rem;
       font-size:.78rem;color:#333}
</style>
<h1>Signed uncertainty certificate → policy gate</h1>
<p class="muted">Certificate: executed two-stage agentic pipeline, chain
interval <b>[__LCB__, __UCB__]</b>, point __POINT__, method
first_order_delta, scope fixed_artifact_evaluation.</p>
<div id="scenarios"></div>
<p>Policy: minimum lower confidence bound
 <b><span id="tauv"></span></b></p>
<input type="range" id="tau" min="0" max="__TMAX__" step="1">
<div id="verdict"></div>
<div id="reason" class="muted"></div>
<svg id="band" viewBox="0 0 760 110"></svg>
<div class="foot">Every verdict on this page is a precomputed run of the
unmodified <code>certificate_policy_gate.evaluate_certificate</code> over
the recorded certificate (matrix in <code>console-results.json</code>,
generated by <code>build_console.py</code>). This page contains no
verification logic — only lookup and drawing. Research demonstrator, not an
operational trust configuration.</div>
<script>
const DATA = __DATA__;
const THRESHOLDS = __THRESHOLDS__;
const LCB = __LCB__, UCB = __UCB__, POINT = __POINT__;
let scen = "valid";
const scenDiv = document.getElementById("scenarios");
for (const [key, s] of Object.entries(DATA)) {
  const b = document.createElement("button");
  b.className = "scen"; b.id = "b_" + key; b.textContent = s.label;
  b.onclick = () => { scen = key; render(); };
  scenDiv.appendChild(b);
}
const slider = document.getElementById("tau");
slider.oninput = render;
slider.value = Math.round((0.75 - THRESHOLDS[0]) /
                          (THRESHOLDS[1] - THRESHOLDS[0]));
function render() {
  const i = +slider.value, tau = THRESHOLDS[i];
  const row = DATA[scen].rows[i];
  document.getElementById("tauv").textContent = tau.toFixed(3);
  for (const key of Object.keys(DATA))
    document.getElementById("b_" + key).classList.toggle("active", key === scen);
  const v = document.getElementById("verdict");
  v.textContent = row.verdict;
  document.getElementById("reason").textContent =
    DATA[scen].detail + " — " + row.reason;
  const svg = document.getElementById("band");
  const x = s => 40 + (s - 0.55) / (1.0 - 0.55) * 680;
  const cross = row.verdict === "DENY";
  svg.innerHTML =
    `<line x1="40" y1="80" x2="720" y2="80" stroke="black"/>` +
    [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95].map(t =>
      `<line x1="${x(t)}" y1="77" x2="${x(t)}" y2="83" stroke="black"/>
       <text x="${x(t)}" y="97" font-size="10" text-anchor="middle">${t.toFixed(2)}</text>`
    ).join("") +
    `<line x1="${x(LCB)}" y1="45" x2="${x(UCB)}" y2="45"
           stroke="black" stroke-width="3" ${cross ? 'stroke-dasharray="5,4"' : ''}/>` +
    `<line x1="${x(LCB)}" y1="37" x2="${x(LCB)}" y2="53" stroke="black" stroke-width="2"/>` +
    `<line x1="${x(UCB)}" y1="37" x2="${x(UCB)}" y2="53" stroke="black" stroke-width="2"/>` +
    `<circle cx="${x(POINT)}" cy="45" r="5" fill="white" stroke="black" stroke-width="2"/>` +
    `<line x1="${x(tau)}" y1="20" x2="${x(tau)}" y2="70"
           stroke="black" stroke-dasharray="3,3"/>` +
    `<text x="${x(tau)}" y="14" font-size="11" text-anchor="middle">policy ${tau.toFixed(3)}</text>`;
}
render();
</script>
"""


def main():
    envelope = json.loads(CERTIFICATE_PATH.read_text())
    score = envelope["certificate"]["measurement"]["composed_score"]
    lcb, ucb = score["interval95"]
    matrix = scenario_matrix(envelope)

    (HERE / "console-results.json").write_text(
        json.dumps({"as_of": AS_OF.isoformat(), "thresholds": THRESHOLDS,
                    "certificate": CERTIFICATE_PATH.name,
                    "scenarios": matrix}, indent=1, sort_keys=True) + "\n")

    html = (HTML
            .replace("__DATA__", json.dumps(matrix))
            .replace("__THRESHOLDS__", json.dumps(THRESHOLDS))
            .replace("__TMAX__", str(len(THRESHOLDS) - 1))
            .replace("__LCB__", f"{lcb:.4f}")
            .replace("__UCB__", f"{ucb:.4f}")
            .replace("__POINT__", f"{score['value']:.4f}"))
    (HERE / "console.html").write_text(html)

    n_runs = sum(len(s["rows"]) for s in matrix.values())
    print(f"console.html + console-results.json written "
          f"({n_runs} real gate evaluations across {len(matrix)} scenarios "
          f"x {len(THRESHOLDS)} thresholds)")
    # sanity: the valid scenario must show all three verdicts across the grid
    verdicts = {r["verdict"] for r in matrix["valid"]["rows"]}
    assert verdicts == {"ALLOW", "ESCALATE", "DENY"}, verdicts
    for key in ("expired", "tamper_payload", "tamper_signature",
                "wrong_subject", "wrong_mandate", "untrusted_key",
                "wrong_scope"):
        assert {r["verdict"] for r in matrix[key]["rows"]} == {"DENY"}, key
    print("sanity: valid scenario spans ALLOW/ESCALATE/DENY; "
          "all seven mutation scenarios are DENY at every threshold")


if __name__ == "__main__":
    main()
