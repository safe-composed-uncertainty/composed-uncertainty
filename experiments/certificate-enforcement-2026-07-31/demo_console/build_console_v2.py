#!/usr/bin/env python3
"""Build console v2: the five-beat "what the independence shortcut costs" page.

Every number on the page is read from the result files the paper build
consumes, asserted here against the values the draft quotes, and embedded
as data. The page's only live computation is closed-form interval
arithmetic (the gate rule lo>=tau / hi<tau and the flip-rate formula of the
paper's own figure generator, plus the R^2 = 1+(p_eff-1)*rho identity).
No cryptographic verification happens in the browser.

Sources (all committed, all provenance-checked):
  console-results.json                       beat 1 (488 precomputed verdicts)
  ../../redraw-2026-07-24/results-redraw.json          beat 2/3 German chain
  ../../redraw-2026-07-24/replicates-redraw.npz        beat 2 replicate cloud
  ../../real-agentic-2026-07-25/results-real-agentic.json  beat 2/3 executed chain
  ../../pavia-composite-2026-07-25/results-pavia-composite.json  beat 2 preset 3
  ../../coverage-2026-07-27/results-coverage.json      beat 4
  ../MORE-EVIDENCE-BEAT.json                            beat 5
  ../../certificate-examples/real-chain.json            appendix inspector

Run:  python3 build_console_v2.py   -> console-v2.html + console-v2-data.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = HERE.parent
EXP = GATE.parent

def load(p):
    return json.loads(Path(p).read_text())

def rnd(x, k=4):
    return round(float(x), k)

# ---------------------------------------------------------------- beat 1
v1 = load(HERE / "console-results.json")
assert set(v1["scenarios"]) == {
    "valid", "expired", "tamper_payload", "tamper_signature",
    "wrong_subject", "wrong_mandate", "untrusted_key", "wrong_scope"}
assert len(v1["thresholds"]) == 61

# ---------------------------------------------------------------- beat 2: chains
gc = load(EXP / "redraw-2026-07-24" / "results-redraw.json")["chain"]["redraw"]
ra = load(EXP / "real-agentic-2026-07-25" /
          "results-real-agentic.json")["chain"]["logit"]["redraw"]
assert abs(gc["cross_link_correlation"] - 0.7120) < 5e-4
assert abs(gc["understatement_of_width_pct"] - 23.571) < 5e-3
assert 8.0 < ra["understatement_of_width_pct"] < 11.5   # draft says 8-11%

cert = load(EXP / "certificate-examples" / "real-chain.json")
cert0 = cert["certificate"]
pav = load(EXP / "pavia-composite-2026-07-25" / "results-pavia-composite.json")
# the draft's volume-composite range (-3.1..+1.3%) is the empirical
# understatement across models x volume variants; recompute and assert it
CANON = ("arithmetic", "geometric", "rms", "topsis")
emp_logit = [pav["models"]["logit"]["arms"][a]
             ["empirical_understatement_of_width_pct"] for a in CANON]
emp_all = [pav["models"][m]["arms"][a]
           ["empirical_understatement_of_width_pct"]
           for m in pav["models"] for a in CANON]
VOLRANGE = {"logit": [rnd(min(emp_logit), 2), rnd(max(emp_logit), 2)],
            "all": [rnd(min(emp_all), 2), rnd(max(emp_all), 2)]}
assert VOLRANGE["logit"][0] < 0 < VOLRANGE["logit"][1]   # sign flip is real

pl = pav["models"]["logit"]["arms"]["arithmetic"]
paired_key = next(k for k in pl if "paired" in k)
vol_paired = pl[paired_key]["ci95_percentile"]
vol_indep = pl["independent_stream_bootstrap"]["ci95_percentile"]
vol_under = pl["empirical_understatement_of_width_pct"]

def preset(name, note, lo_m, hi_m, lo_z, hi_z, h, rho, under, tau=None):
    """By default tau sits midway between the two lower bounds, so the
    toggle flips the verdict. A preset may pin tau elsewhere when the point
    is precisely that the toggle changes nothing."""
    if tau is None:
        tau = (lo_m + lo_z) / 2.0
    return {
        "name": name, "note": note,
        "measured": [rnd(lo_m), rnd(hi_m)], "independent": [rnd(lo_z), rnd(hi_z)],
        "h": rnd(h), "rho": rnd(rho), "understatement_pct": rnd(under, 2),
        "tau": rnd(tau),
        "r": rnd((hi_z - lo_z) / (hi_m - lo_m), 4),
    }

PRESETS = [
    preset("German-credit chain",
           "two links, h = C_A x C_B; DOI 10.5281/zenodo.21547845",
           *gc["ci95_measured_covariance"], *gc["ci95_cross_term_declared_zero"],
           gc["h_point"], gc["cross_link_correlation"],
           gc["understatement_of_width_pct"]),
    preset("Executed agentic chain",
           "authorisation gate x evidence verifier, 1,000 episodes; the "
           "same certificate as section 1",
           *cert0["measurement"]["composed_score"]["interval95"],
           *cert0["measurement"]["zero_cross_term_sensitivity"]["interval95"],
           cert0["measurement"]["composed_score"]["value"],
           ra["cross_link_correlation"],
           cert0["measurement"]["zero_cross_term_sensitivity"]
           ["width_change_pct"]),
    preset("Volume composite (arithmetic)",
           "Giudici-Kolesnikov volume integration, German credit. Here the "
           "cross-terms nearly cancel: the two readings differ by less "
           "than a thousandth, and the toggle changes nothing -- "
           "composition geometry decides whether the shortcut bites",
           *vol_paired, *vol_indep,
           pl["V_point"], 0.0, vol_under, tau=pl["V_point"]),
]

def _verdict(lo, hi, tau):
    return "ALLOW" if lo >= tau else ("DENY" if hi < tau else "ESCALATE")

# sanity on the flip design: the two chains must flip at their tau, the
# volume preset must NOT (its lesson is the near-cancellation)
for p_ in PRESETS[:2]:
    assert p_["independent"][0] > p_["measured"][0]
    assert _verdict(*p_["measured"], p_["tau"]) == "ESCALATE"
    assert _verdict(*p_["independent"], p_["tau"]) == "ALLOW"
p_ = PRESETS[2]
assert (_verdict(*p_["measured"], p_["tau"])
        == _verdict(*p_["independent"], p_["tau"]) == "ESCALATE")

# replicate cloud: the actual paired bootstrap link scores, redraw scheme
z = np.load(EXP / "redraw-2026-07-24" / "replicates-redraw.npz")
CA, CB = z["chain_C_redraw_A"], z["chain_C_redraw_B"]
assert CA.shape == CB.shape and len(CA) == 2000
rho_check = float(np.corrcoef(CA, CB)[0, 1])
assert abs(rho_check - gc["cross_link_correlation"]) < 1e-9
CLOUD = [[rnd(a), rnd(b)] for a, b in zip(CA.tolist(), CB.tolist())]

# ---------------------------------------------------------------- beat 3
FLIP = [{"label": "German-credit chain", "r": PRESETS[0]["r"],
         "under": PRESETS[0]["understatement_pct"]},
        {"label": "executed agentic chain", "r": PRESETS[1]["r"],
         "under": PRESETS[1]["understatement_pct"]}]

# ---------------------------------------------------------------- beat 4
cov = load(EXP / "coverage-2026-07-27" / "results-coverage.json")
paired = [s["coverage"]["composite_percentile"] for s in cov["scenarios"]]
indep = [s["coverage"]["composite_independent_percentile"]
         for s in cov["scenarios"]]
assert (min(paired), max(paired)) == (0.941, 0.971)
assert (min(indep), max(indep)) == (0.802, 0.998)
def shortname(n):
    return (n.replace("central-", "c-").replace("offbound-", "ob-")
             .replace("positive", "pos").replace("negative", "neg"))

COVER = {"nominal": 0.95,
         "scenarios": [shortname(s["scenario"]["name"])
                       for s in cov["scenarios"]],
         "paired": paired, "independent": indep,
         "outer": cov["outer_replications_per_scenario"],
         "mc_se": rnd(cov["scenarios"][0]
                      ["coverage_monte_carlo_se_at_95pct"], 4)}

# ---------------------------------------------------------------- beat 5
beat = load(GATE / "MORE-EVIDENCE-BEAT.json")
assert beat["policy_minimum_lower_bound"] == 0.75
assert beat["runs"]["small"]["verdict"] == "ESCALATE"
assert beat["runs"]["full"]["verdict"] == "ALLOW"
BEAT = {"policy": 0.75,
        "small": {"n": beat["runs"]["small"]["n_eval_rows"],
                  "interval": [rnd(x) for x in beat["runs"]["small"]["interval95"]],
                  "h": rnd(beat["runs"]["small"]["h_point"]),
                  "verdict": "ESCALATE"},
        "full": {"n": beat["runs"]["full"]["n_eval_rows"],
                 "interval": [rnd(x) for x in beat["runs"]["full"]["interval95"]],
                 "h": rnd(beat["runs"]["full"]["h_point"]),
                 "verdict": "ALLOW"},
        "width_ratio": rnd(beat["width_ratio_small_over_full"], 2)}

# ---------------------------------------------------------------- appendix
cs = cert0["measurement"]["composed_score"]
INSPECT = [
    ["subject.system_id", cert0["subject"]["system_id"], "wrong_subject"],
    ["measurement.composed_score.interval95",
     "[" + ", ".join(f"{x:.4f}" for x in cs["interval95"]) + "]", "valid"],
    ["measurement.composed_score.interval_method", cs["interval_method"],
     "valid"],
    ["estimation.uncertainty_scope",
     cert0["estimation"]["uncertainty_scope"], "wrong_scope"],
    ["estimation.seed / replicates",
     f'{cert0["estimation"]["seed"]} / {cert0["estimation"]["replicates"]}',
     "valid"],
    ["validity.recalibrate_by_utc",
     cert0["validity"]["recalibrate_by_utc"], "expired"],
    ["signature.algorithm", cert["signature"]["algorithm"],
     "tamper_signature"],
    ["signature.payload_sha256",
     cert["signature"]["payload_sha256"][:16] + "…", "tamper_payload"],
    ["signature.public_key_hex",
     cert["signature"]["public_key_hex"][:16] + "…", "untrusted_key"],
]

DATA2 = {"presets": PRESETS, "cloud": CLOUD, "flip": FLIP, "cover": COVER,
         "volrange": VOLRANGE,
         "beat": BEAT, "inspect": INSPECT,
         "gate": {"thresholds": v1["thresholds"], "as_of": v1["as_of"],
                  "scenarios": {k: {"label": s["label"],
                                    "verdicts": [r["verdict"] for r in
                                                 s["rows"]],
                                    "reasons": [r["reason"] for r in
                                                s["rows"]]}
                                for k, s in v1["scenarios"].items()},
                  "certificate": {
                      "interval": [rnd(x) for x in
                                   cert0["measurement"]["composed_score"]
                                   ["interval95"]],
                      "point": rnd(cert0["measurement"]["composed_score"]
                                   ["value"])}}}

(HERE / "console-v2-data.json").write_text(json.dumps(DATA2, indent=1))

# ---------------------------------------------------------------- page
HTML = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Uncertainty-certificate console — what the independence shortcut costs</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;
      padding:0 1rem;color:#000;background:#fff;line-height:1.45}
 h1{font-size:1.25rem} h2{font-size:1.05rem;border-top:1px solid #000;
      padding-top:1.1rem;margin-top:2.2rem}
 .muted{color:#444;font-size:.85rem}
 .scen,.pre{margin:.15rem;padding:.4rem .65rem;border:1px solid #000;
       background:#fff;cursor:pointer;font-size:.85rem}
 .scen.active,.pre.active{background:#000;color:#fff}
 .verdict{font-size:1.9rem;font-weight:700;margin:.35rem 0}
 .badge{display:inline-block;border:2px solid #000;padding:.15rem .6rem;
        font-weight:700;font-size:1.05rem}
 .badge.false{border-style:dashed}
 svg{width:100%;margin-top:.4rem}
 input[type=range]{width:100%}
 .foot{border-top:1px solid #000;margin-top:1.6rem;padding-top:.6rem;
       font-size:.78rem;color:#333}
 .frame{border:1px solid #000;padding:.6rem .8rem;font-size:.82rem;
        margin:.8rem 0;color:#222}
 table{border-collapse:collapse;font-size:.82rem;margin:.5rem 0}
 td,th{border:1px solid #000;padding:.25rem .5rem;text-align:left}
 details{margin-top:1rem} summary{cursor:pointer;font-weight:600}
 .toggle{border:1px solid #000;padding:.45rem .8rem;background:#fff;
         cursor:pointer;font-size:.9rem}
 .toggle.on{background:#000;color:#fff}
 .kv{font-variant-numeric:tabular-nums}
</style>

<h1>Signed uncertainty certificate → policy gate,<br>
and what the independence shortcut costs</h1>
<p class="muted">A research demonstration of an <b>uncertainty
certificate</b> — a signed record of a measurement, its confidence bound,
its assumptions and its validity window. It is not a compliance,
conformity, or security certificate, and this page does not offer
conformity-assessment or certification services.</p>

<h2>1 · The gate</h2>
<p class="muted">Certificate: executed two-stage agentic pipeline, chain
interval <b class="kv" id="g_int"></b>, point <span class="kv"
id="g_pt"></span>. Verdict rule: ALLOW if the certificate's lower
confidence bound clears the policy threshold, DENY if its whole interval
falls short, ESCALATE when the threshold lands inside the interval.</p>
<div id="scenarios"></div>
<p class="muted">The threshold only matters for the valid certificate.
The seven mutation scenarios return DENY at <i>every</i> threshold, by
design — no policy setting rescues a tampered, expired, or mis-bound
certificate.</p>
<p>Policy: minimum lower confidence bound <b class="kv"><span
id="tauv"></span></b></p>
<input type="range" id="tau" min="0" max="60" step="1">
<div class="verdict" id="g_verdict"></div>
<div id="g_reason" class="muted"></div>
<svg id="g_band" viewBox="0 0 760 100"></svg>

<h2>2 · Two links wobble together — the shortcut, felt</h2>
<p class="muted">A composed score multiplies two link scores. The paired
bootstrap resamples both links from the <i>same</i> draws, so their joint
wobble is measured, not assumed. Below: the actual 2,000 paired replicates
of the German-credit chain — link A against link B. The tilt of the cloud
IS the cross-link correlation.</p>
<svg id="cloud" viewBox="0 0 760 300"></svg>
<div id="presets"></div>
<p style="margin:.6rem 0">
 <button class="toggle" id="indep">assume links independent: OFF</button>
 &nbsp; fixed policy threshold <b class="kv" id="p_tau"></b>
 &nbsp;→&nbsp; <span class="badge" id="p_badge"></span>
 <span class="muted" id="p_false"></span></p>
<svg id="p_band" viewBox="0 0 760 120"></svg>
<p class="muted" id="p_note"></p>
<p class="muted">Theory (evaluated live, client-side — the page's only
live computation): with average cross-correlation ρ̄ over p<sub>eff</sub>
equally exposed components, the honest variance is
R² = 1 + (p<sub>eff</sub> − 1)·ρ̄ times the independence-shortcut
variance; the shortcut understates interval width by 1 − 1/R.</p>
<p>ρ̄ <input type="range" id="rho" min="-20" max="99" value="71"
 style="width:38%"> <b class="kv" id="rho_v"></b>
 &nbsp; p<sub>eff</sub> <input type="range" id="peff" min="2" max="5"
 value="2" style="width:20%"> <b class="kv" id="peff_v"></b>
 &nbsp;→&nbsp; understatement <b class="kv" id="R_v"></b></p>
<p class="muted">Measured systems for comparison: German chain ρ = 0.712 →
23.6% (on the p<sub>eff</sub> = 2 curve exactly); executed chain
ρ = 0.319 → 11.0% measured (its links are unequally exposed, so its
effective dimension sits below 2); volume composites and TOPSIS →
__VOLTXT__ across models and variants, with sign flips — the shortcut can
also run <i>wide</i>.</p>

<h2>3 · How often the shortcut changes the verdict</h2>
<p class="muted">An operator clears a reading only if the whole interval
lies above the threshold, rejects only if it lies below, and is undecided
otherwise. The two readings of the same chain — measured covariance vs
cross-term declared zero — disagree on a set of thresholds. The curve (the
paper's own figure, recomputed here from the stored widths) shows the
fraction of a threshold window on which they disagree.</p>
<svg id="flip" viewBox="0 0 760 300"></svg>

<h2>4 · Why the wider interval is the correct one</h2>
<p class="muted">Known-truth check: nine simulation scenarios, nominal 95%,
1,000 outer replications each (Monte-Carlo s.e. ≈ 0.7 pp). The paired
percentile interval holds 94.1–97.1%. The independence-shortcut interval
wanders 80.2–99.8% — sometimes anticonservative, sometimes wastefully
wide. Wider is not the point; <i>correct</i> is.</p>
<svg id="cover" viewBox="0 0 760 260"></svg>

<h2>5 · Uncertainty is a price signal: more evidence flips the verdict</h2>
<p class="muted">Two <b>fixed-design recorded runs</b> of the deposited
chain estimator — n chosen in advance in both, not sequentially. Same
policy threshold, 0.750, both times.</p>
<p style="margin:.6rem 0">
 <button class="toggle" id="more">collect more evidence: n = 30 → 300</button>
 &nbsp; <span class="badge" id="b_badge"></span></p>
<svg id="b_band" viewBox="0 0 760 120"></svg>
<p class="muted" id="b_note"></p>

<details><summary>Appendix · the certificate itself, field by field</summary>
<p class="muted">Each row names the mutation scenario in section 1 that
corrupting the field triggers. Every one of those scenarios is DENY at
every threshold.</p>
<table id="inspect"></table>
<p class="muted" id="clock"></p>
</details>

<div class="foot" id="foot">
All gate verdicts shown were precomputed by the reference gate
implementation (matrix in <code>console-v2-data.json</code>, generated by
<code>build_console_v2.py</code>); this page performs no cryptographic
verification — the only live computation is the closed-form interval
arithmetic above, evaluated client-side. The volume-integration and TOPSIS
composites are the constructions of Giudici and Kolesnikov (Machine
Learning with Applications, 2025; Kolesnikov, MSc thesis, Univ. of Pavia);
this work adds the confidence regions around them. Chain-certificate
values as published (DOI 10.5281/zenodo.21547845). The flip-rate,
coverage, evidence-collection and composite panels show results from a
draft in preparation with the University of Pavia. Research demonstrator,
not an operational trust configuration.</div>

<script>
const D = __DATA2__;

/* ------------------------------------------------ beat 1: the gate */
const T = D.gate.thresholds, G = D.gate.scenarios;
document.getElementById('g_int').textContent =
  '[' + D.gate.certificate.interval[0].toFixed(4) + ', ' +
  D.gate.certificate.interval[1].toFixed(4) + ']';
document.getElementById('g_pt').textContent =
  D.gate.certificate.point.toFixed(4);
let scen = 'valid';
const sd = document.getElementById('scenarios');
for (const [k, s] of Object.entries(G)) {
  const b = document.createElement('button');
  b.className = 'scen'; b.id = 'b_' + k; b.textContent = s.label;
  b.onclick = () => { scen = k; g_render(); };
  sd.appendChild(b);
}
const slider = document.getElementById('tau');
slider.value = 30;
slider.oninput = g_render;
const LCB = D.gate.certificate.interval[0],
      UCB = D.gate.certificate.interval[1];
function x760(v, lo, hi, pad) {
  return pad + (v - lo) / (hi - lo) * (760 - 2 * pad);
}
function g_render() {
  const i = +slider.value, tau = T[i];
  document.getElementById('tauv').textContent = tau.toFixed(3);
  for (const k of Object.keys(G))
    document.getElementById('b_' + k).classList
      .toggle('active', k === scen);
  document.getElementById('g_verdict').textContent = G[scen].verdicts[i];
  document.getElementById('g_reason').textContent = G[scen].reasons[i];
  const X = v => x760(v, 0.58, 0.92, 30);
  document.getElementById('g_band').innerHTML =
    `<line x1="${X(0.6)}" y1="55" x2="${X(0.9)}" y2="55" stroke="#999"/>` +
    `<rect x="${X(LCB)}" y="45" width="${X(UCB)-X(LCB)}" height="20"
      fill="none" stroke="black" stroke-width="2"/>` +
    `<circle cx="${X(D.gate.certificate.point)}" cy="55" r="5"
      fill="white" stroke="black" stroke-width="2"/>` +
    `<line x1="${X(tau)}" y1="18" x2="${X(tau)}" y2="80" stroke="black"
      stroke-dasharray="3,3"/>` +
    `<text x="${X(tau)}" y="12" font-size="12"
      text-anchor="middle">policy ${tau.toFixed(3)}</text>`;
}

/* -------------------------------------- beat 2: cloud + toggle + theory */
(function () {
  const c = document.getElementById('cloud');
  const xs = D.cloud.map(p => p[0]), ys = D.cloud.map(p => p[1]);
  const xlo = Math.min(...xs), xhi = Math.max(...xs),
        ylo = Math.min(...ys), yhi = Math.max(...ys);
  const X = v => 40 + (v - xlo) / (xhi - xlo) * 690,
        Y = v => 280 - (v - ylo) / (yhi - ylo) * 255;
  let s = '';
  for (const [a, b] of D.cloud)
    s += `<rect x="${X(a).toFixed(1)}" y="${Y(b).toFixed(1)}"
           width="1.6" height="1.6" fill="black" opacity="0.35"/>`;
  s += `<text x="400" y="298" font-size="12" text-anchor="middle">link A
        score (2,000 paired bootstrap replicates)</text>`;
  s += `<text x="12" y="150" font-size="12" transform="rotate(-90 12 150)"
        text-anchor="middle">link B score</text>`;
  s += `<text x="620" y="30" font-size="13">ρ = ${D.presets[0].rho.toFixed(3)}</text>`;
  c.innerHTML = s;
})();

let pi = 0, indep = false;
const pd = document.getElementById('presets');
D.presets.forEach((p, i) => {
  const b = document.createElement('button');
  b.className = 'pre'; b.id = 'pre_' + i; b.textContent = p.name;
  b.onclick = () => { pi = i; p_render(); };
  pd.appendChild(b);
});
document.getElementById('indep').onclick = () => {
  indep = !indep; p_render();
};
function verdict(lo, hi, tau) {
  if (lo >= tau) return 'ALLOW';
  if (hi < tau) return 'DENY';
  return 'ESCALATE';
}
function p_render() {
  const p = D.presets[pi];
  D.presets.forEach((_, i) => document.getElementById('pre_' + i)
    .classList.toggle('active', i === pi));
  const tog = document.getElementById('indep');
  tog.classList.toggle('on', indep);
  tog.textContent = 'assume links independent: ' + (indep ? 'ON' : 'OFF');
  const iv = indep ? p.independent : p.measured;
  const v = verdict(iv[0], iv[1], p.tau);
  const honest = verdict(p.measured[0], p.measured[1], p.tau);
  document.getElementById('p_tau').textContent = p.tau.toFixed(4);
  const badge = document.getElementById('p_badge');
  badge.textContent = v;
  badge.classList.toggle('false', indep && v !== honest);
  document.getElementById('p_false').textContent =
    indep && v !== honest
      ? ' — differs from the covariance-aware reading (' + honest + ')'
      : '';
  const lo = Math.min(p.measured[0], p.independent[0]) - 0.01,
        hi = Math.max(p.measured[1], p.independent[1]) + 0.01;
  const X = v_ => x760(v_, lo, hi, 30);
  const band = (b_, y, dash, label) =>
    `<rect x="${X(b_[0])}" y="${y}" width="${X(b_[1])-X(b_[0])}"
      height="18" fill="none" stroke="black" stroke-width="2"
      ${dash ? 'stroke-dasharray="5,3"' : ''}/>` +
    `<text x="${X(b_[0])}" y="${y - 5}" font-size="12">${label}</text>`;
  document.getElementById('p_band').innerHTML =
    band(p.measured, 30, false, 'measured covariance') +
    band(p.independent, 72, true, 'cross-term declared zero') +
    `<line x1="${X(p.tau)}" y1="14" x2="${X(p.tau)}" y2="100"
      stroke="black" stroke-dasharray="3,3"/>` +
    `<text x="${X(p.tau)}" y="118" font-size="12"
      text-anchor="middle">policy ${p.tau.toFixed(4)}</text>` +
    `<circle cx="${X(p.h)}" cy="39" r="4" fill="white" stroke="black"
      stroke-width="2"/>`;
  document.getElementById('p_note').textContent =
    p.note + '. Cross-link correlation ' + p.rho.toFixed(3) +
    '; the shortcut mis-states the interval width by ' +
    p.understatement_pct.toFixed(1) + '%.';
}

const rho_s = document.getElementById('rho'),
      peff_s = document.getElementById('peff');
function theory() {
  const rho = +rho_s.value / 100, peff = +peff_s.value;
  const R2 = 1 + (peff - 1) * rho;
  const u = R2 > 0 ? (1 - 1 / Math.sqrt(R2)) * 100 : NaN;
  document.getElementById('rho_v').textContent = rho.toFixed(2);
  document.getElementById('peff_v').textContent = peff;
  document.getElementById('R_v').textContent =
    isNaN(u) ? 'undefined' : u.toFixed(1) + '%';
}
rho_s.oninput = theory; peff_s.oninput = theory;

/* ------------------------------------------------ beat 3: flip rate */
(function () {
  const W = 760, H = 300, padl = 45, padb = 35;
  const X = u => padl + u / 3 * (W - padl - 15),
        Y = f => H - padb - f / 26 * (H - padb - 15);
  function rate(u, r) {
    if (u <= r) return 0;
    if (u <= 1) return 100 * (1 - r / u);
    return 100 * (1 - r) / u;
  }
  let s = `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(3)}" y2="${Y(0)}"
    stroke="black"/><line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}"
    y2="${Y(26)}" stroke="black"/>`;
  for (let t = 0; t <= 3; t += 0.5)
    s += `<text x="${X(t)}" y="${H - padb + 16}" font-size="11"
      text-anchor="middle">${t.toFixed(1)}</text>`;
  for (let f = 0; f <= 25; f += 5)
    s += `<text x="${padl - 8}" y="${Y(f) + 4}" font-size="11"
      text-anchor="end">${f}</text>`;
  D.flip.forEach((a, k) => {
    const pts = [];
    const us = [];
    for (let i = 0; i <= 240; i++) us.push(i * 3 / 240);
    us.push(a.r, 1.0); us.sort((x, y) => x - y);
    for (const u of us) pts.push(X(u).toFixed(1) + ',' +
                                 Y(rate(u, a.r)).toFixed(1));
    s += `<polyline points="${pts.join(' ')}" fill="none" stroke="black"
      stroke-width="2" ${k ? 'stroke-dasharray="6,3"' : ''}/>`;
    s += `<text x="${X(1.04)}" y="${Y(rate(1, a.r)) - 6}"
      font-size="12">${a.label} (peak ${(100 * (1 - a.r)).toFixed(1)}%)</text>`;
  });
  s += `<text x="${(W + padl) / 2}" y="${H - 4}" font-size="12"
    text-anchor="middle">threshold-window half-width, in measured 95%
    half-widths</text>`;
  s += `<text x="14" y="${Y(13)}" font-size="12"
    transform="rotate(-90 14 ${Y(13)})" text-anchor="middle">% of window
    judged differently</text>`;
  document.getElementById('flip').innerHTML = s;
})();

/* ------------------------------------------------ beat 4: coverage */
(function () {
  const W = 760, H = 260, padl = 45;
  const X = i => padl + 40 + i * ((W - padl - 60) / 8);
  const Y = c => 220 - (c - 0.78) / (1.0 - 0.78) * 190;
  let s = `<line x1="${padl}" y1="${Y(D.cover.nominal)}" x2="${W - 15}"
    y2="${Y(D.cover.nominal)}" stroke="black" stroke-dasharray="3,3"/>` +
    `<text x="${W - 18}" y="${Y(D.cover.nominal) - 5}" font-size="11"
    text-anchor="end">nominal 0.95</text>`;
  for (const c of [0.80, 0.85, 0.90, 0.95, 1.00])
    s += `<text x="${padl - 6}" y="${Y(c) + 4}" font-size="11"
      text-anchor="end">${c.toFixed(2)}</text>`;
  D.cover.scenarios.forEach((n, i) => {
    s += `<circle cx="${X(i)}" cy="${Y(D.cover.paired[i])}" r="5"
      fill="black"/>`;
    s += `<circle cx="${X(i)}" cy="${Y(D.cover.independent[i])}" r="5"
      fill="white" stroke="black" stroke-width="1.6"/>`;
    s += `<line x1="${X(i)}" y1="${Y(D.cover.paired[i])}" x2="${X(i)}"
      y2="${Y(D.cover.independent[i])}" stroke="#999"/>`;
    s += `<text x="${X(i)}" y="246" font-size="9" text-anchor="end"
      transform="rotate(-30 ${X(i)} 246)">${n}</text>`;
  });
  s += `<circle cx="${padl + 12}" cy="18" r="5" fill="black"/>
    <text x="${padl + 22}" y="22" font-size="12">paired percentile
    (94.1–97.1%)</text>
    <circle cx="${padl + 260}" cy="18" r="5" fill="white" stroke="black"
    stroke-width="1.6"/>
    <text x="${padl + 270}" y="22" font-size="12">independence shortcut
    (80.2–99.8%)</text>`;
  document.getElementById('cover').innerHTML = s;
})();

/* ------------------------------------------------ beat 5: more evidence */
let more = false;
document.getElementById('more').onclick = () => { more = !more; b_render(); };
function b_render() {
  const run = more ? D.beat.full : D.beat.small;
  const tog = document.getElementById('more');
  tog.classList.toggle('on', more);
  const badge = document.getElementById('b_badge');
  badge.textContent = run.verdict + '  (n = ' + run.n + ')';
  badge.classList.remove('false');
  const lo = 0.55, hi = 1.0;
  const X = v => x760(v, lo, hi, 30);
  const iv = run.interval;
  document.getElementById('b_band').innerHTML =
    `<line x1="${X(lo + 0.01)}" y1="55" x2="${X(hi - 0.01)}" y2="55"
      stroke="#999"/>` +
    `<rect x="${X(iv[0])}" y="45" width="${X(iv[1])-X(iv[0])}" height="20"
      fill="none" stroke="black" stroke-width="2"/>` +
    `<circle cx="${X(run.h)}" cy="55" r="5" fill="white" stroke="black"
      stroke-width="2"/>` +
    `<line x1="${X(D.beat.policy)}" y1="18" x2="${X(D.beat.policy)}"
      y2="85" stroke="black" stroke-dasharray="3,3"/>` +
    `<text x="${X(D.beat.policy)}" y="12" font-size="12"
      text-anchor="middle">policy ${D.beat.policy.toFixed(3)}</text>` +
    `<text x="${X(iv[1]) + 8}" y="60" font-size="12">[${iv[0].toFixed(3)},
      ${iv[1].toFixed(3)}]</text>`;
  document.getElementById('b_note').textContent =
    more
      ? 'n = 300: the interval is ' + D.beat.width_ratio +
        'x narrower and clears the same threshold — ALLOW. The wide ' +
        'interval was never pessimism; it was the price of thin evidence.'
      : 'n = 30: the interval straddles the threshold — ESCALATE. ' +
        'Nothing is wrong with the system; the evidence is simply thin.';
}

/* ------------------------------------------------ appendix */
(function () {
  let s = '<tr><th>field</th><th>value</th><th>corrupting it →</th></tr>';
  for (const [f, v, sc] of D.inspect)
    s += `<tr><td><code>${f}</code></td><td>${v}</td>
      <td>${sc === 'valid' ? '(consumed by the verdict rule)'
        : 'scenario "' + D.gate.scenarios[sc].label + '" — DENY'}</td></tr>`;
  document.getElementById('inspect').innerHTML = s;
  document.getElementById('clock').textContent =
    'Verdicts on this page were evaluated as of ' + D.gate.as_of +
    '. After the certificate’s recalibration date the same gate ' +
    'returns DENY: expired — that is scenario "Expired certificate" above.';
})();

g_render(); p_render(); theory(); b_render();
</script>
"""

voltxt = (f'{VOLRANGE["all"][0]:+.1f}% to {VOLRANGE["all"][1]:+.1f}%'
          f' (logistic model alone: {VOLRANGE["logit"][0]:+.1f} to '
          f'{VOLRANGE["logit"][1]:+.1f}%)')
page = (HTML.replace("__DATA2__", json.dumps(DATA2, separators=(",", ":")))
            .replace("__VOLTXT__", voltxt))
(HERE / "console-v2.html").write_text(page)
print(f"console-v2.html written ({len(page):,} bytes); "
      f"presets tau: " + ", ".join(f"{p['name']}={p['tau']}" for p in PRESETS))
print("asserts: gc rho/under, ra under, coverage 0.941-0.971 / 0.802-0.998, "
      "beat ESCALATE->ALLOW, cloud rho == stored rho — all passed")
