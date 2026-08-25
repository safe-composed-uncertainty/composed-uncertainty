"""The 'more evidence' beat: same system, same policy — DENY-side to ALLOW.

Demo step 5 of the collaboration plan ("increase evidence/sample size,
re-evaluate, and allow the action") had no scripted counterpart. This script
supplies it, honestly:

- the executed two-stage pipeline's frozen episodes are read back
  (real-agentic-2026-07-25/episodes-real-agentic.csv); nothing is refit or
  regenerated beyond the deposited construction;
- the SAME chain estimator as the deposited run (per-stage composed score
  C = geomean(RGA, RGE, RGR); chain h = C_A x C_B; first-order delta interval
  from the paired-bootstrap link covariance; fresh perturbation per replicate)
  is evaluated twice: once on a small seeded subsample of the held-out
  evaluation rows, once on the full held-out set;
- a signed uncertainty certificate is issued for each, and BOTH are run
  through the unmodified certificate_policy_gate against the SAME policy
  (minimum lower confidence bound 0.75, the demo's ALLOW policy);
- with little evidence the interval is wide and straddles the policy bound —
  the gate ESCALATES; with the full evidence the interval narrows, the lower
  bound clears the bound, and the gate ALLOWS. The action itself, the model,
  and the policy never change; only the amount of evidence does.

The subsample SIZE is selected by a deterministic, fully recorded scan over
candidate sizes (single fixed subsample seed): the script takes the smallest
candidate whose interval straddles the fixed 0.75 policy. This is a designed
illustration of interval narrowing, not a discovered phenomenon; every scan
candidate and its verdict is written to the output JSON.

Run:  python3 more_evidence_beat.py          (~2-3 min; logit chain only)
"""

from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.normpath(os.path.join(HERE, ".."))
REAL_DIR = os.path.join(EXPERIMENT_DIR, "real-agentic-2026-07-25")
REDRAW_DIR = os.path.join(EXPERIMENT_DIR, "redraw-2026-07-24")
sys.path.insert(0, os.path.join(REDRAW_DIR, "_stubs"))
sys.path.insert(0, os.path.join(REDRAW_DIR, "safeai-src"))
sys.path.insert(0, EXPERIMENT_DIR)
sys.path.insert(0, HERE)

import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safeai.rga import rga_score
from safeai.rge import rge_score
from safeai.rgr import rgr_score

import build_certificate_examples as bce
from certificate_policy_gate import (
    CertificatePolicy,
    evaluate_certificate,
)

SEED = 20260727           # identical to the deposited real-agentic experiment
B = 2000
NOISE_SD = 0.5
Z = 1.959963984540054
SUBSAMPLE_SEED = SEED + 777
N_CANDIDATES = (12, 16, 20, 24, 30, 40, 60)   # deterministic scan, smallest ESCALATE wins
POLICY_LCB = 0.75         # the demo's ALLOW policy, unchanged across both runs
REDRAW_BASE = {"A": SEED + 100_000, "B": SEED + 200_000}

BEAT_RULE = (
    "fresh perturbation per replicate; seed = stage base + replicate index; "
    "bases: A/logistic 20360727, B/logistic 20460727; evaluation subsample "
    f"drawn once with seed {SUBSAMPLE_SEED}"
)


def load_episodes():
    with open(os.path.join(REAL_DIR, "results-real-agentic.json")) as fh:
        stored = json.load(fh)
    with open(os.path.join(REAL_DIR, "episodes-real-agentic.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    FA, FB = stored["features"]["A"], stored["features"]["B"]
    X = {"A": np.array([[float(r[f"fA_{c}"]) for c in FA] for r in rows]),
         "B": np.array([[float(r[f"fB_{c}"]) for c in FB] for r in rows])}
    y = {"A": np.array([int(r["yA"]) for r in rows]),
         "B": np.array([int(r["yB"]) for r in rows])}
    idx_tr, idx_te = train_test_split(
        np.arange(len(rows)), test_size=stored["n_eval"], random_state=SEED,
        stratify=2 * y["A"] + y["B"])
    split = np.array([r["split"] for r in rows])
    assert set(split[idx_te]) == {"eval"} and set(split[idx_tr]) == {"train"}
    return stored, X, y, idx_tr, idx_te


def geo(v):
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(v, float), 1e-12)))))


def chain_on_rows(models, X, y, rows_idx):
    """The deposited chain construction, restricted to `rows_idx` eval rows.

    Everything downstream of the evaluation sample (masking baselines,
    perturbation scaling, bootstrap) uses only the subsample — that is what
    'having this much evidence' means.
    """
    n = len(rows_idx)
    pre = {}
    rng_global = np.random.default_rng(SEED)
    for s in ("A", "B"):
        mod = models[s]
        Xt = X[s][rows_idx]
        p_full = mod.predict_proba(Xt)[:, 1]
        d = Xt.shape[1]
        cm = Xt.mean(axis=0)
        cs = Xt.std(axis=0, keepdims=True)
        p_masked = np.empty((d, n))
        for j in range(d):
            Xm = Xt.copy()
            Xm[:, j] = cm[j]
            p_masked[j] = mod.predict_proba(Xm)[:, 1]
        pert = rng_global.normal(0.0, NOISE_SD, size=Xt.shape) * cs
        p_pf = mod.predict_proba(Xt + pert)[:, 1]
        pre[s] = (p_full, p_masked, p_pf, cs, Xt, mod)

    point = {}
    for s in ("A", "B"):
        p_full, p_masked, p_pf, _, _, _ = pre[s]
        yb = y[s][rows_idx]
        rga = float(rga_score(yb, p_full))
        rge = float(np.mean([rge_score(p_full, p_masked[j])
                             for j in range(p_masked.shape[0])]))
        rgr = float(rgr_score(p_full, p_pf))
        point[s] = {"RGA": rga, "RGE": rge, "RGR": rgr, "C_geomean": geo([rga, rge, rgr])}

    # per-replicate perturbation predictions, deposited seed rule
    mats = {}
    for s in ("A", "B"):
        _, _, _, cs, Xt, mod = pre[s]
        mat = np.empty((B, n))
        base = REDRAW_BASE[s]
        for b in range(B):
            rb = np.random.default_rng(base + b)
            mat[b] = mod.predict_proba(
                Xt + rb.normal(0.0, NOISE_SD, size=Xt.shape) * cs)[:, 1]
        mats[s] = mat

    shared = np.random.default_rng(SEED + 99)
    CA, CB = [], []
    comp = {"A": [], "B": []}
    for b in range(B):
        idx = shared.integers(0, n, n)
        vals = {}
        for s in ("A", "B"):
            p_full, p_masked, _, _, _, _ = pre[s]
            pf = p_full[idx]
            rga = float(rga_score(y[s][rows_idx][idx], pf))
            rge = float(np.mean([rge_score(pf, p_masked[j][idx])
                                 for j in range(p_masked.shape[0])]))
            rgr = float(rgr_score(pf, mats[s][b][idx]))
            vals[s] = geo([rga, rge, rgr])
            comp[s].append((rga, rge, rgr))
        CA.append(vals["A"])
        CB.append(vals["B"])

    CA, CB = np.asarray(CA), np.asarray(CB)
    va = float(np.cov(CA, ddof=1))
    vb = float(np.cov(CB, ddof=1))
    cab = float(np.cov(CA, CB, ddof=1)[0, 1])
    rho = cab / (va ** 0.5 * vb ** 0.5)
    pa, pb = point["A"]["C_geomean"], point["B"]["C_geomean"]
    h = pa * pb
    var_meas = pb * pb * va + 2 * pa * pb * cab + pa * pa * vb
    var_zero = pb * pb * va + pa * pa * vb
    w_meas, w_zero = 2 * Z * var_meas ** 0.5, 2 * Z * var_zero ** 0.5
    chain = {
        "estimator": "plug-in product of the two full-sample link scores",
        "link_point_values": {"A": pa, "B": pb},
        "cross_link_correlation": rho,
        "link_covariance": [[va, cab], [cab, vb]],
        "h_point": h,
        "bootstrap_link_means": {"A": float(CA.mean()), "B": float(CB.mean())},
        "ci95_measured_covariance": [h - Z * var_meas ** 0.5, h + Z * var_meas ** 0.5],
        "ci95_cross_term_declared_zero": [h - Z * var_zero ** 0.5, h + Z * var_zero ** 0.5],
        "width_measured": w_meas,
        "width_declared_zero": w_zero,
        "understatement_of_width_pct": 100 * (1 - w_zero / w_meas),
    }
    # per-stage Sigma + percentile CI for the link envelopes
    redraw = {}
    for s in ("A", "B"):
        arr = np.asarray(comp[s])           # (B, 3)
        Cs = np.exp(np.mean(np.log(np.maximum(arr, 1e-12)), axis=1))
        redraw[s] = {
            "Sigma": np.cov(arr.T, ddof=1).tolist(),
            "ci_C_paired_percentile": [float(np.percentile(Cs, 2.5)),
                                       float(np.percentile(Cs, 97.5))],
        }
    return point, chain, redraw, n


def certify(tag, n_rows, point, chain, redraw):
    from pathlib import Path
    sample_digest = bce.sha256_file(
        Path(REAL_DIR) / "episodes-real-agentic.csv")
    substrate = {
        "name": "executed authorization-and-evidence pipeline with safeai metric layer",
        "version_or_commit": "39768fcd5264c881f7174268bbffda52b298ae89",
    }
    dataset = (f"1,000 executed randomized episodes; {n_rows} held-out "
               f"evaluation rows ({tag})")
    links = []
    for s, label in (("A", "authorization-stage probabilistic scorer"),
                     ("B", "evidence-verification-stage probabilistic scorer")):
        links.append(bce.link_certificate(
            certificate_id=f"beat-{tag}-stage-{s.lower()}-link-20260801",
            system_id=label, substrate=substrate, dataset=dataset,
            sample_digest=sample_digest, point=point[s], redraw=redraw[s],
            seed=SEED, redraw_rule=BEAT_RULE))
    chain_env = bce.chain_certificate(
        certificate_id=f"beat-{tag}-two-stage-chain-20260801",
        system_id="executed two-stage agentic research pipeline",
        substrate=substrate, dataset=dataset, sample_digest=sample_digest,
        chain=chain, link_a_result=redraw["A"], link_b_result=redraw["B"],
        link_envelopes=links, seed=SEED, redraw_rule=BEAT_RULE)
    return chain_env, links


def gate(envelope):
    certificate = envelope["certificate"]
    policy = CertificatePolicy(
        policy_id=f"research-chain-lcb-{POLICY_LCB:.2f}",
        expected_certificate_type="chain",
        expected_system_id=certificate["subject"]["system_id"],
        expected_substrate=certificate["subject"]["substrate"],
        minimum_lower_bound=POLICY_LCB,
        accepted_interval_methods=("first_order_delta",),
        accepted_uncertainty_scopes=("fixed_artifact_evaluation",),
        trusted_public_keys=(envelope["signature"]["public_key_hex"],),
        allow_publication_example_key=True,
    )
    return evaluate_certificate(
        envelope, policy,
        mandate_certificate_payload_sha256=envelope["signature"]["payload_sha256"],
        as_of_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def main():
    stored, X, y, idx_tr, idx_te = load_episodes()
    models = {}
    for s in ("A", "B"):
        mod = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=SEED))
        mod.fit(X[s][idx_tr], y[s][idx_tr])
        models[s] = mod

    out = {"policy_minimum_lower_bound": POLICY_LCB, "B": B, "seed": SEED,
           "subsample_seed": SUBSAMPLE_SEED, "design_note": (
               "small-N selected by a deterministic recorded scan over "
               "N_CANDIDATES (single fixed subsample seed): smallest candidate "
               "whose interval straddles the fixed policy bound; a designed "
               "illustration of interval narrowing under the deposited "
               "estimator, not a discovered phenomenon"),
           "scan": [], "runs": {}}
    print(f"policy: minimum lower confidence bound {POLICY_LCB} (fixed for both runs)\n")

    small_rows = None
    for cand in N_CANDIDATES:
        rows_c = np.sort(np.random.default_rng(SUBSAMPLE_SEED)
                         .choice(idx_te, size=cand, replace=False))
        point_c, chain_c, redraw_c, _ = chain_on_rows(models, X, y, rows_c)
        lcb_c, ucb_c = chain_c["ci95_measured_covariance"]
        straddles = bool(np.isfinite(lcb_c) and np.isfinite(ucb_c)
                         and lcb_c < POLICY_LCB <= ucb_c)
        def fin(x):
            return float(x) if np.isfinite(x) else None
        out["scan"].append({"n": int(cand), "h": fin(chain_c["h_point"]),
                            "interval95": [fin(lcb_c), fin(ucb_c)],
                            "straddles_policy": bool(straddles),
                            "note": (None if np.isfinite(lcb_c) else
                                     "interval undefined: bootstrap resamples at this n "
                                     "can draw a single-class sample where RGA is undefined")})
        print(f"  scan n={cand:>3}: h={chain_c['h_point']:.4f} "
              f"CI95=[{lcb_c:.4f}, {ucb_c:.4f}] "
              f"{'<- selected' if straddles and small_rows is None else ''}")
        if straddles and small_rows is None:
            small_rows = rows_c
    assert small_rows is not None, "no scan candidate straddles the policy bound"
    print()
    for tag, rows_idx in (("small", small_rows), ("full", idx_te)):
        point, chain, redraw, n_rows = chain_on_rows(models, X, y, rows_idx)
        env, _ = certify(tag, n_rows, point, chain, redraw)
        decision = gate(env)
        lcb, ucb = chain["ci95_measured_covariance"]
        out["runs"][tag] = {
            "n_eval_rows": int(n_rows),
            "h_point": chain["h_point"],
            "interval95": chain["ci95_measured_covariance"],
            "width": chain["width_measured"],
            "verdict": decision.verdict,
            "reason": decision.reason,
            "certificate_payload_sha256": env["signature"]["payload_sha256"],
        }
        print(f"[{tag:>5}] n={n_rows:>3}  h={chain['h_point']:.4f}  "
              f"CI95=[{lcb:.4f}, {ucb:.4f}]  width={chain['width_measured']:.4f}"
              f"  ->  {decision.verdict}")

    small_v = out["runs"]["small"]["verdict"]
    full_v = out["runs"]["full"]["verdict"]
    assert small_v in ("ESCALATE", "DENY") and full_v == "ALLOW", (
        f"beat did not realize the progression: small={small_v}, full={full_v}")
    ratio = out["runs"]["small"]["width"] / out["runs"]["full"]["width"]
    out["width_ratio_small_over_full"] = ratio
    print(f"\nwidth ratio small/full = {ratio:.2f} "
          f"(≈ sqrt({out['runs']['full']['n_eval_rows']}/{out['runs']['small']['n_eval_rows']})"
          f" = {np.sqrt(out['runs']['full']['n_eval_rows']/out['runs']['small']['n_eval_rows']):.2f})")
    with open(os.path.join(HERE, "MORE-EVIDENCE-BEAT.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote MORE-EVIDENCE-BEAT.json")
    print("\nSame system, same action, same policy. Only the amount of "
          "evidence changed:\n  "
          f"{small_v} at n={out['runs']['small']['n_eval_rows']}  ->  "
          f"ALLOW at n={out['runs']['full']['n_eval_rows']}")


if __name__ == "__main__":
    main()
