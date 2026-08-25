#!/usr/bin/env python3
"""Two-stage worked example from the live agentic pipeline - execution harness.

INTERNAL WORKING ARTIFACT. This file names repositories, hosts, and project
codenames for reproducibility. None of those names may appear in the
manuscript or in any article-facing text. In the article the substrate is
described only as "a live agentic research pipeline operated by the first
author". Do not commit this file or this directory without an explicit
go-ahead.

Implements Design G1 of DESIGN.md (2026-07-25): N seeded episodes are run
through the real deployed chain vendored in ./pipeline-src (six-check
authorisation gate `aaam.activate`, four-check offline relying-party
verifier `rp_verify.verify`, real ECDSA P-256 throughout). Stage labels are
the outputs of executed verification runs - nothing simulated, nothing
hand-assigned. Episode parameter draws are synthetic randomised replays
(disclosed). Two small probabilistic scorers (logit primary, random forest
sensitivity) are trained on the executed episodes because the deployed
gates are deterministic and a 0/1 gate score gives degenerate RGA
(disclosed). RGA/RGE/RGR from the pinned safeai vendored by the redraw
study; paired bootstrap, joint 6-metric covariance, two-link chain
h = C_A * C_B exactly as in redraw-2026-07-24.

Run:
  python3 harness.py --tag pilot        # N=100, B=200, separate seed, pilot/ dir
  python3 harness.py --tag production   # N=1000, B=2000, frozen CONFIG
  python3 harness.py --tag production --out-dir <dir>   # recomputation
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
    category=UserWarning,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "pipeline-src")
REDRAW_DIR = os.path.normpath(os.path.join(HERE, "..", "redraw-2026-07-24"))
SAFEAI = os.path.join(REDRAW_DIR, "safeai-src")
STUBS = os.path.join(REDRAW_DIR, "_stubs")
sys.path.insert(0, STUBS)
sys.path.insert(0, SAFEAI)
sys.path.insert(0, PIPE)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aaam
import attester
import mandate
import rp_verify
from aaam import AAAMReject
from crypto import gen_ec, jwk_thumbprint, pub_jwk


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
from jcs import H, Hs
from sam_sim import SAM
from trusted_list import TrustedList

from safeai.rga import rga_score
from safeai.rge import rge_score
from safeai.rgr import rgr_score

# ----------------------------------------------------------------------------
# CONFIG (frozen before the production run; see CONFIG-real-agentic.md)
# ----------------------------------------------------------------------------
SEED = 20260727
ALPHA = 0.05
NOISE_SD = 0.5            # RGR: N(0, NOISE_SD) * col_std on every feature column
AUD = "rp://demo"
PRINCIPAL = "did:legal:acme-ou"
AGENT = "agent://acme/invoice-bot#1"

TOOLS = ["pay_invoice", "refund_invoice", "wire_transfer", "update_ledger"]
KNOWN_PAYEES = ["acme", "globex", "initech", "umbrella"]
CAPS = [500.0, 2000.0, 5000.0, 20000.0]
DTBS_KEYS = ["bind_ver", "action_hash", "outcome_sha256", "attestation_digest",
             "mandate_digest", "holder_cnf_thumbprint", "nonce"]

# channel id, probability, expected (yA, yB), source vector ids
# sources: demo-attacks = accountable-agentic-action/demo/test_attacks.py case #
#          demo-m2      = accountable-agentic-action/demo/test_m2.py
#          mm-ablation  = machine-mandate/run_ablation.py CASES row (gate lever)
#          eatf-vec     = eatf-verifier/test-vectors/invalid/<name>
CHANNELS = [
    ("clean",                  0.500, (0, 0), ["baseline (happy path, demo aaa_demo.py M1)"]),
    ("over_scope_amount",      0.070, (1, 0), ["mm-ablation:over-limit-payment(L4)", "demo-attacks:2-out-of-scope"]),
    ("method_not_allowed",     0.050, (1, 0), ["mm-ablation:prompt-injected-wire-transfer(L4)"]),
    ("expired_mandate",        0.040, (1, 1), ["demo-attacks:4-expired-mandate"]),
    ("untrusted_issuer",       0.030, (1, 1), ["eatf-vec:untrusted-issuer", "mm-ablation:issuer-not-on-trusted-list(L3)"]),
    ("wrong_holder_key",       0.025, (1, 1), ["demo-attacks:5-wrong-holder-key"]),
    ("forged_credential_sig",  0.025, (1, 1), ["mm-ablation:forged-altered-credential(L1)", "eatf-vec:bad-signature-classical"]),
    ("payee_swap_action_hash", 0.030, (1, 1), ["mm-ablation:payee-swapped-after-approval(L4)", "demo-attacks:1-action-hash-mismatch", "demo-m2:confused-deputy"]),
    ("stale_attestation",      0.025, (1, 1), ["mm-ablation:replayed-stale-attestation(L2)"]),
    ("nonce_replay_cross",     0.010, (1, 1), ["demo-attacks:3-nonce-replay (cross-session variant)"]),
    ("nonce_replay_same",      0.010, (1, 0), ["demo-attacks:3-nonce-replay (same-session variant, gate check 5)"]),
    ("dtbs_field_tamper",      0.050, (0, 1), ["demo-m2:binding-tamper-fuzz(7 fields)", "eatf-vec:tampered-metadata", "eatf-vec:tampered-overt-receipt"]),
    ("forged_seal",            0.030, (0, 1), ["demo-attacks:6-forged-seal", "eatf-vec:bad-signature-classical"]),
    ("tampered_ear",           0.030, (0, 1), ["demo-attacks:7-tampered-EAR", "eatf-vec:tampered-canonical-bin"]),
    ("outcome_mutation",       0.030, (0, 1), ["eatf-vec:tampered-canonical-bin (content mutated after sealing)"]),
    ("untrusted_sam",          0.030, (0, 1), ["demo-attacks:8-untrusted-SAM", "eatf-vec:untrusted-issuer (sealer analogue)"]),
    ("missing_component",      0.015, (0, 1), ["eatf-vec:missing-canonical-bin"]),
]
assert abs(sum(p for _, p, _, _ in CHANNELS) - 1.0) < 1e-12

FEATURES_A = ["tool_code", "log10_amount", "payee_unknown", "n_methods_allowed",
              "log10_cap", "headroom_frac", "method_in_scope", "expiry_margin_s",
              "issuer_on_list"]
FEATURES_B = ["tool_code", "log10_amount", "payee_unknown", "headroom_frac",
              "expiry_margin_s", "issuer_on_list", "n_package_fields",
              "nonce_match", "ear_nonce_match"]

METRICS = ["RGA", "RGE", "RGR"]
STAGES = ["A", "B"]
MODEL_TAGS = ["logit", "rf"]
REDRAW_BASE = {("A", "logit"): SEED + 100_000, ("B", "logit"): SEED + 200_000,
               ("A", "rf"): SEED + 300_000, ("B", "rf"): SEED + 400_000}


# ----------------------------------------------------------------------------
# Episode generation: draw parameters, execute the REAL chain, record verdicts
# ----------------------------------------------------------------------------
def flip_char(s, pos):
    c = "A" if s[pos] != "A" else "B"
    return s[:pos] + c + s[pos + 1:]


def unconditional_seal(action, ear, sd_jwt, holder_jwk, nonce, sam):
    """Harness step, disclosed: seal the episode's evidence even when the gate
    denied, modelling an adversarial agent that proceeds despite the gate.
    Same DTBS/R composition as aaam.activate lines 52-62; the seal itself is
    a real single-use ECDSA seal produced by the same SAM code."""
    dtbs = {
        "bind_ver": "aaa/0.1",
        "action_hash": H(action),
        "outcome_sha256": ear["outcome_sha256"],
        "attestation_digest": H(ear),
        "mandate_digest": Hs(sd_jwt),
        "holder_cnf_thumbprint": jwk_thumbprint(holder_jwk),
        "nonce": nonce,
    }
    seal = sam.activate(dtbs)
    return {**dtbs, "seal": seal, "sam_jwk": sam.jwk}


def run_episode(i, episode_seed):
    rng = np.random.default_rng([episode_seed, i])
    probs = np.array([p for _, p, _, _ in CHANNELS])
    ch_idx = int(rng.choice(len(CHANNELS), p=probs))
    channel, _, expected_y, sources = CHANNELS[ch_idx]

    # ---- parameter draws (synthetic, disclosed) ----
    cap = float(CAPS[int(rng.integers(len(CAPS)))])
    n_methods = int(rng.integers(1, 4))
    tool_idx = int(rng.integers(len(TOOLS)))
    tool = TOOLS[tool_idx]
    others = [t for t in TOOLS if t != tool]
    rng.shuffle(others)
    scope_methods = sorted([tool] + others[:n_methods - 1])
    if channel == "method_not_allowed":
        scope_methods = sorted(others[:n_methods])   # presented tool excluded
    if channel == "over_scope_amount":
        amount = round(cap * 10 ** rng.uniform(0.005, 0.5), 2)   # 1.01x..3.16x cap
    else:
        amount = round(cap * 10 ** rng.uniform(-2.0, -0.02), 2)  # 1%..95% of cap
    payee_unknown_flag = int(rng.random() < 0.2)
    payee = (KNOWN_PAYEES[int(rng.integers(len(KNOWN_PAYEES)))]
             if not payee_unknown_flag else f"px-{int(rng.integers(1000)):03d}")
    if channel == "expired_mandate":
        ttl = float(rng.uniform(-600.0, -5.0))
    else:
        ttl = float(rng.uniform(30.0, 600.0))
    scope = {"methods": scope_methods, "max_amount_eur": cap}

    invoice = f"INV-{i:05d}"
    action_mandated = {"method": tool, "args": {"invoice": invoice,
                                                "amount_eur": amount, "payee": payee}}
    action_presented = action_mandated
    if channel == "payee_swap_action_hash":
        swapped_unknown = int(rng.random() < 0.8)
        new_payee = (f"px-{int(rng.integers(1000)):03d}" if swapped_unknown
                     else [p for p in KNOWN_PAYEES if p != payee][int(rng.integers(3))])
        action_presented = {"method": tool, "args": {"invoice": invoice,
                                                     "amount_eur": amount,
                                                     "payee": new_payee}}
        payee_unknown_flag = swapped_unknown

    # ---- real cryptographic material, per-episode identities ----
    issuer, holder, ak = gen_ec(), gen_ec(), gen_ec()
    sam = SAM()
    tl = TrustedList()
    issuer_on_list = 0 if channel == "untrusted_issuer" else 1
    tl.add_issuer(pub_jwk(gen_ec()) if not issuer_on_list else pub_jwk(issuer))
    tl.add_qtsp(sam.jwk)
    seal_sam = sam
    if channel == "untrusted_sam":
        seal_sam = SAM()          # rogue sealer, not on the trusted list

    nonce = f"n-{i:05d}-{int(rng.integers(1 << 30)):08x}"
    nonce_gate, nonce_rp, ear_nonce = nonce, nonce, nonce
    store = set()
    if channel == "nonce_replay_cross":
        # agent replays materials bound to an old nonce against a fresh challenge
        fresh = f"n-{i:05d}-fresh-{int(rng.integers(1 << 30)):08x}"
        nonce_gate, nonce_rp = fresh, fresh
    if channel == "nonce_replay_same":
        store.add(nonce)          # same-session second activation: SAD-style replay
    if channel == "stale_attestation":
        ear_nonce = f"n-{i:05d}-stale-{int(rng.integers(1 << 30)):08x}"

    sd = mandate.issue(issuer, PRINCIPAL, AGENT, scope, H(action_presented),
                       pub_jwk(holder), ttl=ttl, jti=f"m-{i:05d}")
    if channel == "payee_swap_action_hash":   # mandate binds the ORIGINAL action
        sd = mandate.issue(issuer, PRINCIPAL, AGENT, scope, H(action_mandated),
                           pub_jwk(holder), ttl=ttl, jti=f"m-{i:05d}")
    sd_presented = sd
    if channel == "forged_credential_sig":
        parts = sd.split(".")
        pos = int(rng.integers(len(parts[2]) - 4))
        parts[2] = flip_char(parts[2], pos)
        sd_presented = ".".join(parts)

    kb_signer = gen_ec() if channel == "wrong_holder_key" else holder
    kb = mandate.present(sd_presented, kb_signer, nonce, AUD)
    outcome = {"ok": True, "invoice": invoice, "amount_eur": amount, "payee":
               action_presented["args"]["payee"]}
    ear = attester.attest(outcome, ear_nonce, ak)

    # ---- STAGE A: the deployed six-check gate, run unmodified ----
    tamper_detail = ""
    try:
        bas = aaam.activate(action_presented, sd_presented, kb, ear,
                            pub_jwk(issuer), seal_sam, tl, store, nonce_gate, AUD)
        yA, deny = 0, ""
    except AAAMReject as e:
        yA, deny, bas = 1, str(e), None
    if bas is None:
        bas = unconditional_seal(action_presented, ear, sd_presented,
                                 pub_jwk(holder), nonce, seal_sam)

    # ---- post-gate tamper channels (applied AFTER stage A, before stage B) ----
    bas2, ear2, outcome2 = dict(bas), dict(ear), dict(outcome)
    if channel == "dtbs_field_tamper":
        fld = DTBS_KEYS[int(rng.integers(len(DTBS_KEYS)))]
        bas2[fld] = str(bas2[fld]) + "X"
        tamper_detail = f"dtbs:{fld}"
    elif channel == "forged_seal":
        seg = bas2["seal"].split(".")
        pos = int(rng.integers(len(seg[2]) - 1))
        seg[2] = flip_char(seg[2], pos)
        bas2["seal"] = ".".join(seg)
        tamper_detail = "seal-signature-byte"
    elif channel == "tampered_ear":
        fld = ["outcome_sha256", "quote_sig"][int(rng.integers(2))]
        ear2[fld] = str(ear2[fld])[:-2] + "xx"
        tamper_detail = f"ear:{fld}"
    elif channel == "outcome_mutation":
        outcome2 = dict(outcome)
        outcome2["amount_eur"] = round(amount * float(rng.uniform(1.5, 20.0)), 2)
        tamper_detail = "outcome:amount_eur"
    elif channel == "missing_component":
        fld = (DTBS_KEYS + ["seal"])[int(rng.integers(8))]
        del bas2[fld]
        tamper_detail = f"missing:{fld}"

    # ---- STAGE B: the deployed four-check offline verifier, run unmodified;
    #      an exception on malformed input is recorded as REJECT (disclosed) ----
    try:
        res = rp_verify.verify(bas2, sd_presented, kb, ear2, outcome2,
                               action_presented, pub_jwk(issuer), tl, nonce_rp, AUD)
        yB = 0 if res["ok"] else 1
        checks = {k: bool(res[k]) for k in
                  ("mandate", "execution", "sole_control", "legal_effect")}
        if yB == 0:
            diag = ""
        elif not checks["mandate"]:
            diag = "mandate-or-binding-invalid"
        elif not checks["execution"]:
            diag = "content-mutated"
        elif not checks["sole_control"]:
            diag = "seal-invalid"
        else:
            diag = "sealer-not-listed"
    except Exception as e:
        yB, checks = 1, {"mandate": None, "execution": None,
                         "sole_control": None, "legal_effect": None}
        diag = f"malformed-package ({e.__class__.__name__})"

    # ---- tabular features (pre-decision observables only; no crypto bits) ----
    amt = action_presented["args"]["amount_eur"]
    fA = [float(tool_idx), float(np.log10(amt)), float(payee_unknown_flag),
          float(len(scope_methods)), float(np.log10(cap)),
          float((cap - amt) / cap), float(tool in scope_methods), float(ttl),
          float(issuer_on_list)]
    fB = [float(tool_idx), float(np.log10(amt)), float(payee_unknown_flag),
          float((cap - amt) / cap), float(ttl), float(issuer_on_list),
          float(len(bas2)), float(bas2.get("nonce") == nonce_rp),
          float(ear2.get("nonce") == nonce_rp)]

    return {"i": i, "channel": channel, "sources": "; ".join(sources),
            "expected_yA": expected_y[0], "expected_yB": expected_y[1],
            "tool": tool, "amount_eur": amt,
            "payee": action_presented["args"]["payee"],
            "payee_unknown": payee_unknown_flag, "cap": cap,
            "n_methods": len(scope_methods), "ttl_s": round(ttl, 2),
            "issuer_on_list": issuer_on_list, "tamper_detail": tamper_detail,
            "yA": yA, "A_deny_reason": deny, "yB": yB,
            "B_mandate": checks["mandate"], "B_execution": checks["execution"],
            "B_sole_control": checks["sole_control"],
            "B_legal_effect": checks["legal_effect"], "B_diag": diag,
            "fA": fA, "fB": fB}


# ----------------------------------------------------------------------------
# Metric layer (mirrors redraw-2026-07-24/redraw_experiment.py)
# ----------------------------------------------------------------------------
def geo_mean(v):
    return float(np.exp(np.mean(np.log(np.clip(v, 1e-12, None)))))


def corr_dict(M):
    C = np.corrcoef(M, rowvar=False)
    return {"matrix": C.tolist(),
            "corr_RGA_RGE": float(C[0, 1]),
            "corr_RGA_RGR": float(C[0, 2]),
            "corr_RGE_RGR": float(C[1, 2])}


def chain_block(CA, CB, point_A, point_B):
    boot_A, boot_B = float(CA.mean()), float(CB.mean())
    S = np.cov(np.vstack([CA, CB]))
    rho = float(np.corrcoef(CA, CB)[0, 1])
    h_point = float(point_A * point_B)
    grad = np.array([point_B, point_A])
    var_measured = float(grad @ S @ grad)
    var_declared0 = float(grad[0] ** 2 * S[0, 0] + grad[1] ** 2 * S[1, 1])
    z = 1.959963984540054
    ci_meas = [h_point - z * np.sqrt(var_measured), h_point + z * np.sqrt(var_measured)]
    ci_decl = [h_point - z * np.sqrt(var_declared0), h_point + z * np.sqrt(var_declared0)]
    w_meas, w_decl = ci_meas[1] - ci_meas[0], ci_decl[1] - ci_decl[0]
    return {"estimator": "plug-in product of the two full-sample link scores",
            "link_point_values": {"A": float(point_A), "B": float(point_B)},
            "cross_link_correlation": rho, "h_point": h_point,
            "link_covariance": S.tolist(),
            "bootstrap_link_means": {"A": boot_A, "B": boot_B},
            "bootstrap_mean_product_sensitivity": boot_A * boot_B,
            "ci95_measured_covariance": [float(v) for v in ci_meas],
            "ci95_cross_term_declared_zero": [float(v) for v in ci_decl],
            "width_measured": float(w_meas), "width_declared_zero": float(w_decl),
            "understatement_of_width_pct": float(100 * (1 - w_decl / w_meas))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["pilot", "production"], required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.tag == "pilot":
        N, B, episode_seed = 100, 200, SEED + 555
        out_dir = args.out_dir or os.path.join(HERE, "pilot")
        stem = "pilot"
    else:
        N, B, episode_seed = 1000, 2000, SEED
        out_dir = args.out_dir or HERE
        stem = "real-agentic"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[config] tag={args.tag} N={N} B={B} episode_seed={episode_seed} "
          f"seed={SEED} noise_sd={NOISE_SD}")

    # ---- generate + execute the real chain ----
    eps = [run_episode(i, episode_seed) for i in range(N)]
    t_gen = time.time() - t0
    print(f"[episodes] {N} chained executions in {t_gen:.1f}s "
          f"({1000 * t_gen / N:.1f} ms/episode)")

    # ground-truth-by-construction validation: executed verdict must match the
    # channel's expected cell for every episode (fabrication tripwire)
    mism = [(e["i"], e["channel"], e["yA"], e["yB"]) for e in eps
            if (e["yA"], e["yB"]) != (e["expected_yA"], e["expected_yB"])]
    if mism:
        for m in mism[:20]:
            print(f"[MISMATCH] episode {m[0]} channel {m[1]} executed "
                  f"(yA,yB)=({m[2]},{m[3]})")
        raise SystemExit(f"ABORT: {len(mism)} episodes contradict their "
                         f"channel's expected verdict cell")
    print("[labels] all executed verdicts match their channel's expected cell")

    X_A = np.array([e["fA"] for e in eps])
    X_B = np.array([e["fB"] for e in eps])
    y_A = np.array([e["yA"] for e in eps])
    y_B = np.array([e["yB"] for e in eps])
    y_cell = 2 * y_A + y_B

    n_eval = 300 if args.tag == "production" else 30
    idx_all = np.arange(N)
    idx_tr, idx_te = train_test_split(idx_all, test_size=n_eval,
                                      random_state=SEED, stratify=y_cell)
    n = len(idx_te)
    cells = {f"A{a}B{b}": int(np.sum((y_A[idx_te] == a) & (y_B[idx_te] == b)))
             for a in (0, 1) for b in (0, 1)}
    print(f"[split] train={len(idx_tr)} eval={n} | eval 2x2 cells {cells} | "
          f"prevalence eval: DENY={y_A[idx_te].mean():.3f} "
          f"REJECT={y_B[idx_te].mean():.3f}")

    X = {"A": X_A, "B": X_B}
    y = {"A": y_A, "B": y_B}
    Xtr = {s: X[s][idx_tr] for s in STAGES}
    Xte = {s: X[s][idx_te] for s in STAGES}
    ytr = {s: y[s][idx_tr] for s in STAGES}
    yte = {s: y[s][idx_te] for s in STAGES}

    # ---- trained-for-the-purpose scorers (disclosed) ----
    models, aucs = {}, {}
    for s in STAGES:
        for m in MODEL_TAGS:
            if m == "logit":
                mod = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, random_state=SEED))
            else:
                mod = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                             n_jobs=-1)
            mod.fit(Xtr[s], ytr[s])
            models[(s, m)] = mod
            p = mod.predict_proba(Xte[s])[:, 1]
            aucs[f"{s}_{m}"] = float(roc_auc_score(yte[s], p))
    print(f"[scorers] held-out AUC: " +
          " ".join(f"{k}={v:.3f}" for k, v in aucs.items()))
    for k, v in aucs.items():
        if v > 0.99:
            print(f"[WARN] separability trap: AUC {k} = {v:.4f} > 0.99 "
                  f"(pre-registered failure condition, DESIGN.md Section 5)")

    # ---- precompute prediction vectors (fixed-draw order documented) ----
    pre = {}
    rng_global = np.random.default_rng(SEED)
    for s in STAGES:                       # order: (A,logit),(A,rf),(B,logit),(B,rf)
        for m in MODEL_TAGS:
            mod = models[(s, m)]
            Xt = Xte[s]
            p_full = mod.predict_proba(Xt)[:, 1]
            d = Xt.shape[1]
            col_mean = Xt.mean(axis=0)
            col_std = Xt.std(axis=0, keepdims=True)
            p_masked = np.empty((d, n))
            for j in range(d):
                Xm = Xt.copy()
                Xm[:, j] = col_mean[j]
                p_masked[j] = mod.predict_proba(Xm)[:, 1]
            pert = rng_global.normal(0.0, NOISE_SD, size=Xt.shape) * col_std
            p_pert_fixed = mod.predict_proba(Xt + pert)[:, 1]
            pre[(s, m)] = (p_full, p_masked, p_pert_fixed, col_std)
    print(f"[precompute] done at {time.time() - t0:.1f}s")

    # ---- per-replicate redraw matrices (REDRAW scheme, seeded per replicate) --
    t1 = time.time()
    p_pert_redraw = {}
    for (s, m), (p_full, p_masked, _, col_std) in pre.items():
        mod = models[(s, m)]
        Xt = Xte[s]
        mat = np.empty((B, n))
        base = REDRAW_BASE[(s, m)]
        for b in range(B):
            rb = np.random.default_rng(base + b)
            pert_b = rb.normal(0.0, NOISE_SD, size=Xt.shape) * col_std
            mat[b] = mod.predict_proba(Xt + pert_b)[:, 1]
        p_pert_redraw[(s, m)] = mat
    print(f"[redraw] 4 arms x {B} fresh perturbation predictions in "
          f"{time.time() - t1:.1f}s")

    def metric_vector_fixed(s, m, idx):
        p_full, p_masked, p_pert_fixed, _ = pre[(s, m)]
        yb, pf = yte[s][idx], p_full[idx]
        rga = rga_score(yb, pf)
        rge = float(np.mean([rge_score(pf, p_masked[j][idx])
                             for j in range(p_masked.shape[0])]))
        rgr = rgr_score(pf, p_pert_fixed[idx])
        return np.array([rga, rge, rgr])

    results = {
        "design": "G1 single chained generator on the end-to-end demonstrator "
                  "(DESIGN.md Section 3)",
        "tag": args.tag, "N": N, "n_train": len(idx_tr), "n_eval": n,
        "B": B, "alpha": ALPHA, "noise_sd": NOISE_SD, "seed": SEED,
        "episode_seed": episode_seed,
        "episode_seed_rule": "per-episode rng = np.random.default_rng([episode_seed, i])",
        "redraw_seed_rule": f"per-replicate seed = REDRAW_BASE[(stage,model)] + b; "
                            f"REDRAW_BASE = {{{', '.join(f'{k}: {v}' for k, v in REDRAW_BASE.items())}}}",
        "index_streams": {
            "paired_bootstrap": "default_rng(SEED+1), re-instantiated per (stage, model) arm",
            "chain_shared": "default_rng(SEED+99), re-instantiated per model arm",
        },
        "safeai_commit": subprocess.check_output(
            ["git", "-C", SAFEAI, "rev-parse", "HEAD"]).decode().strip(),
        "pipeline_module_sha256": {
            f: sha256_file(os.path.join(PIPE, f))
            for f in sorted(os.listdir(PIPE)) if f.endswith(".py")},
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__,
                     "sklearn": __import__("sklearn").__version__},
        "channels": [{"id": c, "prob": p, "expected_cell": list(e), "sources": src}
                     for c, p, e, src in CHANNELS],
        "features": {"A": FEATURES_A, "B": FEATURES_B},
        "channel_counts": {c: int(sum(1 for e in eps if e["channel"] == c))
                           for c, _, _, _ in CHANNELS},
        "eval_cells_2x2": cells,
        "prevalence": {"train_DENY": float(ytr["A"].mean()),
                       "train_REJECT": float(ytr["B"].mean()),
                       "eval_DENY": float(yte["A"].mean()),
                       "eval_REJECT": float(yte["B"].mean())},
        "scorer_auc_eval": aucs,
        "stages": {}, "joint": {}, "chain": {},
    }

    # ---- per-(stage, model) paired bootstrap, fixed-draw and redraw schemes --
    # `store` is persistence only (added 2026-07-27): it records replicate
    # matrices that are already computed, so that Monte Carlo error bars and the
    # DeLong cross-check can be recomputed without a fresh run. No random draw,
    # no arithmetic and no ordering is changed by these stores.
    store = {}
    qlo, qhi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    for s in STAGES:
        results["stages"][s] = {}
        for m in MODEL_TAGS:
            t2 = time.time()
            p_full = pre[(s, m)][0]
            boot_rng = np.random.default_rng(SEED + 1)
            paired_F = np.empty((B, 3))
            paired_R = np.empty((B, 3))
            for b in range(B):
                idx = boot_rng.integers(0, n, n)
                vec = metric_vector_fixed(s, m, idx)
                paired_F[b] = vec
                rgr_b = rgr_score(p_full[idx], p_pert_redraw[(s, m)][b][idx])
                paired_R[b] = [vec[0], vec[1], rgr_b]
            point = metric_vector_fixed(s, m, np.arange(n))
            CF = np.array([geo_mean(v) for v in paired_F])
            CR = np.array([geo_mean(v) for v in paired_R])
            store[f"paired_fixed_{s}_{m}"] = paired_F
            store[f"paired_redraw_{s}_{m}"] = paired_R
            store[f"point_{s}_{m}"] = point
            results["stages"][s][m] = {
                "point_fixed_draw": dict(zip(METRICS + ["C_geomean"],
                                             [float(v) for v in point] + [geo_mean(point)])),
                "fixed_draw": {
                    "Corr": corr_dict(paired_F),
                    "Sigma": np.cov(paired_F, rowvar=False).tolist(),
                    "sd": {mm: float(paired_F[:, k].std(ddof=1))
                           for k, mm in enumerate(METRICS)},
                    "ci_C_paired_percentile": [float(np.percentile(CF, qlo)),
                                               float(np.percentile(CF, qhi))],
                    "sd_C_paired": float(CF.std(ddof=1)),
                },
                "redraw": {
                    "Corr": corr_dict(paired_R),
                    "Sigma": np.cov(paired_R, rowvar=False).tolist(),
                    "sd": {mm: float(paired_R[:, k].std(ddof=1))
                           for k, mm in enumerate(METRICS)},
                    "ci_C_paired_percentile": [float(np.percentile(CR, qlo)),
                                               float(np.percentile(CR, qhi))],
                    "sd_C_paired": float(CR.std(ddof=1)),
                    "mean_RGR": float(paired_R[:, 2].mean()),
                },
            }
            cR = results["stages"][s][m]["redraw"]["Corr"]
            print(f"[{s}/{m}] point RGA={point[0]:.4f} RGE={point[1]:.4f} "
                  f"RGR={point[2]:.4f} C={geo_mean(point):.4f} | redraw corr "
                  f"RGA-RGE={cR['corr_RGA_RGE']:+.3f} RGA-RGR={cR['corr_RGA_RGR']:+.3f} "
                  f"RGE-RGR={cR['corr_RGE_RGR']:+.3f} | {time.time() - t2:.1f}s")

    # ---- two-link chain + joint 6-metric covariance (shared index stream) ----
    t3 = time.time()
    JOINT_NAMES = [f"{s}_{mm}" for s in STAGES for mm in METRICS]
    for m in MODEL_TAGS:
        shared_rng = np.random.default_rng(SEED + 99)
        CA_fix = np.empty(B); CB_fix = np.empty(B)
        CA_red = np.empty(B); CB_red = np.empty(B)
        joint_fix = np.empty((B, 6)); joint_red = np.empty((B, 6))
        chain_idx = np.empty((B, n), dtype=np.int32)   # persistence only
        for b in range(B):
            idx = shared_rng.integers(0, n, n)
            chain_idx[b] = idx
            vecA = metric_vector_fixed("A", m, idx)
            vecB = metric_vector_fixed("B", m, idx)
            rgrA = rgr_score(pre[("A", m)][0][idx], p_pert_redraw[("A", m)][b][idx])
            rgrB = rgr_score(pre[("B", m)][0][idx], p_pert_redraw[("B", m)][b][idx])
            CA_fix[b] = geo_mean(vecA); CB_fix[b] = geo_mean(vecB)
            CA_red[b] = geo_mean([vecA[0], vecA[1], rgrA])
            CB_red[b] = geo_mean([vecB[0], vecB[1], rgrB])
            joint_fix[b] = np.concatenate([vecA, vecB])
            joint_red[b] = [vecA[0], vecA[1], rgrA, vecB[0], vecB[1], rgrB]
        store[f"chain_joint_fixed_{m}"] = joint_fix
        store[f"chain_joint_redraw_{m}"] = joint_red
        store[f"chain_C_fixed_A_{m}"] = CA_fix
        store[f"chain_C_fixed_B_{m}"] = CB_fix
        store[f"chain_C_redraw_A_{m}"] = CA_red
        store[f"chain_C_redraw_B_{m}"] = CB_red
        store[f"chain_row_index_{m}"] = chain_idx
        results["joint"][m] = {
            "metric_order": JOINT_NAMES,
            "fixed_draw": {"Sigma6": np.cov(joint_fix, rowvar=False).tolist(),
                           "Corr6": np.corrcoef(joint_fix, rowvar=False).tolist()},
            "redraw": {"Sigma6": np.cov(joint_red, rowvar=False).tolist(),
                       "Corr6": np.corrcoef(joint_red, rowvar=False).tolist()},
        }
        point_A = results["stages"]["A"][m]["point_fixed_draw"]["C_geomean"]
        point_B = results["stages"]["B"][m]["point_fixed_draw"]["C_geomean"]
        results["chain"][m] = {
            "links": {"A": f"stage-A scorer ({m})", "B": f"stage-B scorer ({m})"},
            "chain_functional": "h = C_A * C_B (both-must-hold)",
            "fixed_draw": chain_block(CA_fix, CB_fix, point_A, point_B),
            "redraw": chain_block(CA_red, CB_red, point_A, point_B),
        }
        for scheme in ("fixed_draw", "redraw"):
            ch = results["chain"][m][scheme]
            print(f"[chain/{m}/{scheme}] rho={ch['cross_link_correlation']:+.4f} "
                  f"h={ch['h_point']:.4f} "
                  f"understatement={ch['understatement_of_width_pct']:.1f}%")
    print(f"[chain+joint] pass in {time.time() - t3:.1f}s")

    # ---- outputs ----
    # replicate persistence (added 2026-07-27, no effect on any estimate)
    for s in STAGES:
        store[f"y_eval_{s}"] = yte[s]
        for m in MODEL_TAGS:
            store[f"p_full_{s}_{m}"] = pre[(s, m)][0]
    out_npz = os.path.join(out_dir, f"replicates-{stem}.npz")
    np.savez_compressed(out_npz, **store)
    print(f"[replicates] {out_npz}")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    out_json = os.path.join(out_dir, f"results-{stem}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_csv = os.path.join(out_dir, f"episodes-{stem}.csv")
    split_of = {int(i): "train" for i in idx_tr}
    split_of.update({int(i): "eval" for i in idx_te})
    cols = ["i", "channel", "sources", "tool", "amount_eur", "payee",
            "payee_unknown", "cap", "n_methods", "ttl_s", "issuer_on_list",
            "tamper_detail", "yA", "A_deny_reason", "yB", "B_mandate",
            "B_execution", "B_sole_control", "B_legal_effect", "B_diag"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_seed"] + cols + ["split"] +
                   [f"fA_{c}" for c in FEATURES_A] + [f"fB_{c}" for c in FEATURES_B])
        for e in eps:
            w.writerow([episode_seed] + [e[c] for c in cols] +
                       [split_of[e["i"]]] + list(e["fA"]) + list(e["fB"]))
    print(f"[done] {out_json}\n[done] {out_csv}\n[total] "
          f"{results['runtime_seconds']}s")


if __name__ == "__main__":
    main()
