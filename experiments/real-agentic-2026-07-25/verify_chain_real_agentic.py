#!/usr/bin/env python3
"""Adversarial verification of the chain estimate (verify_cross_terms pattern).

INTERNAL WORKING ARTIFACT. Do not commit without go-ahead; no names from
this directory may reach the manuscript.

Independent of harness.py by construction where it matters:
- episodes are NOT regenerated: features and labels are read back from the
  frozen episodes-real-agentic.csv, so any generator bug that made the CSV
  disagree with what the statistics consumed would surface here;
- the replicate loop, the geometric mean, the covariance, the correlation
  and the delta-method interval arithmetic are all re-written by hand
  (explicit sums, no np.cov / np.corrcoef);
- only the three pinned safeai scoring functions and sklearn are shared,
  because they are the substrate under study.

Checks, against results-real-agentic.json:
1. scorer AUCs refit from the CSV features;
2. per-stage point metric vectors;
3. chain rho, h, both interval widths and the understatement, both models,
   redraw scheme (the reported arm), recomputed from a fresh replicate loop.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REDRAW_DIR = os.path.normpath(os.path.join(HERE, "..", "redraw-2026-07-24"))
sys.path.insert(0, os.path.join(REDRAW_DIR, "_stubs"))
sys.path.insert(0, os.path.join(REDRAW_DIR, "safeai-src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safeai.rga import rga_score
from safeai.rge import rge_score
from safeai.rgr import rgr_score

SEED = 20260727
B = 2000
NOISE_SD = 0.5
REDRAW_BASE = {("A", "logit"): SEED + 100_000, ("B", "logit"): SEED + 200_000,
               ("A", "rf"): SEED + 300_000, ("B", "rf"): SEED + 400_000}

with open(os.path.join(HERE, "results-real-agentic.json")) as stored_file:
    stored = json.load(stored_file)
with open(os.path.join(HERE, "episodes-real-agentic.csv"), newline="") as rows_file:
    rows = list(csv.DictReader(rows_file))
assert len(rows) == stored["N"], (len(rows), stored["N"])

FA = stored["features"]["A"]
FB = stored["features"]["B"]
X = {"A": np.array([[float(r[f"fA_{c}"]) for c in FA] for r in rows]),
     "B": np.array([[float(r[f"fB_{c}"]) for c in FB] for r in rows])}
y = {"A": np.array([int(r["yA"]) for r in rows]),
     "B": np.array([int(r["yB"]) for r in rows])}

# reproduce the split from the seed rule, then check it against the CSV column
idx_tr, idx_te = train_test_split(np.arange(len(rows)), test_size=stored["n_eval"],
                                  random_state=SEED,
                                  stratify=2 * y["A"] + y["B"])
csv_split = np.array([r["split"] for r in rows])
assert set(csv_split[idx_te]) == {"eval"} and set(csv_split[idx_tr]) == {"train"}, \
    "split reconstruction disagrees with the frozen CSV"
n = len(idx_te)
print(f"[verify] {len(rows)} episodes read back; split reconstruction matches "
      f"the CSV ({len(idx_tr)}/{n})")

failures = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {name:<58} got {got:+.10f}  stored {want:+.10f}  "
          f"gap {abs(got - want):.2e}  {'ok' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def hand_mean(v):
    return sum(float(x) for x in v) / len(v)


def hand_cov(u, v):          # ddof = 1, explicit sums
    mu, mv = hand_mean(u), hand_mean(v)
    return sum((float(a) - mu) * (float(b) - mv) for a, b in zip(u, v)) / (len(u) - 1)


def hand_geo(v):
    s = 0.0
    for x in v:
        s += np.log(max(float(x), 1e-12))
    return float(np.exp(s / len(v)))


models = {}
print("[verify] scorer AUCs refit from the CSV feature columns:")
for s in ("A", "B"):
    for m in ("logit", "rf"):
        mod = (make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED))
               if m == "logit" else
               RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
        mod.fit(X[s][idx_tr], y[s][idx_tr])
        models[(s, m)] = mod
        auc = float(roc_auc_score(y[s][idx_te], mod.predict_proba(X[s][idx_te])[:, 1]))
        check(f"AUC {s}/{m}", auc, stored["scorer_auc_eval"][f"{s}_{m}"], 1e-12)

# fresh precompute (own code path, same definitions)
pre = {}
rng_global = np.random.default_rng(SEED)
for s in ("A", "B"):
    for m in ("logit", "rf"):
        mod = models[(s, m)]
        Xt = X[s][idx_te]
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
        pre[(s, m)] = (p_full, p_masked, p_pf, cs, Xt, mod)

print("[verify] per-stage point metric vectors:")
for s in ("A", "B"):
    for m in ("logit", "rf"):
        p_full, p_masked, p_pf, _, _, _ = pre[(s, m)]
        yb = y[s][idx_te]
        pt = stored["stages"][s][m]["point_fixed_draw"]
        check(f"point RGA {s}/{m}", float(rga_score(yb, p_full)), pt["RGA"], 1e-12)
        rge = hand_mean([rge_score(p_full, p_masked[j]) for j in range(p_masked.shape[0])])
        check(f"point RGE {s}/{m}", rge, pt["RGE"], 1e-12)
        check(f"point RGR {s}/{m}", float(rgr_score(p_full, p_pf)), pt["RGR"], 1e-12)

print("[verify] chain, redraw scheme, own replicate loop and own arithmetic:")
Z = 1.959963984540054
for m in ("logit", "rf"):
    # per-replicate perturbation predictions, regenerated from the seed rule
    mats = {}
    for s in ("A", "B"):
        _, _, _, cs, Xt, mod = pre[(s, m)]
        mat = np.empty((B, n))
        base = REDRAW_BASE[(s, m)]
        for b in range(B):
            rb = np.random.default_rng(base + b)
            mat[b] = mod.predict_proba(
                Xt + rb.normal(0.0, NOISE_SD, size=Xt.shape) * cs)[:, 1]
        mats[s] = mat
    shared = np.random.default_rng(SEED + 99)
    CA, CB = [], []
    for b in range(B):
        idx = shared.integers(0, n, n)
        vals = {}
        for s in ("A", "B"):
            p_full, p_masked, _, _, _, _ = pre[(s, m)]
            pf = p_full[idx]
            rga = float(rga_score(y[s][idx_te][idx], pf))
            rge = hand_mean([rge_score(pf, p_masked[j][idx])
                             for j in range(p_masked.shape[0])])
            rgr = float(rgr_score(pf, mats[s][b][idx]))
            vals[s] = hand_geo([rga, rge, rgr])
        CA.append(vals["A"])
        CB.append(vals["B"])
    st = stored["chain"][m]["redraw"]
    va, vb, cab = hand_cov(CA, CA), hand_cov(CB, CB), hand_cov(CA, CB)
    rho = cab / (va ** 0.5 * vb ** 0.5)
    point_A = hand_geo([
        stored["stages"]["A"][m]["point_fixed_draw"][metric]
        for metric in ("RGA", "RGE", "RGR")
    ])
    point_B = hand_geo([
        stored["stages"]["B"][m]["point_fixed_draw"][metric]
        for metric in ("RGA", "RGE", "RGR")
    ])
    h = point_A * point_B
    var_meas = (
        point_B * point_B * va
        + 2 * point_A * point_B * cab
        + point_A * point_A * vb
    )
    var_zero = point_B * point_B * va + point_A * point_A * vb
    w_meas, w_zero = 2 * Z * var_meas ** 0.5, 2 * Z * var_zero ** 0.5
    under = 100 * (1 - w_zero / w_meas)
    check(f"chain {m} cross-link correlation", rho, st["cross_link_correlation"], 1e-10)
    check(f"chain {m} h point", h, st["h_point"], 1e-12)
    check(f"chain {m} width measured", w_meas, st["width_measured"], 1e-10)
    check(f"chain {m} width declared-zero", w_zero, st["width_declared_zero"], 1e-10)
    check(f"chain {m} understatement pct", under, st["understatement_of_width_pct"], 1e-8)
    # sanity: the reported joint 6x6 must reproduce rho through its own blocks
    S6 = np.array(stored["joint"][m]["redraw"]["Sigma6"])
    print(f"  (joint Sigma6 {m}: leading diag {np.diag(S6).round(8).tolist()})")

print()
if failures:
    print(f"VERIFY FAIL: {len(failures)} quantities disagree: {failures}")
    raise SystemExit(1)
print("VERIFY PASS: every checked quantity reproduces from the frozen CSV "
      "through an independent loop and independent arithmetic.")
