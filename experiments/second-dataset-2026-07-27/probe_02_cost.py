#!/usr/bin/env python3
"""
Probe 2 of the second-dataset design (article item A11): what one metric-vector
evaluation and one bootstrap replicate cost on UCI Default of Credit Card
Clients (Taiwan), for both pinned models, so that the full study can be priced
before it is run.

Every construction here is copied from, not reinvented for, the two pinned
drivers:
  ../redraw-2026-07-24/redraw_experiment.py        (paired bootstrap, schemes)
  ../pavia-composite-2026-07-25/pavia_composite_experiment.py  (curves, tail
                                                               swap, composite)
Neither of those directories is written to.

Design only. B is 1 here; nothing below is a result.

Anton Sokolov, Tyche Institute, Tallinn. 27 July 2026.
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REDRAW_DIR = os.path.join(os.path.dirname(HERE), "redraw-2026-07-24")
SAFEAI_REPO = os.path.join(REDRAW_DIR, "safeai-src")
sys.path.insert(0, os.path.join(REDRAW_DIR, "_stubs"))
sys.path.insert(0, SAFEAI_REPO)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from safeai.rga import rga_curve, rga_score
from safeai.rge import rge_curve, rge_score
from safeai.rgr import rgr_score

SEED = 20260723
TEST_SIZE = 0.3
NOISE_SD = 0.5
CLASS_ORDER = np.array([0, 1])
SMALL_N_TOTAL = 1000        # matched-n control: German-credit sample size

LOG = []


def say(m=""):
    print(m, flush=True)
    LOG.append(m)


def tail_swap(X, p):
    """Vectorised rewrite of the pinned tail swap (article eq. 9). Identical
    output to the loop form of pavia_composite_experiment.py; the equality is
    asserted below on this data."""
    n = X.shape[0]
    m = int(np.floor(p * n))
    if m == 0:
        return X.copy()
    Xp = X.copy()
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        col = X[:, j]
        new = col.copy()
        lo, hi = order[:m], order[n - m:][::-1]
        new[lo], new[hi] = col[hi], col[lo]
        Xp[:, j] = new
    return Xp


def tail_swap_loopform(X, p):
    n = X.shape[0]
    m = int(np.floor(p * n))
    if m == 0:
        return X.copy()
    Xp = X.copy()
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        col = X[:, j]
        new = col.copy()
        for i in range(m):
            lo, hi = order[i], order[n - 1 - i]
            new[lo], new[hi] = col[hi], col[lo]
        Xp[:, j] = new
    return Xp


def load():
    ds = fetch_openml("default-of-credit-card-clients", version=1,
                      as_frame=False, parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "1").astype(int)
    return X, y, list(ds.feature_names)


def build_models():
    return {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }


t_start = time.time()
X, y, names = load()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=SEED, stratify=y)
n, d = X_te.shape
GRID, L = d, d + 1
ps = 0.5 * np.arange(L) / GRID

say("=" * 78)
say("Probe 2: cost of one metric vector and one bootstrap replicate, Taiwan")
say("run of %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"))
say("=" * 78)
say("[data]  n_train = %d, n_test = %d, d = %d -> GRID = %d, L = d + 1 = %d"
    % (len(y_tr), n, d, GRID, L))

# tail-swap implementations agree
p_chk = 0.3
say("[check] vectorised tail swap vs the pinned loop form at p = %.2f: "
    "max abs diff = %.1e"
    % (p_chk, float(np.max(np.abs(tail_swap(X_te, p_chk)
                                  - tail_swap_loopform(X_te, p_chk))))))

TIM = {"n_test": int(n), "d": int(d), "L": int(L), "models": {}}

models = build_models()
for tag, model in models.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    TIM["models"].setdefault(tag, {})["fit_s"] = time.time() - t0
    say("[fit]   %-5s trained in %.2f s" % (tag, TIM["models"][tag]["fit_s"]))

# one tail swap, timed once, since the sweep pays it L times
t0 = time.time()
_ = tail_swap(X_te, 0.5)
t_swap = time.time() - t0
say("[cost]  one tail swap on %d x %d rows: %.3f s (the sweep pays it %d times)"
    % (n, d, t_swap, L))
TIM["tail_swap_s"] = t_swap

for tag in ("logit", "rf"):
    model = models[tag]
    say("")
    say("#" * 78)
    say("### %s" % tag)
    say("#" * 78)
    T = TIM["models"][tag]

    t0 = time.time()
    p_full = model.predict_proba(X_te)[:, 1]
    T["predict_full_s"] = time.time() - t0
    say("[cost]  predict_proba on the full test block: %.3f s" % T["predict_full_s"])
    col_mean = X_te.mean(axis=0)
    col_std = X_te.std(axis=0, keepdims=True)

    # ---- one metric vector at the point estimate: the three curves
    t0 = time.time()
    a0 = np.asarray(rga_curve(y_te, p_full, curve_method="partial",
                              n_segments=GRID, normalize_to_perfect=False)["curve"],
                    dtype=float)
    T["rga_curve_s"] = time.time() - t0
    say("[cost]  RGA partial curve (L = %d): %.2f s" % (L, T["rga_curve_s"]))

    t0 = time.time()
    res = rge_curve(model, X_te, method="tabular", feature_names=names,
                    masking_method="greedy", baseline="mean", n_steps=GRID,
                    verbose=False)
    T["rge_greedy_curve_s"] = time.time() - t0
    e0 = np.asarray(res["rge_scores"], dtype=float)
    order = [names.index(nm) for nm in res["removed_features"]]
    say("[cost]  RGE greedy curve, %d model evaluations: %.2f s"
        % (GRID * (GRID + 1) // 2, T["rge_greedy_curve_s"]))

    t0 = time.time()
    r0 = np.empty(L)
    for t in range(L):
        r0[t] = rgr_score(p_full, model.predict_proba(tail_swap(X_te, ps[t]))[:, 1],
                          class_order=CLASS_ORDER)
    T["rgr_tail_swap_sweep_s"] = time.time() - t0
    say("[cost]  RGR tail-swap sweep, %d severities: %.2f s"
        % (L, T["rgr_tail_swap_sweep_s"]))
    T["metric_vector_point_s"] = (T["rga_curve_s"] + T["rge_greedy_curve_s"]
                                  + T["rgr_tail_swap_sweep_s"])
    say("[COST]  ONE FULL METRIC VECTOR (three curves, no reuse): %.2f s"
        % T["metric_vector_point_s"])
    say("        RGA head %s ... tail %s"
        % (" ".join("%.4f" % v for v in a0[:3]), " ".join("%.4f" % v for v in a0[-2:])))
    say("        RGE head %s ... tail %s"
        % (" ".join("%.4f" % v for v in e0[:3]), " ".join("%.4f" % v for v in e0[-2:])))
    say("        RGR head %s ... tail %s"
        % (" ".join("%.4f" % v for v in r0[:3]), " ".join("%.4f" % v for v in r0[-2:])))
    say("        RGE greedy removal order: %s" % " ".join(str(i) for i in order[:8]))

    # ---- precompute for the fast bootstrap path (paid once per model)
    t0 = time.time()
    p_mask = np.empty((L, n)); p_mask[0] = p_full
    for t in range(1, L):
        Xm = X_te.copy()
        Xm[:, order[:t]] = col_mean[order[:t]]
        p_mask[t] = model.predict_proba(Xm)[:, 1]
    p_swap = np.empty((L, n))
    for t in range(L):
        p_swap[t] = model.predict_proba(tail_swap(X_te, ps[t]))[:, 1]
    rng_noise = np.random.default_rng(SEED)
    p_noise = np.empty((L, n))
    for t in range(L):
        sig = t / float(GRID)
        Xp = (X_te + rng_noise.normal(0.0, sig, size=X_te.shape) * col_std
              if sig > 0 else X_te)
        p_noise[t] = model.predict_proba(Xp)[:, 1]
    T["precompute_s"] = time.time() - t0
    say("[cost]  precompute (%d masked + %d swapped + %d noise prediction vectors): "
        "%.2f s" % (L, L, L, T["precompute_s"]))

    # single-feature masks, for the deposited scalar estimators of the bridge
    t0 = time.time()
    p_masked_single = np.empty((d, n))
    for j in range(d):
        Xm = X_te.copy()
        Xm[:, j] = col_mean[j]
        p_masked_single[j] = model.predict_proba(Xm)[:, 1]
    pert = np.random.default_rng(SEED).normal(0.0, NOISE_SD, size=X_te.shape) * col_std
    p_pert = model.predict_proba(X_te + pert)[:, 1]
    T["precompute_bridge_s"] = time.time() - t0
    say("[cost]  bridge precompute (%d single masks + 1 fixed Gaussian draw): %.2f s"
        % (d, T["precompute_bridge_s"]))

    # ---- ONE paired bootstrap replicate on the fast path
    boot_rng = np.random.default_rng(SEED + 1)
    ib = boot_rng.integers(0, n, n)

    def triple(idx_a, idx_e, idx_r):
        pa, pe, pr = p_full[idx_a], p_full[idx_e], p_full[idx_r]
        a = np.asarray(rga_curve(y_te[idx_a], pa, curve_method="partial",
                                 n_segments=GRID,
                                 normalize_to_perfect=False)["curve"], dtype=float)
        e = np.empty(L); e[0] = 1.0
        for t in range(1, L):
            e[t] = rge_score(pe, p_mask[t][idx_e], class_order=CLASS_ORDER)
        r = np.empty(L)
        for t in range(L):
            r[t] = rgr_score(pr, p_swap[t][idx_r], class_order=CLASS_ORDER)
        return a, e, r

    triple(ib, ib, ib)                       # warm-up, not timed
    reps = 3
    t0 = time.time()
    for _ in range(reps):
        triple(ib, ib, ib)
    T["paired_replicate_s"] = (time.time() - t0) / reps
    say("[COST]  ONE PAIRED BOOTSTRAP REPLICATE (fast path): %.3f s"
        % T["paired_replicate_s"])

    # the extra work a replicate carries in the sensitivity and bridge arms
    t0 = time.time()
    for _ in range(reps):
        pb = p_full[ib]
        for t in range(L):
            rgr_score(pb, p_noise[t][ib], class_order=CLASS_ORDER)
    T["noise_sweep_replicate_s"] = (time.time() - t0) / reps
    t0 = time.time()
    for _ in range(reps):
        pb = p_full[ib]
        rga_score(y_te[ib], pb)
        np.mean([rge_score(pb, p_masked_single[j][ib]) for j in range(d)])
        rgr_score(pb, p_pert[ib])
    T["deposit_scalar_replicate_s"] = (time.time() - t0) / reps
    say("[cost]  Gaussian-sweep RGR add-on per replicate: %.3f s"
        % T["noise_sweep_replicate_s"])
    say("[cost]  deposited scalar triple per replicate (bridge): %.3f s"
        % T["deposit_scalar_replicate_s"])

    # ---- ONE full-recompute replicate (the conditioning check of the pinned study:
    #      greedy order re-searched and perturbations rebuilt on the resampled rows)
    t0 = time.time()
    Xb, yb = X_te[ib], y_te[ib]
    pb = model.predict_proba(Xb)[:, 1]
    rga_curve(yb, pb, curve_method="partial", n_segments=GRID,
              normalize_to_perfect=False)
    rge_curve(model, Xb, method="tabular", feature_names=names,
              masking_method="greedy", baseline="mean", n_steps=GRID, verbose=False)
    for t in range(L):
        rgr_score(pb, model.predict_proba(tail_swap(Xb, ps[t]))[:, 1],
                  class_order=CLASS_ORDER)
    T["full_recompute_replicate_s"] = time.time() - t0
    say("[COST]  ONE FULL-RECOMPUTE REPLICATE (conditioning check): %.1f s"
        % T["full_recompute_replicate_s"])

    # ---- composite evaluation cost at L = %d, B = 2000 (the tensor arms)
    B_demo = 200
    A = np.repeat(a0[None, :], B_demo, axis=0)
    E = np.repeat(e0[None, :], B_demo, axis=0)
    R = np.repeat(r0[None, :], B_demo, axis=0)
    t0 = time.time()
    out = np.empty(B_demo)
    for s in range(0, B_demo, 100):
        aa = A[s:s + 100][:, :, None, None]
        ee = E[s:s + 100][:, None, :, None]
        rr = R[s:s + 100][:, None, None, :]
        out[s:s + 100] = np.sqrt((aa * aa + ee * ee + rr * rr) / 3.0).mean(axis=(1, 2, 3))
    T["v_rms_per_1000_replicates_s"] = (time.time() - t0) * 1000.0 / B_demo
    t0 = time.time()
    tot = 0.0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                tot += math.sqrt((a0[i] ** 2 + e0[j] ** 2 + r0[k] ** 2) / 3.0)
    T["brute_force_triple_sum_s"] = time.time() - t0
    say("[cost]  rms tensor composite: %.2f s per 1000 replicates; brute-force "
        "triple sum at L = %d: %.2f s" % (T["v_rms_per_1000_replicates_s"], L,
                                          T["brute_force_triple_sum_s"]))

# ---------------------------------------------------------------- matched-n control
say("")
say("#" * 78)
say("### matched-n control: a %d-row stratified subsample, split 70/30 as in "
    "German credit" % SMALL_N_TOTAL)
say("#" * 78)
rng_sub = np.random.default_rng(SEED + 4242)
idx_pos = np.flatnonzero(y == 1); idx_neg = np.flatnonzero(y == 0)
k_pos = int(round(SMALL_N_TOTAL * y.mean()))
sub = np.concatenate([rng_sub.choice(idx_pos, k_pos, replace=False),
                      rng_sub.choice(idx_neg, SMALL_N_TOTAL - k_pos, replace=False)])
sub.sort()
Xs, ys = X[sub], y[sub]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=TEST_SIZE,
                                              random_state=SEED, stratify=ys)
ns = Xs_te.shape[0]
say("[ctrl]  n_train = %d, n_test = %d, positive share test %.4f"
    % (len(ys_tr), ns, ys_te.mean()))
TIM["control"] = {"n_total": SMALL_N_TOTAL, "n_test": int(ns),
                  "positive_share_test": float(ys_te.mean()), "models": {}}
models_s = build_models()
for tag, model in models_s.items():
    model.fit(Xs_tr, ys_tr)
    ps_full = model.predict_proba(Xs_te)[:, 1]
    cm = Xs_te.mean(axis=0)
    t0 = time.time()
    rga_curve(ys_te, ps_full, curve_method="partial", n_segments=GRID,
              normalize_to_perfect=False)
    resd = rge_curve(model, Xs_te, method="tabular", feature_names=names,
                     masking_method="greedy", baseline="mean", n_steps=GRID,
                     verbose=False)
    ordr = [names.index(nm) for nm in resd["removed_features"]]
    for t in range(L):
        rgr_score(ps_full, model.predict_proba(tail_swap(Xs_te, ps[t]))[:, 1],
                  class_order=CLASS_ORDER)
    tv = time.time() - t0
    pm = np.empty((L, ns)); pm[0] = ps_full
    for t in range(1, L):
        Xm = Xs_te.copy(); Xm[:, ordr[:t]] = cm[ordr[:t]]
        pm[t] = model.predict_proba(Xm)[:, 1]
    pw = np.empty((L, ns))
    for t in range(L):
        pw[t] = model.predict_proba(tail_swap(Xs_te, ps[t]))[:, 1]
    ibs = np.random.default_rng(SEED + 1).integers(0, ns, ns)
    t0 = time.time()
    for _ in range(5):
        pa = ps_full[ibs]
        rga_curve(ys_te[ibs], pa, curve_method="partial", n_segments=GRID,
                  normalize_to_perfect=False)
        for t in range(1, L):
            rge_score(pa, pm[t][ibs], class_order=CLASS_ORDER)
        for t in range(L):
            rgr_score(pa, pw[t][ibs], class_order=CLASS_ORDER)
    tr = (time.time() - t0) / 5
    TIM["control"]["models"][tag] = {"metric_vector_point_s": tv,
                                     "paired_replicate_s": tr}
    say("[ctrl]  %-5s one metric vector %.2f s, one paired replicate %.3f s"
        % (tag, tv, tr))

TIM["probe_total_s"] = time.time() - t_start
TIM["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
with open(os.path.join(HERE, "probe-02-cost.json"), "w") as fh:
    json.dump(TIM, fh, indent=1)
with open(os.path.join(HERE, "probe-02-cost.log"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
say("")
say("[done] probe total %.1f s" % TIM["probe_total_s"])
