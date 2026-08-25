#!/usr/bin/env python3
"""
Probe 3 of the second-dataset design (article item A11): the costs probe 2 did
not measure, an independent re-measurement of the two it did, and the runtime
projection for the whole study assembled from measured constants only.

Probe 2 priced the curve construction of pavia_composite_experiment.py. It did
not price the two arms that live in redraw_experiment.py -- the per-replicate
perturbation redraw of scheme B, and the shared-index chain pass -- and it did
not establish how the per-replicate cost scales in n, which is the number that
decides what a reduced design would buy. This probe supplies those, re-measures
the metric vector and the paired replicate so that probe 2 is corroborated
rather than trusted, and prints the projection.

Every construction is copied from, not reinvented for, the pinned drivers:
  ../redraw-2026-07-24/redraw_experiment.py
  ../pavia-composite-2026-07-25/pavia_composite_experiment.py
  ../pavia-composite-2026-07-25/verify_cross_terms.py
None of those directories is written to.

Design only. Nothing below is a result: B is 1 or a handful everywhere, and the
scaling block scores sliced arrays whose VALUES are meaningless -- only their
sizes, and therefore the timings, are representative.

Anton Sokolov, Tyche Institute, Tallinn. 27 July 2026.
"""
import json
import os
import socket
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
B_PLAN = 2000
B_FULL_PLAN = {"logit": 300, "rf": 30}
REPS = 5                      # repeats for each timed micro-benchmark

LOG = []


def say(m=""):
    print(m, flush=True)
    LOG.append(m)


def tail_swap(X, p):
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


def build_models():
    return {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }


def timeit(fn, reps=REPS):
    fn()                                  # warm-up, not timed
    t0 = time.time()
    for _ in range(reps):
        fn()
    return (time.time() - t0) / reps


t_start = time.time()
say("=" * 78)
say("Probe 3: the unmeasured costs, a re-measurement, and the projection")
say("run of %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"))
say("=" * 78)

# ---------------------------------------------------------------- offline check
# The pinned drivers call fetch_openml at every run. If the OpenML cache is cold
# the full study inherits a network dependency, which a reproducibility statement
# would have to disclose. Settle it by measurement: block the socket layer and
# fetch again.
_real_socket = socket.socket


class _NoSocket(_real_socket):
    def __init__(self, *a, **k):
        raise OSError("probe 3: network deliberately disabled for the offline check")


socket.socket = _NoSocket
try:
    t0 = time.time()
    ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                      parser="liac-arff")
    t_offline = time.time() - t0
    offline_ok = True
except Exception as exc:                                   # noqa: BLE001
    t_offline = float("nan")
    offline_ok = False
    say("[offline] fetch FAILED with the socket layer disabled: %s: %s"
        % (exc.__class__.__name__, exc))
finally:
    socket.socket = _real_socket
if offline_ok:
    say("[offline] fetch_openml succeeded with the socket layer disabled in %.2f s "
        "-> the local scikit-learn OpenML cache serves data id %s; the full study "
        "needs no network" % (t_offline, ds.details.get("id")))

X = ds.data.astype(float)
y = (ds.target == "1").astype(int)
names = list(ds.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=SEED, stratify=y)
n, d = X_te.shape
GRID, L = d, d + 1
ps = 0.5 * np.arange(L) / GRID
say("[data]  n_test = %d, d = %d, GRID = %d, L = %d" % (n, d, GRID, L))

OUT = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "offline_fetch_ok": offline_ok, "offline_fetch_seconds": t_offline,
       "n_test": int(n), "d": int(d), "L": int(L),
       "B_plan": B_PLAN, "B_full_plan": B_FULL_PLAN,
       "reps_per_benchmark": REPS, "models": {}}

# --------------------------------------------- tail-swap bite, Taiwan vs German
# The perturbation is a rank exchange. Where a column is tie-heavy the exchange
# can be an identity on the swapped rows, so the family's severity is not the
# nominal p but the share of rows whose value actually moves. Measured on both
# datasets so that the severity axes are known to be comparable before anything
# is compared across them.
say("")
say("[bite] share of test rows whose value actually changes under the tail swap")
bite = {}
for lab, Xt in (("taiwan_n%d" % n, X_te),):
    bite[lab] = {}
    for p in (0.125, 0.25, 0.5):
        bite[lab]["p=%.3f" % p] = float((tail_swap(Xt, p) != Xt).mean())
dsg = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
Xg = dsg.data.astype(float)
yg = (dsg.target == "good").astype(int)
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=TEST_SIZE,
                                              random_state=SEED, stratify=yg)
bite["german_n%d" % Xg_te.shape[0]] = {}
for p in (0.125, 0.25, 0.5):
    bite["german_n%d" % Xg_te.shape[0]]["p=%.3f" % p] = float(
        (tail_swap(Xg_te, p) != Xg_te).mean())
for lab in bite:
    say("  %-16s %s" % (lab, "  ".join("%s -> %.3f" % (k, v)
                                       for k, v in bite[lab].items())))
OUT["tail_swap_bite_mean_over_columns"] = bite
del Xg, yg, Xg_tr, Xg_te, yg_tr, yg_te, dsg

# ---------------------------------------------------------------- the two models
models = build_models()
for tag, model in models.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    OUT["models"].setdefault(tag, {})["fit_s"] = time.time() - t0

col_mean = X_te.mean(axis=0)
col_std = X_te.std(axis=0, keepdims=True)

for tag in ("logit", "rf"):
    model = models[tag]
    T = OUT["models"][tag]
    say("")
    say("#" * 78)
    say("### %s   (fit %.2f s)" % (tag, T["fit_s"]))
    say("#" * 78)

    p_full = model.predict_proba(X_te)[:, 1]

    # --- (1) re-measurement: one full metric vector, three curves, no reuse
    t0 = time.time()
    a0 = np.asarray(rga_curve(y_te, p_full, curve_method="partial", n_segments=GRID,
                              normalize_to_perfect=False)["curve"], dtype=float)
    t_rga = time.time() - t0
    t0 = time.time()
    res = rge_curve(model, X_te, method="tabular", feature_names=names,
                    masking_method="greedy", baseline="mean", n_steps=GRID,
                    verbose=False)
    t_rge = time.time() - t0
    e0 = np.asarray(res["rge_scores"], dtype=float)
    order = [names.index(nm) for nm in res["removed_features"]]
    t0 = time.time()
    r0 = np.empty(L)
    for t in range(L):
        r0[t] = rgr_score(p_full, model.predict_proba(tail_swap(X_te, ps[t]))[:, 1],
                          class_order=CLASS_ORDER)
    t_rgr = time.time() - t0
    T["metric_vector_point_s"] = t_rga + t_rge + t_rgr
    T["metric_vector_parts_s"] = {"rga_curve": t_rga, "rge_greedy_curve": t_rge,
                                  "rgr_tail_swap_sweep": t_rgr}
    say("[remeasure] ONE FULL METRIC VECTOR = %.2f s "
        "(RGA %.2f + RGE greedy %.2f + RGR sweep %.2f)"
        % (T["metric_vector_point_s"], t_rga, t_rge, t_rgr))

    # --- precompute for the fast bootstrap path (paid once per model)
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
    T["precompute_curve_s"] = time.time() - t0

    t0 = time.time()
    p_mask_single = np.empty((d, n))
    for j in range(d):
        Xm = X_te.copy()
        Xm[:, j] = col_mean[j]
        p_mask_single[j] = model.predict_proba(Xm)[:, 1]
    pert_fixed = np.random.default_rng(SEED).normal(0.0, NOISE_SD,
                                                    size=X_te.shape) * col_std
    p_pert_fixed = model.predict_proba(X_te + pert_fixed)[:, 1]
    T["precompute_bridge_s"] = time.time() - t0
    say("[precompute] curve path %.2f s, deposited-scalar path %.2f s"
        % (T["precompute_curve_s"], T["precompute_bridge_s"]))

    ib = np.random.default_rng(SEED + 1).integers(0, n, n)

    # --- (2) re-measurement: one paired bootstrap replicate on the curve path
    def curve_triple(idx):
        pa = p_full[idx]
        a = np.asarray(rga_curve(y_te[idx], pa, curve_method="partial",
                                 n_segments=GRID,
                                 normalize_to_perfect=False)["curve"], dtype=float)
        e = np.empty(L); e[0] = 1.0
        for t in range(1, L):
            e[t] = rge_score(pa, p_mask[t][idx], class_order=CLASS_ORDER)
        r = np.empty(L)
        for t in range(L):
            r[t] = rgr_score(pa, p_swap[t][idx], class_order=CLASS_ORDER)
        return a, e, r

    T["curve_replicate_s"] = timeit(lambda: curve_triple(ib))
    say("[remeasure] ONE PAIRED CURVE REPLICATE (fast path) = %.4f s"
        % T["curve_replicate_s"])

    # --- Gaussian-sweep RGR add-on, carried by every paired replicate of the
    #     sensitivity arm and by the bridge
    def noise_sweep(idx):
        pb = p_full[idx]
        return [rgr_score(pb, p_noise[t][idx], class_order=CLASS_ORDER)
                for t in range(L)]

    T["noise_sweep_replicate_s"] = timeit(lambda: noise_sweep(ib))

    # --- the deposited SCALAR triple: the redraw driver's metric_vector_fixed
    def scalar_triple(idx):
        pb = p_full[idx]
        return (rga_score(y_te[idx], pb),
                float(np.mean([rge_score(pb, p_mask_single[j][idx])
                               for j in range(d)])),
                rgr_score(pb, p_pert_fixed[idx]))

    T["scalar_replicate_s"] = timeit(lambda: scalar_triple(ib))
    say("[cost] Gaussian-sweep add-on %.4f s ; deposited scalar triple %.4f s"
        % (T["noise_sweep_replicate_s"], T["scalar_replicate_s"]))

    # --- (3) NOT measured by probe 2: scheme B, the per-replicate perturbation
    #     redraw. redraw_experiment.py draws a fresh Gaussian matrix per replicate
    #     and re-predicts the whole test block before any resampling happens.
    counter = {"b": 0}

    def redraw_one():
        rb = np.random.default_rng(SEED + 100_000 + counter["b"])
        counter["b"] += 1
        pert_b = rb.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
        return model.predict_proba(X_te + pert_b)[:, 1]

    T["redraw_perturbation_and_predict_s"] = timeit(redraw_one)
    say("[NEW ] scheme-B redraw, one fresh perturbation + full-block predict "
        "= %.4f s" % T["redraw_perturbation_and_predict_s"])

    # --- (4) NOT measured by probe 2: one full-recompute replicate, timed here
    #     again because it is the single most expensive constant in the plan
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
    say("[remeasure] ONE FULL-RECOMPUTE REPLICATE (conditioning check) = %.1f s"
        % T["full_recompute_replicate_s"])

    # --- (5) how the replicate cost scales in n. Timing only: the arrays below
    #     are SLICES, so the scores they produce are meaningless. Only the sizes,
    #     and therefore the timings, are representative.
    scal = {}
    for ns in (300, 1000, 3000, 9000):
        if ns > n:
            continue
        idx_s = np.random.default_rng(SEED + 1).integers(0, ns, ns)
        y_s, pf_s = y_te[:ns], p_full[:ns]
        pm_s, pw_s = p_mask[:, :ns], p_swap[:, :ns]

        def one(idx=idx_s, ys=y_s, pf=pf_s, pm=pm_s, pw=pw_s):
            pa = pf[idx]
            rga_curve(ys[idx], pa, curve_method="partial", n_segments=GRID,
                      normalize_to_perfect=False)
            for t in range(1, L):
                rge_score(pa, pm[t][idx], class_order=CLASS_ORDER)
            for t in range(L):
                rgr_score(pa, pw[t][idx], class_order=CLASS_ORDER)

        scal[str(ns)] = timeit(one)
    T["replicate_cost_by_n_s"] = scal
    lo, hi = min(scal), max(scal)
    expo = (np.log(scal[hi] / scal[lo]) / np.log(int(hi) / int(lo)))
    T["replicate_cost_exponent_in_n"] = float(expo)
    say("[scaling] paired replicate cost by n: %s  -> empirical exponent %.2f"
        % ("  ".join("n=%s %.4f s" % (k, v) for k, v in scal.items()), expo))

# ---------------------------------------------------------------- memory
idx_bytes_int64 = B_PLAN * n * 8
say("")
say("[memory] one B x n index stream at B = %d, n = %d: %.0f MB as int64, "
    "%.0f MB as int32" % (B_PLAN, n, idx_bytes_int64 / 1e6, idx_bytes_int64 / 2e6))
say("         the pinned curve driver holds one paired stream plus three "
    "independent streams = %.0f MB int64" % (4 * idx_bytes_int64 / 1e6))
OUT["index_stream_MB_int64"] = idx_bytes_int64 / 1e6
OUT["index_streams_held_MB_int64"] = 4 * idx_bytes_int64 / 1e6

# ---------------------------------------------------------------- projection
say("")
say("#" * 78)
say("### projection of the full study, from the constants measured above")
say("#" * 78)


def M(tag, key):
    return OUT["models"][tag][key]


plan = []
# --- driver 1: the scalar/chain driver (redraw_experiment.py on Taiwan)
for tag in ("logit", "rf"):
    plan.append(("D1 %s deposited-scalar precompute" % tag, M(tag, "precompute_bridge_s")))
    plan.append(("D1 %s scheme-B redraw, B fresh perturbations" % tag,
                 B_PLAN * M(tag, "redraw_perturbation_and_predict_s")))
    plan.append(("D1 %s paired pass, both schemes" % tag,
                 B_PLAN * M(tag, "scalar_replicate_s")))
plan.append(("D1 chain, shared stream, both models, both schemes",
             B_PLAN * (M("logit", "scalar_replicate_s") + M("rf", "scalar_replicate_s"))))
# --- driver 2: the curve/volume driver (pavia_composite_experiment.py on Taiwan)
for tag in ("logit", "rf"):
    plan.append(("D2 %s point curves" % tag, M(tag, "metric_vector_point_s")))
    plan.append(("D2 %s curve precompute" % tag, M(tag, "precompute_curve_s")))
    plan.append(("D2 %s paired pass + Gaussian-sweep arm" % tag,
                 B_PLAN * (M(tag, "curve_replicate_s") + M(tag, "noise_sweep_replicate_s"))))
    plan.append(("D2 %s independent-stream pass" % tag,
                 B_PLAN * M(tag, "curve_replicate_s")))
    plan.append(("D2 %s conditioning check, B_full = %d" % (tag, B_FULL_PLAN[tag]),
                 B_FULL_PLAN[tag] * M(tag, "full_recompute_replicate_s")))
# --- driver 3: the estimator bridge (verify_cross_terms.py on Taiwan)
for tag in ("logit", "rf"):
    plan.append(("D3 %s bridge: paired curve + independent + scalar + sweep" % tag,
                 B_PLAN * (2 * M(tag, "curve_replicate_s") + M(tag, "scalar_replicate_s")
                           + M(tag, "noise_sweep_replicate_s"))))

total = sum(v for _, v in plan)
for nm, v in plan:
    say("  %-58s %8.1f s  (%5.1f min)" % (nm, v, v / 60.0))
say("  %-58s %8.1f s  (%5.1f min)" % ("TOTAL, serial, one core-set", total, total / 60.0))
OUT["projection"] = {"items_s": {nm: v for nm, v in plan}, "total_s": total,
                     "total_min": total / 60.0}

# --- what a reduced design would cost
red = {}
red["B_1000_instead_of_2000"] = total - 0.5 * (total - sum(
    v for nm, v in plan if "conditioning" in nm or "point curves" in nm
    or "precompute" in nm))
red["rf_conditioning_at_B_full_10"] = total - (B_FULL_PLAN["rf"] - 10) * M(
    "rf", "full_recompute_replicate_s")
OUT["reduced_designs_s"] = red
say("")
say("[reduce] B = 1000 everywhere except the fixed costs: %.1f min"
    % (red["B_1000_instead_of_2000"] / 60.0))
say("[reduce] rf conditioning check at B_full = 10 instead of %d: %.1f min"
    % (B_FULL_PLAN["rf"], red["rf_conditioning_at_B_full_10"] / 60.0))

OUT["probe_total_s"] = time.time() - t_start
with open(os.path.join(HERE, "probe-03-projection.json"), "w") as fh:
    json.dump(OUT, fh, indent=1)
with open(os.path.join(HERE, "probe-03-projection.log"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
say("")
say("[done] probe total %.1f s" % OUT["probe_total_s"])
