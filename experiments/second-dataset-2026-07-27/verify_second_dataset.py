#!/usr/bin/env python3
"""
Independent recomputation of the second-dataset run, written to the pattern of
../pavia-composite-2026-07-25/verify_cross_terms.py.

The article's frozen-corpus rule is that a number enters the text only after two
independent recomputations agree. This script is the second one. It shares no
code with second_dataset_experiment.py: every construction below is rewritten
from the definitions, the index streams are redrawn from the documented recipe,
and the only thing read out of the primary run is the set of comparison targets,
which are diffed rather than reused.

  A. the factorisation of the triple sum, by brute force, including the negative
     terminal-RGA-knot case that the primary run reports;
  B. the paired replicates and the covariance of the three curve summaries,
     regenerated from scratch, and the delta-method understatements by hand;
  C. the deposited-style SCALAR estimators on the SAME index streams;
  D. the six-rung estimator bridge;
  E. the deterministic knots cannot be the mechanism;
  F. the two-link chain on the shared stream, both conditioning schemes;
  G. the per-severity-index correlations.

Anton Sokolov, Tyche Institute, Tallinn. 28 July 2026.
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REDRAW_DIR = os.path.join(os.path.dirname(HERE), "redraw-2026-07-24")
sys.path.insert(0, os.path.join(REDRAW_DIR, "_stubs"))
sys.path.insert(0, os.path.join(REDRAW_DIR, "safeai-src"))

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
B = int(os.environ.get("SD_B", "2000"))
Z = 1.959963984540054
CLASS_ORDER = np.array([0, 1])
NOISE_SD = 0.5
REDRAW_BASE = {"logit": SEED + 100_000, "rf": SEED + 200_000}
QLO, QHI = 2.5, 97.5
TAGS = ["logit", "rf"]

# Agreement tolerances, per model, fixed from a measured property of the substrate
# rather than from the gaps this script goes on to find (probe-05-rf-determinism.log,
# run before this file was amended):
#
#   the 300-tree forest's predict_proba is NOT bitwise reproducible -- two calls on
#   one fitted forest differ by up to 2.220e-16 -- and at n = 9000 its probabilities
#   are extremely tie-dense: 442 distinct values among 9000 rows, and 8,558 of the
#   8,999 adjacent sorted pairs are closer than 1e-12. RGA, RGE and RGR are rank
#   functionals, so a 2.2e-16 perturbation reorders many near-tied pairs and moves a
#   score by of the order of 1e-7. The primary run sees the same thing internally:
#   its own fast path agrees with the package curve to 1.5e-7 on the forest and to
#   0.0e+00 on the logistic model, where the probabilities are continuous.
#
# The logistic model is therefore held to exact agreement and the forest to a band
# that this floating-point floor cannot exceed. Every measured gap is printed and
# stored regardless of the flag, so the tolerance decides a label and not a claim.
# For scale: the narrowest interval width this study reports is 5.3e-3, so 1e-6 on a
# curve knot is 2e-4 of it.
TOL_POINT = {"logit": 1e-12, "rf": 1e-6}
TOL_CORR = {"logit": 1e-12, "rf": 1e-3}
TOL_PCT = {"logit": 1e-9, "rf": 1e-2}

with open(os.path.join(HERE, "results-second-dataset.json")) as fh:
    STUDY = json.load(fh)

LOG = []
DIFFS = []


def say(m=""):
    print(m, flush=True)
    LOG.append(m)


def note(label, recomputed, stored, tol):
    gap = abs(float(recomputed) - float(stored))
    DIFFS.append({"label": label, "recomputed": float(recomputed),
                  "stored": float(stored), "abs_gap": gap, "tol": tol,
                  "ok": bool(gap <= tol)})
    return gap


def corr3(M):
    return np.corrcoef(M, rowvar=False)


def swap_tails(X, p):
    """Rewritten from the definition: within each column, the lowest floor(p*n)
    order statistics exchange values with the highest, pairwise from the outside
    in. Built here with an index permutation rather than with the primary run's
    fancy-index assignment, so that the two implementations differ."""
    n = X.shape[0]
    m = int(np.floor(p * n))
    if m == 0:
        return X.copy()
    Xp = X.copy()
    for j in range(X.shape[1]):
        rank = np.argsort(X[:, j], kind="stable")
        perm = np.arange(n)
        for i in range(m):
            perm[i], perm[n - 1 - i] = perm[n - 1 - i], perm[i]
        Xp[rank, j] = X[rank[perm], j]
    return Xp


def real_cbrt(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def geo_mean(v):
    return float(np.exp(np.mean(np.log(np.clip(v, 1e-12, None)))))


t_start = time.time()
say("=" * 78)
say("Independent recomputation of the second-dataset run")
say("run of %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"))
say("=" * 78)
say("[target] %s" % os.path.join(HERE, "results-second-dataset.json"))
say("[target] primary run of %s, B = %d" % (STUDY["generated_utc"], STUDY["B"]))

# ------------------------------------------------------------------ A. factorisation
say("")
say("=" * 78)
say("A. factorisation of the triple sum, re-verified by brute force")
say("=" * 78)
rng_chk = np.random.default_rng(12345)
cases = [(rng_chk.uniform(0.01, 1.0, 24), rng_chk.uniform(0.01, 1.0, 24),
          rng_chk.uniform(0.01, 1.0, 24)) for _ in range(3)]
for tag in TAGS:
    cp = STUDY["models"][tag]["curves_point"]
    cases.append((np.array(cp["RGA_partial"]), np.array(cp["RGE_greedy_mean_baseline"]),
                  np.array(cp["RGR_tail_swap"])))
worst = {"arith": 0.0, "geom": 0.0, "rms_gap_from_candidates": None}
neg_terminal = []
for ci, (a, e, r) in enumerate(cases):
    L = len(a)
    s_ar = s_ge = s_rm = 0.0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                s_ar += (a[i] + e[j] + r[k]) / 3.0
                s_ge += real_cbrt(a[i] * e[j] * r[k])
                s_rm += math.sqrt((a[i] ** 2 + e[j] ** 2 + r[k] ** 2) / 3.0)
    s_ar /= L ** 3; s_ge /= L ** 3; s_rm /= L ** 3
    cf_ar = (a.mean() + e.mean() + r.mean()) / 3.0
    cf_ge = np.cbrt(a).mean() * np.cbrt(e).mean() * np.cbrt(r).mean()
    worst["arith"] = max(worst["arith"], abs(s_ar - cf_ar))
    worst["geom"] = max(worst["geom"], abs(s_ge - cf_ge))
    cand1 = math.sqrt((a.mean() ** 2 + e.mean() ** 2 + r.mean() ** 2) / 3.0)
    cand2 = (np.sqrt(a ** 2 / 3).mean() + np.sqrt(e ** 2 / 3).mean()
             + np.sqrt(r ** 2 / 3).mean())
    g1, g2 = abs(s_rm - cand1), abs(s_rm - cand2)
    if worst["rms_gap_from_candidates"] is None or min(g1, g2) < worst["rms_gap_from_candidates"]:
        worst["rms_gap_from_candidates"] = min(g1, g2)
    if ci >= 3:
        tag = TAGS[ci - 3]
        neg_terminal.append((tag, float(a[-1])))
        st = STUDY["models"][tag]["factorisation_check"]
        note("A %s arithmetic tensor" % tag, s_ar,
             st["arithmetic"]["V_vectorised_tensor"], 1e-12)
        note("A %s geometric tensor" % tag, s_ge,
             st["geometric"]["V_vectorised_tensor"], 1e-12)
        note("A %s rms tensor" % tag, s_rm, st["rms"]["V_vectorised_tensor"], 1e-12)
say("  arithmetic closed form : max |brute - closed| = %.3e  (factorises)" % worst["arith"])
say("  geometric  closed form : max |brute - closed| = %.3e  (factorises)" % worst["geom"])
say("  rms: nearest natural candidate closed form is off by at least %.3e"
    % worst["rms_gap_from_candidates"])
for tag, v in neg_terminal:
    say("  terminal RGA knot, %s: %.3e (negative -> the pinned ** form is nan; the real "
        "cube root is used here and agrees with the tensor)" % (tag, v))
assert worst["arith"] < 1e-13 and worst["geom"] < 1e-13
assert worst["rms_gap_from_candidates"] > 1e-4

# ------------------------------------------------------------------ data and models
ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                  parser="liac-arff")
X = ds.data.astype(float)
y = (ds.target == "1").astype(int)
names = list(ds.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
n, d = X_te.shape
GRID, L = d, d + 1
ps = 0.5 * np.arange(L) / GRID
say("")
say("[data] n_test = %d, d = %d, L = %d (stored: %d, %d, %d)"
    % (n, d, L, STUDY["n_test"], STUDY["d"], STUDY["curve_length_L"]))
assert (n, d, L) == (STUDY["n_test"], STUDY["d"], STUDY["curve_length_L"])

models = {
    "logit": make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, random_state=SEED)),
    "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}
for m in models.values():
    m.fit(X_tr, y_tr)

IDX = np.empty((B, n), dtype=np.int64)
boot_rng = np.random.default_rng(SEED + 1)
for b in range(B):
    IDX[b] = boot_rng.integers(0, n, n)
IND = np.empty((3, B, n), dtype=np.int64)
for k in range(3):
    rk = np.random.default_rng(SEED + 10 + k)
    for b in range(B):
        IND[k, b] = rk.integers(0, n, n)

dep_rng = np.random.default_rng(SEED)          # logit first, then rf

KEEP = {}
for tag in TAGS:
    say("")
    say("=" * 78)
    say("%s" % tag)
    say("=" * 78)
    model = models[tag]
    p_full = model.predict_proba(X_te)[:, 1]
    col_mean = X_te.mean(axis=0)
    col_std = X_te.std(axis=0, keepdims=True)

    # ---- point curves, from the package, independently of the primary script
    a0 = np.asarray(rga_curve(y_te, p_full, curve_method="partial", n_segments=GRID,
                              normalize_to_perfect=False)["curve"], dtype=float)
    res = rge_curve(model, X_te, method="tabular", feature_names=names,
                    masking_method="greedy", baseline="mean", n_steps=GRID,
                    verbose=False)
    e0 = np.asarray(res["rge_scores"], dtype=float)
    order = [names.index(nm) for nm in res["removed_features"]]
    r0 = np.array([rgr_score(p_full, model.predict_proba(swap_tails(X_te, ps[t]))[:, 1],
                             class_order=CLASS_ORDER) for t in range(L)])
    cp = STUDY["models"][tag]["curves_point"]
    gaps = (np.max(np.abs(a0 - np.array(cp["RGA_partial"]))),
            np.max(np.abs(e0 - np.array(cp["RGE_greedy_mean_baseline"]))),
            np.max(np.abs(r0 - np.array(cp["RGR_tail_swap"]))))
    say("[point] max abs gap vs the primary curves: RGA %.2e  RGE %.2e  RGR %.2e" % gaps)
    note("point curves %s" % tag, max(gaps), 0.0, TOL_POINT[tag])
    same_order = (cp["RGE_greedy_removal_order"] == [names[i] for i in order])
    say("[point] greedy removal order identical to the primary run: %s" % same_order)
    assert max(gaps) < TOL_POINT[tag]

    # ---- precompute, curve construction
    p_mask = np.empty((L, n)); p_mask[0] = p_full
    for t in range(1, L):
        Xm = X_te.copy()
        Xm[:, order[:t]] = col_mean[order[:t]]
        p_mask[t] = model.predict_proba(Xm)[:, 1]
    p_swap = np.empty((L, n))
    for t in range(L):
        p_swap[t] = model.predict_proba(swap_tails(X_te, ps[t]))[:, 1]
    rng_noise = np.random.default_rng(SEED)
    p_noise = np.empty((L, n))
    for t in range(L):
        sig = t / float(GRID)
        Xp = X_te + rng_noise.normal(0.0, sig, size=X_te.shape) * col_std if sig > 0 else X_te
        p_noise[t] = model.predict_proba(Xp)[:, 1]

    # ---- precompute, deposited scalar construction
    p_masked_single = np.empty((d, n))
    for j in range(d):
        Xm = X_te.copy()
        Xm[:, j] = col_mean[j]
        p_masked_single[j] = model.predict_proba(Xm)[:, 1]
    pert = dep_rng.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
    p_pert = model.predict_proba(X_te + pert)[:, 1]

    # ---- paired pass: curves AND both scalar conventions on the same streams
    t0 = time.time()
    Ab = np.empty((B, L)); Eb = np.empty((B, L)); Rb = np.empty((B, L))
    Rb_noise_mean = np.empty(B)
    SC = np.empty((B, 3))
    SC_co = np.empty((B, 2))
    for b in range(B):
        ib = IDX[b]
        pf = p_full[ib]
        Ab[b] = rga_curve(y_te[ib], pf, curve_method="partial", n_segments=GRID,
                          normalize_to_perfect=False)["curve"]
        Eb[b, 0] = 1.0
        for t in range(1, L):
            Eb[b, t] = rge_score(pf, p_mask[t][ib], class_order=CLASS_ORDER)
        acc = 0.0
        for t in range(L):
            Rb[b, t] = rgr_score(pf, p_swap[t][ib], class_order=CLASS_ORDER)
            acc += rgr_score(pf, p_noise[t][ib], class_order=CLASS_ORDER)
        Rb_noise_mean[b] = acc / L
        SC[b, 0] = rga_score(y_te[ib], pf)
        SC[b, 1] = np.mean([rge_score(pf, p_masked_single[j][ib]) for j in range(d)])
        SC[b, 2] = rgr_score(pf, p_pert[ib])
        SC_co[b, 0] = np.mean([rge_score(pf, p_masked_single[j][ib],
                                         class_order=CLASS_ORDER) for j in range(d)])
        SC_co[b, 1] = rgr_score(pf, p_pert[ib], class_order=CLASS_ORDER)
        if (b + 1) % 500 == 0:
            say("  [paired] %d/%d, %.0f s" % (b + 1, B, time.time() - t0))
    say("[paired] regenerated B = %d replicates in %.1f s" % (B, time.time() - t0))

    # ---- B. covariance of the three curve summaries, from scratch
    Smean = np.column_stack([Ab.mean(1), Eb.mean(1), Rb.mean(1)])
    Scbrt = np.column_stack([np.cbrt(Ab).mean(1), np.cbrt(Eb).mean(1), np.cbrt(Rb).mean(1)])
    Sig = np.cov(Smean, rowvar=False)
    Cor = corr3(Smean)
    stored_sig = np.array(STUDY["models"][tag]["arms"]["arithmetic"]["delta_method"]
                          ["Sigma_of_summaries"])
    say("[B] max |Sigma_recomputed - Sigma_stored| = %.2e" % np.max(np.abs(Sig - stored_sig)))
    note("Sigma %s" % tag, np.max(np.abs(Sig - stored_sig)), 0.0,
         1e-14 if tag == "logit" else 1e-8)
    stored_cor = STUDY["models"][tag]["curve_summaries"]["mean_of_curve"]["Corr"]
    say("[B] corr(curve means): RGA-RGE %+.6f  RGA-RGR %+.6f  RGE-RGR %+.6f"
        % (Cor[0, 1], Cor[0, 2], Cor[1, 2]))
    say("[B] primary          : RGA-RGE %+.6f  RGA-RGR %+.6f  RGE-RGR %+.6f"
        % (stored_cor[0][1], stored_cor[0][2], stored_cor[1][2]))
    note("P1 corr RGE-RGR %s" % tag, Cor[1, 2], stored_cor[1][2], TOL_CORR[tag])
    note("P2 corr RGA-RGR %s" % tag, Cor[0, 2], stored_cor[0][2], TOL_CORR[tag])
    note("corr RGA-RGE %s" % tag, Cor[0, 1], stored_cor[0][1], TOL_CORR[tag])

    g = np.full(3, 1.0 / 3.0)
    var_meas = float(g @ Sig @ g)
    var_decl = float(np.sum(g ** 2 * np.diag(Sig)))
    under = 100.0 * (1.0 - math.sqrt(var_decl) / math.sqrt(var_meas))
    share = 100.0 * (1.0 - var_decl / var_meas)
    st = STUDY["models"][tag]["arms"]["arithmetic"]["delta_method"]
    say("[B] arithmetic by hand: under%% = %+0.4f (primary %+0.4f), cross share %+0.4f "
        "(primary %+0.4f)" % (under, st["understatement_of_width_pct"], share,
                              st["cross_covariance_share_of_variance_pct"]))
    note("P3 arithmetic %s" % tag, under, st["understatement_of_width_pct"], TOL_PCT[tag])

    Sg = np.cov(Scbrt, rowvar=False)
    s0 = np.array([np.cbrt(a0).mean(), np.cbrt(e0).mean(), np.cbrt(r0).mean()])
    gg = np.array([s0[1] * s0[2], s0[0] * s0[2], s0[0] * s0[1]])
    vm = float(gg @ Sg @ gg); vd = float(np.sum(gg ** 2 * np.diag(Sg)))
    ug = 100.0 * (1.0 - math.sqrt(vd) / math.sqrt(vm))
    stg = STUDY["models"][tag]["arms"]["geometric"]["delta_method"]
    say("[B] geometric  by hand: under%% = %+0.4f (primary %+0.4f)"
        % (ug, stg["understatement_of_width_pct"]))
    note("P3 geometric %s" % tag, ug, stg["understatement_of_width_pct"], TOL_PCT[tag])

    M = np.sqrt((a0[:, None, None] ** 2 + e0[None, :, None] ** 2
                 + r0[None, None, :] ** 2) / 3.0)
    inv = 1.0 / M
    c = 1.0 / (3.0 * L ** 3)
    ga = a0 * inv.sum(axis=(1, 2)) * c
    ge = e0 * inv.sum(axis=(0, 2)) * c
    gr = r0 * inv.sum(axis=(0, 1)) * c
    P = np.column_stack([Ab @ ga, Eb @ ge, Rb @ gr])
    Sp = np.cov(P, rowvar=False)
    vm = float(Sp.sum()); vd = float(np.trace(Sp))
    ur = 100.0 * (1.0 - math.sqrt(vd) / math.sqrt(vm))
    strms = STUDY["models"][tag]["arms"]["rms"]["delta_method"]
    say("[B] rms        by hand: under%% = %+0.4f (primary %+0.4f); corr of projections "
        "RGA-RGE %+.4f RGA-RGR %+.4f RGE-RGR %+.4f"
        % (ur, strms["understatement_of_width_pct"], corr3(P)[0, 1], corr3(P)[0, 2],
           corr3(P)[1, 2]))
    note("P3 rms %s" % tag, ur, strms["understatement_of_width_pct"], TOL_PCT[tag])

    # ---- independent-stream pass, curves only, for the empirical column
    t0 = time.time()
    Ai = np.empty((B, L)); Ei = np.empty((B, L)); Ri = np.empty((B, L))
    for b in range(B):
        ia, ie, ir = IND[0, b], IND[1, b], IND[2, b]
        Ai[b] = rga_curve(y_te[ia], p_full[ia], curve_method="partial", n_segments=GRID,
                          normalize_to_perfect=False)["curve"]
        Ei[b, 0] = 1.0
        pe = p_full[ie]
        for t in range(1, L):
            Ei[b, t] = rge_score(pe, p_mask[t][ie], class_order=CLASS_ORDER)
        pr = p_full[ir]
        for t in range(L):
            Ri[b, t] = rgr_score(pr, p_swap[t][ir], class_order=CLASS_ORDER)
    say("[ind] regenerated independent streams in %.1f s" % (time.time() - t0))
    for kind, fn in (("arithmetic",
                      lambda A, E, R: (A.mean(1) + E.mean(1) + R.mean(1)) / 3.0),
                     ("geometric",
                      lambda A, E, R: np.cbrt(A).mean(1) * np.cbrt(E).mean(1)
                      * np.cbrt(R).mean(1))):
        Vb_, Vi_ = fn(Ab, Eb, Rb), fn(Ai, Ei, Ri)
        wb = np.percentile(Vb_, QHI) - np.percentile(Vb_, QLO)
        wi = np.percentile(Vi_, QHI) - np.percentile(Vi_, QLO)
        emp = 100.0 * (1.0 - wi / wb)
        stored = STUDY["models"][tag]["arms"][kind]["empirical_understatement_of_width_pct"]
        say("[ind] %-10s empirical understatement %+0.4f%% (primary %+0.4f%%)"
            % (kind, emp, stored))
        note("P4 %s %s" % (kind, tag), emp, stored, TOL_PCT[tag])

    # ---- C. the deposited-style scalar estimators on the same streams
    C_sc = corr3(SC)
    stored_sc = STUDY["scalar_construction"]["models"][tag]["fixed_draw"]["Corr"]
    say("[C] scalar metrics on the SAME index streams:")
    say("    recomputed: RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f"
        % (C_sc[0, 1], C_sc[0, 2], C_sc[1, 2]))
    say("    primary   : RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f"
        % (stored_sc["corr_RGA_RGE"], stored_sc["corr_RGA_RGR"], stored_sc["corr_RGE_RGR"]))
    note("scalar corr RGE-RGR %s" % tag, C_sc[1, 2], stored_sc["corr_RGE_RGR"], TOL_CORR[tag])

    # ---- D. the six-rung bridge
    rge_cm, rgr_cm = Smean[:, 1], Smean[:, 2]

    def cc(u, v):
        return float(np.corrcoef(u, v)[0, 1])

    bridge = [
        ("scalar RGE (bare)      x scalar RGR (bare)      [deposit pair]",
         cc(SC[:, 1], SC[:, 2])),
        ("scalar RGE (class_ord) x scalar RGR (class_ord) [convention only]",
         cc(SC_co[:, 0], SC_co[:, 1])),
        ("curve-mean RGE         x scalar RGR (class_ord) [RGE estimator swapped]",
         cc(rge_cm, SC_co[:, 1])),
        ("scalar RGE (class_ord) x curve-mean RGR (swap)  [RGR estimator swapped]",
         cc(SC_co[:, 0], rgr_cm)),
        ("curve-mean RGE         x noise-sweep RGR mean   [sweep, same family]",
         cc(rge_cm, Rb_noise_mean)),
        ("curve-mean RGE         x curve-mean RGR (swap)  [study pair]",
         cc(rge_cm, rgr_cm)),
    ]
    stored_bridge = [r["correlation"] for r in
                     STUDY["models"][tag]["estimator_bridge_RGE_RGR"]["rungs"]]
    say("[D] RGE-RGR correlation, bridged one estimator change at a time:")
    for i, (lab, v) in enumerate(bridge):
        say("      %+.4f (primary %+.4f)   %s" % (v, stored_bridge[i], lab))
        note("P6 rung %d %s" % (i + 1, tag), v, stored_bridge[i], TOL_CORR[tag])
    ends = STUDY["models"][tag]["estimator_bridge_RGA_endpoints"]
    say("[D] RGA-RGE: scalars %+.4f -> curve means %+.4f (primary %+.4f -> %+.4f)"
        % (cc(SC[:, 0], SC[:, 1]), cc(Smean[:, 0], rge_cm),
           ends["RGA_RGE_scalars"], ends["RGA_RGE_curve_means"]))
    say("[D] RGA-RGR: scalars %+.4f -> curve means %+.4f (primary %+.4f -> %+.4f)"
        % (cc(SC[:, 0], SC[:, 2]), cc(Smean[:, 0], rgr_cm),
           ends["RGA_RGR_scalars"], ends["RGA_RGR_curve_means"]))

    # ---- E. the deterministic knots cannot be the mechanism
    det_dropped = np.column_stack([Ab[:, :-1].mean(1), Eb[:, 1:].mean(1), Rb[:, 1:].mean(1)])
    Cd = corr3(det_dropped)
    say("[E] max |corr(all %d knots) - corr(deterministic knot deleted)| = %.2e"
        % (L, np.max(np.abs(Cd - Cor))))
    Sd = np.cov(det_dropped, rowvar=False)
    vm_d = float(g @ Sd @ g); vd_d = float(np.sum(g ** 2 * np.diag(Sd)))
    say("[E] arithmetic cross share, deterministic knots deleted: %+0.4f%% "
        "(all knots: %+0.4f%%)" % (100.0 * (1.0 - vd_d / vm_d), share))
    say("[E] replicate sd of the deterministic knots: RGA[-1] %.2e, RGE[0] %.2e, "
        "RGR[0] %.2e" % (Ab[:, -1].std(), Eb[:, 0].std(), Rb[:, 0].std()))

    # ---- G. per-severity-index correlations
    per = []
    for t in range(L):
        u, v = Eb[:, t], Rb[:, t]
        per.append(None if (u.std() <= 1e-12 or v.std() <= 1e-12)
                   else float(np.corrcoef(u, v)[0, 1]))
    stored_per = STUDY["models"][tag]["curve_summaries"]["per_severity_index_correlation"]["RGE_RGR"]
    gap_per = max(abs(a - b_) for a, b_ in zip(per, stored_per)
                  if a is not None and b_ is not None)
    say("[G] per-severity-index RGE-RGR, max abs gap vs the primary: %.2e" % gap_per)
    note("per-index RGE-RGR %s" % tag, gap_per, 0.0, TOL_CORR[tag])
    say("[G] recomputed: %s" % " ".join("na" if v is None else "%+.3f" % v for v in per))

    # ---- diagonal identity spot-check
    diag_arith = ((Ab + Eb + Rb) / 3.0).mean(1)
    prod_arith = (Ab.mean(1) + Eb.mean(1) + Rb.mean(1)) / 3.0
    say("[diag] arithmetic diagonal == product space at every replicate: max |diff| = %.2e"
        % np.max(np.abs(diag_arith - prod_arith)))

    KEEP[tag] = {"p_full": p_full, "p_masked_single": p_masked_single, "p_pert": p_pert,
                 "col_std": col_std, "SC": SC}

# ------------------------------------------------------------------ F. the chain
say("")
say("=" * 78)
say("F. the two-link chain, shared index stream, both conditioning schemes")
say("=" * 78)
t0 = time.time()
p_pert_redraw = {}
for tag in TAGS:
    model = models[tag]
    mat = np.empty((B, n))
    base = REDRAW_BASE[tag]
    col_std = KEEP[tag]["col_std"]
    for b in range(B):
        rb = np.random.default_rng(base + b)
        pert_b = rb.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
        mat[b] = model.predict_proba(X_te + pert_b)[:, 1]
    p_pert_redraw[tag] = mat
    say("[redraw] %s: %d fresh draws in %.1f s cumulative" % (tag, B, time.time() - t0))

# scheme-B per-model correlations
for tag in TAGS:
    SC = KEEP[tag]["SC"]
    p_full = KEEP[tag]["p_full"]
    pb_mat = np.empty((B, 3))
    for b in range(B):
        ib = IDX[b]
        pb_mat[b] = [SC[b, 0], SC[b, 1],
                     rgr_score(p_full[ib], p_pert_redraw[tag][b][ib])]
    Cr = corr3(pb_mat)
    st = STUDY["scalar_construction"]["models"][tag]["redraw"]["Corr"]
    say("[F] %s redraw scalar corr: RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f "
        "(primary %+.4f %+.4f %+.4f)"
        % (tag, Cr[0, 1], Cr[0, 2], Cr[1, 2], st["corr_RGA_RGE"], st["corr_RGA_RGR"],
           st["corr_RGE_RGR"]))
    note("redraw scalar corr RGE-RGR %s" % tag, Cr[1, 2], st["corr_RGE_RGR"], TOL_CORR[tag])

shared_rng = np.random.default_rng(SEED + 99)
CA_fix = np.empty(B); CB_fix = np.empty(B)
CA_red = np.empty(B); CB_red = np.empty(B)
for b in range(B):
    idx = shared_rng.integers(0, n, n)
    for tag, af, ar in (("logit", CA_fix, CA_red), ("rf", CB_fix, CB_red)):
        p_full = KEEP[tag]["p_full"]
        pb = p_full[idx]
        vec = np.array([rga_score(y_te[idx], pb),
                        float(np.mean([rge_score(pb, KEEP[tag]["p_masked_single"][j][idx])
                                       for j in range(d)])),
                        rgr_score(pb, KEEP[tag]["p_pert"][idx])])
        af[b] = geo_mean(vec)
        ar[b] = geo_mean([vec[0], vec[1],
                          rgr_score(pb, p_pert_redraw[tag][b][idx])])
    if (b + 1) % 500 == 0:
        say("  [chain] %d/%d, %.0f s" % (b + 1, B, time.time() - t0))

point = {}
for tag in TAGS:
    p_full = KEEP[tag]["p_full"]
    point[tag] = geo_mean(np.array([
        rga_score(y_te, p_full),
        float(np.mean([rge_score(p_full, KEEP[tag]["p_masked_single"][j])
                       for j in range(d)])),
        rgr_score(p_full, KEEP[tag]["p_pert"])]))

for scheme, CA, CB in (("fixed_draw", CA_fix, CB_fix), ("redraw", CA_red, CB_red)):
    S = np.cov(np.vstack([CA, CB]))
    rho = float(np.corrcoef(CA, CB)[0, 1])
    h = point["logit"] * point["rf"]
    grad = np.array([point["rf"], point["logit"]])
    vm = float(grad @ S @ grad)
    vd = float(grad[0] ** 2 * S[0, 0] + grad[1] ** 2 * S[1, 1])
    w_m, w_d = 2 * Z * math.sqrt(vm), 2 * Z * math.sqrt(vd)
    und = 100.0 * (1.0 - w_d / w_m)
    st = STUDY["chain"][scheme]
    say("[F] %-10s rho %+.4f (primary %+.4f)  h %.6f (primary %.6f)  "
        "understatement %+0.4f%% (primary %+0.4f%%)"
        % (scheme, rho, st["cross_link_correlation"], h, st["h_point"], und,
           st["understatement_of_width_pct"]))
    note("P5 rho %s" % scheme, rho, st["cross_link_correlation"], max(TOL_CORR.values()))
    note("P5 understatement %s" % scheme, und, st["understatement_of_width_pct"],
         max(TOL_PCT.values()))

# ------------------------------------------------------------------ verdict
say("")
say("=" * 78)
say("verification summary")
say("=" * 78)
bad = [dd for dd in DIFFS if not dd["ok"]]
say("  %d of %d recomputed quantities agree with the primary run inside tolerance"
    % (len(DIFFS) - len(bad), len(DIFFS)))
for dd in bad:
    say("  MISMATCH %-34s recomputed %.10f  primary %.10f  gap %.3e (tol %.1e)"
        % (dd["label"], dd["recomputed"], dd["stored"], dd["abs_gap"], dd["tol"]))
say("  worst gap over all checks: %.3e" % max(dd["abs_gap"] for dd in DIFFS))
say("[done] %.1f s" % (time.time() - t_start))

with open(os.path.join(HERE, "verify-second-dataset.json"), "w") as fh:
    json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "primary_run": STUDY["generated_utc"], "B": B, "checks": DIFFS,
               "all_agree": bool(not bad),
               "worst_abs_gap": max(dd["abs_gap"] for dd in DIFFS)}, fh, indent=1)
with open(os.path.join(HERE, "verify-second-dataset.log"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
