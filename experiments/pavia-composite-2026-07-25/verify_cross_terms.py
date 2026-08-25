#!/usr/bin/env python3
"""
Adversarial verification of the cross-term sign finding of the Pavia composite
study, written and run independently of pavia_composite_experiment.py.

The study reports width understatements that are NEGATIVE for most arms (the
zero-cross-term interval is WIDER than the measured-covariance interval), the
opposite sign to the scalar-composite study and to the deposited article. This
script decides among three worlds -- (1) sign-convention or construction bug,
(2) real decorrelation of the curve summaries, (3) construction artifact -- by:

  A. re-verifying the factorisation of the triple sum by brute force;
  B. regenerating the paired replicates from the documented recipe (identical
     index streams: default_rng(SEED + 1), one row resample per replicate) and
     recomputing the covariance of the three curve summaries from scratch;
  C. recomputing the DEPOSITED scalar estimators (RGA scalar, RGE as the mean
     over single-feature masks, RGR as one fixed Gaussian draw) on the SAME
     index streams -- the deposit uses the same recipe, so its Corr_paired is
     directly commensurable -- and reproducing the deposit's correlations;
  D. bridging the two constructions estimator by estimator, to isolate which
     change moves the RGE-RGR correlation from +0.60/+0.34 to -0.14/-0.31;
  E. checking analytically and numerically that the deterministic zero-severity
     anchors CANNOT be the mechanism: a curve mean is an affine function of its
     random knots, so deleting the deterministic knot leaves every correlation
     and every cross-covariance share exactly unchanged.

Everything here is recomputed; nothing is copied from the study's results file
except for the comparison targets, which are read back and diffed.

Anton Sokolov, Tyche Institute, Tallinn. 25 July 2026.
"""
import json
import math
import os
import sys
import time

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
B = 2000
Z = 1.959963984540054
CLASS_ORDER = np.array([0, 1])
NOISE_SD = 0.5

with open(os.path.join(HERE, "results-pavia-composite.json")) as fh:
    STUDY = json.load(fh)
with open(os.path.join(os.path.dirname(HERE), "results.json")) as fh:
    DEPOSIT = json.load(fh)


def say(msg=""):
    print(msg, flush=True)


def corr3(M):
    return np.corrcoef(M, rowvar=False)


def tail_swap(X, p):
    """Independent rewrite of the article's eq.-(9) perturbation: for each column,
    the lowest floor(p*n) ranks exchange values with the highest, pairwise from
    the outside in."""
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


# ------------------------------------------------------------------ A. factorisation
say("=" * 78)
say("A. factorisation of the triple sum, re-verified by brute force")
say("=" * 78)
rng_chk = np.random.default_rng(12345)
worst = {"arith": 0.0, "geom": 0.0, "rms_gap_from_candidates": None}
cases = [(rng_chk.uniform(0.01, 1.0, 21), rng_chk.uniform(0.01, 1.0, 21),
          rng_chk.uniform(0.01, 1.0, 21)) for _ in range(3)]
for tag in ("logit", "rf"):
    cp = STUDY["models"][tag]["curves_point"]
    cases.append((np.array(cp["RGA_partial"]), np.array(cp["RGE_greedy_mean_baseline"]),
                  np.array(cp["RGR_tail_swap"])))
for a, e, r in cases:
    L = len(a)
    s_ar = s_ge = s_rm = 0.0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                s_ar += (a[i] + e[j] + r[k]) / 3.0
                s_ge += (a[i] * e[j] * r[k]) ** (1.0 / 3.0)
                s_rm += math.sqrt((a[i] ** 2 + e[j] ** 2 + r[k] ** 2) / 3.0)
    s_ar /= L ** 3; s_ge /= L ** 3; s_rm /= L ** 3
    cf_ar = (a.mean() + e.mean() + r.mean()) / 3.0
    cf_ge = np.cbrt(a).mean() * np.cbrt(e).mean() * np.cbrt(r).mean()
    worst["arith"] = max(worst["arith"], abs(s_ar - cf_ar))
    worst["geom"] = max(worst["geom"], abs(s_ge - cf_ge))
    # two natural pseudo-closed-forms for the rms variant, to show neither is it
    cand1 = math.sqrt((a.mean() ** 2 + e.mean() ** 2 + r.mean() ** 2) / 3.0)
    cand2 = (np.sqrt(a ** 2 / 3).mean() + np.sqrt(e ** 2 / 3).mean()
             + np.sqrt(r ** 2 / 3).mean())
    g1, g2 = abs(s_rm - cand1), abs(s_rm - cand2)
    if worst["rms_gap_from_candidates"] is None or min(g1, g2) < worst["rms_gap_from_candidates"]:
        worst["rms_gap_from_candidates"] = min(g1, g2)
say("  arithmetic closed form : max |brute - closed| = %.3e  (factorises)" % worst["arith"])
say("  geometric  closed form : max |brute - closed| = %.3e  (factorises)" % worst["geom"])
say("  rms: nearest natural candidate closed form is off by at least %.3e"
    % worst["rms_gap_from_candidates"])
assert worst["arith"] < 1e-13 and worst["geom"] < 1e-13
assert worst["rms_gap_from_candidates"] > 1e-4

# ------------------------------------------------------------------ data and models
ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
X = ds.data.astype(float)
y = (ds.target == "good").astype(int)
names = list(ds.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
n, d = X_te.shape
GRID, L = d, d + 1
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

# the deposit's module-level rng, consumed logit first then rf, as in its main()
dep_rng = np.random.default_rng(SEED)

REPORT = {}
for tag in ("logit", "rf"):
    say("")
    say("=" * 78)
    say("%s" % tag)
    say("=" * 78)
    model = models[tag]
    p_full = model.predict_proba(X_te)[:, 1]
    col_mean = X_te.mean(axis=0)
    col_std = X_te.std(axis=0, keepdims=True)

    # ---- point curves, from the package, independently of the study script
    a0 = np.asarray(rga_curve(y_te, p_full, curve_method="partial", n_segments=GRID,
                              normalize_to_perfect=False)["curve"], dtype=float)
    res = rge_curve(model, X_te, method="tabular", feature_names=names,
                    masking_method="greedy", baseline="mean", n_steps=GRID,
                    verbose=False)
    e0 = np.asarray(res["rge_scores"], dtype=float)
    order = [names.index(nm) for nm in res["removed_features"]]
    ps = 0.5 * np.arange(L) / GRID
    r0 = np.array([rgr_score(p_full,
                             model.predict_proba(tail_swap(X_te, ps[t]))[:, 1],
                             class_order=CLASS_ORDER) for t in range(L)])
    cp = STUDY["models"][tag]["curves_point"]
    gaps = (np.max(np.abs(a0 - np.array(cp["RGA_partial"]))),
            np.max(np.abs(e0 - np.array(cp["RGE_greedy_mean_baseline"]))),
            np.max(np.abs(r0 - np.array(cp["RGR_tail_swap"]))))
    say("[point] max abs gap vs study curves: RGA %.2e  RGE %.2e  RGR %.2e"
        % gaps)
    assert max(gaps) < 1e-12

    # ---- precompute prediction vectors (study construction)
    p_mask = np.empty((L, n)); p_mask[0] = p_full
    for t in range(1, L):
        Xm = X_te.copy()
        Xm[:, order[:t]] = col_mean[order[:t]]
        p_mask[t] = model.predict_proba(Xm)[:, 1]
    p_swap = np.empty((L, n))
    for t in range(L):
        p_swap[t] = model.predict_proba(tail_swap(X_te, ps[t]))[:, 1]
    rng_noise = np.random.default_rng(SEED)     # per model, as in the study
    p_noise = np.empty((L, n))
    for t in range(L):
        sig = t / float(GRID)
        Xp = X_te + rng_noise.normal(0.0, sig, size=X_te.shape) * col_std if sig > 0 else X_te
        p_noise[t] = model.predict_proba(Xp)[:, 1]

    # ---- precompute prediction vectors (deposit construction)
    p_masked_single = np.empty((d, n))
    for j in range(d):
        Xm = X_te.copy()
        Xm[:, j] = col_mean[j]
        p_masked_single[j] = model.predict_proba(Xm)[:, 1]
    pert = dep_rng.normal(0.0, NOISE_SD, size=X_te.shape) * X_te.std(axis=0, keepdims=True)
    p_pert = model.predict_proba(X_te + pert)[:, 1]

    dep_point = DEPOSIT["models"][tag]["point"]
    rga_s0 = float(rga_score(y_te, p_full))
    rge_s0 = float(np.mean([rge_score(p_full, p_masked_single[j]) for j in range(d)]))
    rgr_s0 = float(rgr_score(p_full, p_pert))
    say("[deposit point] recomputed RGA/RGE/RGR = %.10f / %.10f / %.10f" %
        (rga_s0, rge_s0, rgr_s0))
    say("[deposit point] stored     RGA/RGE/RGR = %.10f / %.10f / %.10f  "
        "(max |gap| %.2e)" % (dep_point["RGA"], dep_point["RGE"], dep_point["RGR"],
                              max(abs(rga_s0 - dep_point["RGA"]),
                                  abs(rge_s0 - dep_point["RGE"]),
                                  abs(rgr_s0 - dep_point["RGR"]))))

    # ---- paired pass: curves AND scalars on the same index streams
    t0 = time.time()
    Ab = np.empty((B, L)); Eb = np.empty((B, L)); Rb = np.empty((B, L))
    Rb_noise_mean = np.empty(B)
    SC = np.empty((B, 3))       # deposited scalar estimators, bare-vector convention
    SC_co = np.empty((B, 2))    # same RGE/RGR scalars, probability-matrix convention
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
    say("[paired] regenerated B = %d replicates in %.1f s" % (B, time.time() - t0))

    # ---- B. covariance of the three curve summaries, from scratch
    Smean = np.column_stack([Ab.mean(1), Eb.mean(1), Rb.mean(1)])
    Scbrt = np.column_stack([np.cbrt(Ab).mean(1), np.cbrt(Eb).mean(1),
                             np.cbrt(Rb).mean(1)])
    Sig = np.cov(Smean, rowvar=False)
    Cor = corr3(Smean)
    stored = np.array(STUDY["models"][tag]["arms"]["arithmetic"]["delta_method"]
                      ["Sigma_of_summaries"])
    say("[B] Sigma of curve means, recomputed:")
    for row in Sig:
        say("      [% .6e % .6e % .6e]" % tuple(row))
    say("[B] max |Sigma_recomputed - Sigma_stored| = %.2e" % np.max(np.abs(Sig - stored)))
    say("[B] corr(curve means): RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f"
        % (Cor[0, 1], Cor[0, 2], Cor[1, 2]))

    # delta arithmetic, by hand, for the arithmetic arm
    g = np.full(3, 1.0 / 3.0)
    var_meas = float(g @ Sig @ g)
    var_decl = float(np.sum(g ** 2 * np.diag(Sig)))
    under = 100.0 * (1.0 - math.sqrt(var_decl) / math.sqrt(var_meas))
    share = 100.0 * (1.0 - var_decl / var_meas)
    st = STUDY["models"][tag]["arms"]["arithmetic"]["delta_method"]
    say("[B] arithmetic arm by hand: var_meas %.6e (stored %.6e)  var_decl %.6e "
        "(stored %.6e)" % (var_meas, st["var_measured_covariance"], var_decl,
                           st["var_cross_terms_declared_zero"]))
    say("[B] arithmetic under%% = %+0.3f (stored %+0.3f), cross share = %+0.3f "
        "(stored %+0.3f)" % (under, st["understatement_of_width_pct"], share,
                             st["cross_covariance_share_of_variance_pct"]))
    assert abs(under - st["understatement_of_width_pct"]) < 0.02

    # geometric arm from scratch
    Sg = np.cov(Scbrt, rowvar=False)
    s0 = np.array([np.cbrt(a0).mean(), np.cbrt(e0).mean(), np.cbrt(r0).mean()])
    gg = np.array([s0[1] * s0[2], s0[0] * s0[2], s0[0] * s0[1]])
    vm = float(gg @ Sg @ gg); vd = float(np.sum(gg ** 2 * np.diag(Sg)))
    stg = STUDY["models"][tag]["arms"]["geometric"]["delta_method"]
    say("[B] geometric under%% = %+0.3f (stored %+0.3f)"
        % (100.0 * (1.0 - math.sqrt(vd) / math.sqrt(vm)),
           stg["understatement_of_width_pct"]))

    # rms arm from scratch: entry-level gradient projections
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
    strms = STUDY["models"][tag]["arms"]["rms"]["delta_method"]
    say("[B] rms under%% = %+0.3f (stored %+0.3f); corr of projections "
        "RGA-RGE %+.4f RGA-RGR %+.4f RGE-RGR %+.4f"
        % (100.0 * (1.0 - math.sqrt(vd) / math.sqrt(vm)),
           strms["understatement_of_width_pct"],
           corr3(P)[0, 1], corr3(P)[0, 2], corr3(P)[1, 2]))

    # independent-stream pass, curves only, for the empirical column
    t0 = time.time()
    Vi = {"arithmetic": np.empty(B), "geometric": np.empty(B)}
    Ai = np.empty((B, L)); Ei = np.empty((B, L)); Ri = np.empty((B, L))
    for b in range(B):
        ia, ie, ir = IND[0, b], IND[1, b], IND[2, b]
        Ai[b] = rga_curve(y_te[ia], p_full[ia], curve_method="partial",
                          n_segments=GRID, normalize_to_perfect=False)["curve"]
        Ei[b, 0] = 1.0
        pe = p_full[ie]
        for t in range(1, L):
            Ei[b, t] = rge_score(pe, p_mask[t][ie], class_order=CLASS_ORDER)
        pr = p_full[ir]
        for t in range(L):
            Ri[b, t] = rgr_score(pr, p_swap[t][ir], class_order=CLASS_ORDER)
    say("[ind] regenerated independent streams in %.1f s" % (time.time() - t0))
    for kind, fn in (("arithmetic", lambda A, E, R: (A.mean(1) + E.mean(1) + R.mean(1)) / 3.0),
                     ("geometric", lambda A, E, R: np.cbrt(A).mean(1) * np.cbrt(E).mean(1) * np.cbrt(R).mean(1))):
        Vb_, Vi_ = fn(Ab, Eb, Rb), fn(Ai, Ei, Ri)
        wb = np.percentile(Vb_, 97.5) - np.percentile(Vb_, 2.5)
        wi = np.percentile(Vi_, 97.5) - np.percentile(Vi_, 2.5)
        emp = 100.0 * (1.0 - wi / wb)
        say("[ind] %-10s empirical understatement %+0.2f%% (stored %+0.2f%%)"
            % (kind, emp,
               STUDY["models"][tag]["arms"][kind]["empirical_understatement_of_width_pct"]))

    # ---- C. the deposited scalar estimators on the same streams
    C_sc = corr3(SC)
    dep_corr = np.array(DEPOSIT["models"][tag]["Corr_paired"])
    say("[C] scalar metrics on the SAME index streams:")
    say("    recomputed corr: RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f"
        % (C_sc[0, 1], C_sc[0, 2], C_sc[1, 2]))
    say("    deposit    corr: RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f  "
        "(max |gap| %.2e)"
        % (dep_corr[0, 1], dep_corr[0, 2], dep_corr[1, 2],
           np.max(np.abs(C_sc - dep_corr))))

    # ---- D. the estimator-by-estimator bridge for the RGE-RGR pair
    rge_curvemean = Smean[:, 1]
    rgr_curvemean = Smean[:, 2]
    rge_scalar, rgr_scalar = SC[:, 1], SC[:, 2]
    rge_scalar_co, rgr_scalar_co = SC_co[:, 0], SC_co[:, 1]
    def c(u, v):
        return float(np.corrcoef(u, v)[0, 1])
    bridge = [
        ("scalar RGE (bare)      x scalar RGR (bare)      [deposit pair]",
         c(rge_scalar, rgr_scalar)),
        ("scalar RGE (class_ord) x scalar RGR (class_ord) [convention only]",
         c(rge_scalar_co, rgr_scalar_co)),
        ("curve-mean RGE         x scalar RGR (class_ord) [RGE estimator swapped]",
         c(rge_curvemean, rgr_scalar_co)),
        ("scalar RGE (class_ord) x curve-mean RGR (swap)  [RGR estimator swapped]",
         c(rge_scalar_co, rgr_curvemean)),
        ("curve-mean RGE         x noise-sweep RGR mean   [sweep, same family]",
         c(rge_curvemean, Rb_noise_mean)),
        ("curve-mean RGE         x curve-mean RGR (swap)  [study pair]",
         c(rge_curvemean, rgr_curvemean)),
    ]
    say("[D] RGE-RGR correlation, bridged one estimator change at a time:")
    for lab, v in bridge:
        say("      %+.4f   %s" % (v, lab))
    say("[D] same bridge for RGA-RGE and RGA-RGR endpoints:")
    say("      RGA-RGE: scalars %+.4f -> curve means %+.4f"
        % (c(SC[:, 0], rge_scalar), c(Smean[:, 0], rge_curvemean)))
    say("      RGA-RGR: scalars %+.4f -> curve means %+.4f"
        % (c(SC[:, 0], rgr_scalar), c(Smean[:, 0], rgr_curvemean)))

    # ---- E. the anchors cannot be the mechanism
    # each curve has exactly one deterministic knot (RGA: terminal 0; RGE and RGR:
    # anchor 1), so the L-knot mean is an affine function of the 20 random knots
    # and deleting the deterministic knot changes no correlation and no share.
    det_dropped = np.column_stack([Ab[:, :-1].mean(1), Eb[:, 1:].mean(1),
                                   Rb[:, 1:].mean(1)])
    Cd = corr3(det_dropped)
    say("[E] max |corr(all 21 knots) - corr(deterministic knot deleted)| = %.2e"
        % np.max(np.abs(Cd - Cor)))
    Sd = np.cov(det_dropped, rowvar=False)
    vm_d = float(g @ Sd @ g); vd_d = float(np.sum(g ** 2 * np.diag(Sd)))
    say("[E] arithmetic cross share, deterministic knots deleted: %+0.3f%% "
        "(all knots: %+0.3f%%)"
        % (100.0 * (1.0 - vd_d / vm_d), share))
    say("[E] replicate sd of the deterministic knots: RGA[-1] %.2e, RGE[0] %.2e, "
        "RGR[0] %.2e" % (Ab[:, -1].std(), Eb[:, 0].std(), Rb[:, 0].std()))

    # ---- diagonal identity spot-check (section 7 of the study)
    diag_arith = ((Ab + Eb + Rb) / 3.0).mean(1)
    prod_arith = (Ab.mean(1) + Eb.mean(1) + Rb.mean(1)) / 3.0
    say("[diag] arithmetic diagonal == product space at every replicate: "
        "max |diff| = %.2e" % np.max(np.abs(diag_arith - prod_arith)))
    d_geom0 = float(np.cbrt(a0 * e0 * r0).mean()
                    - np.cbrt(a0).mean() * np.cbrt(e0).mean() * np.cbrt(r0).mean())
    say("[diag] geometric diagonal - product at the point: %+.6f (stored %+.6f)"
        % (d_geom0,
           STUDY["models"][tag]["product_space_vs_diagonal"]["geometric"]
           ["diagonal_minus_product_space"]))

    REPORT[tag] = {
        "corr_curve_means": [Cor[0, 1], Cor[0, 2], Cor[1, 2]],
        "corr_scalars_same_streams": [C_sc[0, 1], C_sc[0, 2], C_sc[1, 2]],
        "bridge_RGE_RGR": {lab: v for lab, v in bridge},
    }

say("")
say("[verdict] see the verification log and the study summary, section 5a")
