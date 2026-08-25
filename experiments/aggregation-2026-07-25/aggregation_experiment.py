#!/usr/bin/env python3
"""
Aggregation study for the composed rank-graduation (SAFE) score, 2026-07-25.

Question. The composed interval of the deposited experiment is built for ONE
aggregator (the geometric mean). Section 5 of the outline promises that the
construction is aggregator-agnostic: a weighted sum, a TOPSIS composite and a
Shapley-based composite each get an interval BY THE SAME CONSTRUCTION, and the
intervals differ. This study builds that layer for a family of composites, so
that whichever composite the Pavia integrated-metrics line actually uses, only
the composite function has to be swapped.

Construction (identical for every composite, which is the point).
  1. Estimate the metric vector theta = (RGA, RGE, RGR) on ONE shared test
     sample, with a fixed trained model.
  2. Resample test rows once per replicate and recompute the WHOLE vector on
     that same resample (paired bootstrap). The empirical covariance of the B
     vectors estimates Sigma/n.
  3. Push each replicate vector through the composite -> percentile interval.
  4. Linearise the composite at the point estimate -> delta interval, with
     var = grad' Sigma grad, and the same quantity with the cross-terms
     DECLARED ZERO, var0 = sum_k grad_k^2 Sigma_kk. The gap between the two
     is the throughline of the article and is reported per composite.

Composites (arms):
  wsum_equal        w = (1/3, 1/3, 1/3)
  wsum_accuracy_led w = (0.50, 0.20, 0.30)
  wsum_accuracy_light w = (0.20, 0.40, 0.40)
  geomean           (RGA*RGE*RGR)^(1/3)  -- the deposited composite, kept as the
                    continuity case; this study must reproduce it exactly
  topsis            ideal point fixed at 1, anti-ideal at 0 (the metrics are
                    already normalised to [0,1]), Euclidean distance, closeness
                    coefficient cc = d_minus / (d_minus + d_plus). A DATA-
                    DEPENDENT ideal point would make the composite depend on the
                    other alternatives and would need the full joint covariance
                    ACROSS models, which is out of scope here.
  shapley_min       characteristic function v(S) = min_{k in S} theta_k,
                    v(empty) = 0; Shapley value of each metric; composite = sum
                    of the Shapley values. This is ONE defensible instantiation,
                    not a canonical one. The Shapley-Lorenz line of the Pavia
                    group is a DIFFERENT construction and this is a placeholder
                    until their composite is known.
  shapley_prod      same machinery with v(S) = prod_{k in S} theta_k, as a
                    SMOOTH counterpart to shapley_min.

Substrate. safeai (github.com/koleso500/safeai) at pinned commit 39768fc, the
vendored clone and import stubs of the 2026-07-24 redraw run, reused read-only.
This script never clones, never writes into the substrate, and never touches the
deposited experiment directories.

Outputs: results-aggregation.json (all numbers), run-aggregation.log.
"""
import itertools
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REDRAW_DIR = os.path.join(os.path.dirname(HERE), "redraw-2026-07-24")
SAFEAI_REPO = os.path.join(REDRAW_DIR, "safeai-src")
STUBS = os.path.join(REDRAW_DIR, "_stubs")
CACHE = os.path.join(HERE, ".cache")
DEPOSIT_FIXED = os.path.join(os.path.dirname(HERE), "results.json")
DEPOSIT_REDRAW = os.path.join(REDRAW_DIR, "results-redraw.json")

for _p, _what in ((SAFEAI_REPO, "vendored safeai clone"), (STUBS, "import stubs")):
    if not os.path.isdir(_p):
        raise SystemExit(
            "missing %s: this study reuses the pinned substrate of the redraw run "
            "and neither clones nor writes one." % _what)

sys.path.insert(0, STUBS)
sys.path.insert(0, SAFEAI_REPO)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from safeai.rga import rga_score
from safeai.rge import rge_score
from safeai.rgr import rgr_score

# ---- constants, all inherited from the deposited runs so the numbers are commensurable
SEED = 20260723
B = int(os.environ.get("AGG_B", "2000"))
ALPHA = 0.05
NOISE_SD = 0.5
Z = 1.959963984540054                      # two-sided 95%, as hardcoded in both deposited runs
REDRAW_BASE = {"logit": SEED + 100_000, "rf": SEED + 200_000}
METRICS = ["RGA", "RGE", "RGR"]
TAGS = ["logit", "rf"]
QLO, QHI = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)

W_EQUAL = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
W_ACC_LED = np.array([0.50, 0.20, 0.30])
W_ACC_LIGHT = np.array([0.20, 0.40, 0.40])

LOG_LINES = []


def say(msg):
    print(msg, flush=True)
    LOG_LINES.append(msg)


# ---------------------------------------------------------------- data / models
def load_data():
    ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "good").astype(int)
    return X, y, "statlog-german-credit (OpenML credit-g v1)"


def metric_vector(y, p_full, p_masked, p_pert, idx):
    """(RGA, RGE, RGR) on the row subset idx -- the shared-sample functional vector."""
    yb, pf = y[idx], p_full[idx]
    rga = rga_score(yb, pf)
    rge = float(np.mean([rge_score(pf, p_masked[j][idx]) for j in range(p_masked.shape[0])]))
    rgr = rgr_score(pf, p_pert[idx])
    return np.array([rga, rge, rgr])


# ---------------------------------------------------------------- composites
SUBSETS = [S for r in range(4) for S in itertools.combinations(range(3), r)]


def _game_min(M, S):
    if not S:
        return np.zeros(M.shape[0])
    return M[:, list(S)].min(axis=1)


def _game_prod(M, S):
    if not S:
        return np.zeros(M.shape[0])
    return M[:, list(S)].prod(axis=1)


def shapley_matrix(M, game):
    """Exact Shapley values of a 3-player game, evaluated row-wise on M (B, 3)."""
    K = M.shape[1]
    cache = {S: game(M, S) for S in SUBSETS}
    phi = np.zeros_like(M, dtype=float)
    for k in range(K):
        others = [j for j in range(K) if j != k]
        for r in range(K):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
                phi[:, k] += w * (cache[tuple(sorted(S + (k,)))] - cache[S])
    return phi


def c_wsum(M, w):
    return M @ w


def c_geomean(M):
    return np.exp(np.mean(np.log(np.clip(M, 1e-12, None)), axis=1))


def c_topsis(M):
    d_plus = np.sqrt(((M - 1.0) ** 2).sum(axis=1))
    d_minus = np.sqrt((M ** 2).sum(axis=1))
    return d_minus / (d_minus + d_plus)


def c_shapley_min(M):
    return shapley_matrix(M, _game_min).sum(axis=1)


def c_shapley_prod(M):
    return shapley_matrix(M, _game_prod).sum(axis=1)


# ---- analytic gradients at a point vector v (shape (3,))
def g_wsum(v, w):
    return np.asarray(w, dtype=float).copy()


def g_geomean(v):
    C = float(c_geomean(v[None, :])[0])
    return C / (3.0 * v)                       # dC/dtheta_k, as in the deposited run


def g_topsis(v):
    d_plus = float(np.sqrt(((v - 1.0) ** 2).sum()))
    d_minus = float(np.sqrt((v ** 2).sum()))
    return (d_plus * (v / d_minus) - d_minus * ((v - 1.0) / d_plus)) / (d_minus + d_plus) ** 2


def g_shapley_min(v):
    g = np.zeros_like(v)
    g[int(np.argmin(v))] = 1.0                 # sub-gradient; undefined at a tie
    return g


def g_shapley_prod(v):
    return float(v.prod()) / v


def numerical_grad(fn, v, h=1e-6):
    g = np.zeros_like(v)
    for k in range(len(v)):
        vp, vm = v.copy(), v.copy()
        vp[k] += h
        vm[k] -= h
        g[k] = (float(fn(vp[None, :])[0]) - float(fn(vm[None, :])[0])) / (2.0 * h)
    return g


ARMS = [
    ("wsum_equal", lambda M: c_wsum(M, W_EQUAL), lambda v: g_wsum(v, W_EQUAL),
     "weighted sum, equal weights w = (1/3, 1/3, 1/3); linear, delta method exact"),
    ("wsum_accuracy_led", lambda M: c_wsum(M, W_ACC_LED), lambda v: g_wsum(v, W_ACC_LED),
     "weighted sum, w = (0.50, 0.20, 0.30); a declared unequal profile, not a recommendation"),
    ("wsum_accuracy_light", lambda M: c_wsum(M, W_ACC_LIGHT), lambda v: g_wsum(v, W_ACC_LIGHT),
     "weighted sum, w = (0.20, 0.40, 0.40); downweights the variance-dominant component"),
    ("geomean", c_geomean, g_geomean,
     "geometric mean (RGA*RGE*RGR)^(1/3); the deposited composite, continuity case"),
    ("topsis", c_topsis, g_topsis,
     "TOPSIS with ideal fixed at 1 and anti-ideal at 0, Euclidean distance, "
     "closeness cc = d_minus/(d_minus + d_plus)"),
    ("shapley_min", c_shapley_min, g_shapley_min,
     "sum of Shapley values of the game v(S) = min_{k in S} theta_k, v(empty) = 0"),
    ("shapley_prod", c_shapley_prod, g_shapley_prod,
     "sum of Shapley values of the game v(S) = prod_{k in S} theta_k, v(empty) = 0"),
]


# ---------------------------------------------------------------- interval blocks
def cross_term_decomposition(g, Sigma):
    """Which PAIR carries the cross-covariance, and how much gradient weight sits on
    each metric. Without this the per-arm cross-term share is a bare number and the
    mechanism behind the ordering across composites cannot be read off."""
    pairs = {"RGA_RGE": (0, 1), "RGA_RGR": (0, 2), "RGE_RGR": (1, 2)}
    contrib = {nm: float(2.0 * g[i] * g[j] * Sigma[i, j]) for nm, (i, j) in pairs.items()}
    cross_total = float(sum(contrib.values()))
    diag = {m: float(g[k] ** 2 * Sigma[k, k]) for k, m in enumerate(METRICS)}
    diag_total = float(sum(diag.values()))
    absg = float(np.abs(g).sum())
    return {
        "gradient_share_of_L1_pct": {m: float(100.0 * abs(g[k]) / absg) if absg > 0 else 0.0
                                     for k, m in enumerate(METRICS)},
        "diagonal_contributions_to_variance": diag,
        "diagonal_share_of_variance_pct": {m: float(100.0 * diag[m] / (diag_total + cross_total))
                                           for m in METRICS} if (diag_total + cross_total) > 0 else {},
        "cross_contributions_to_variance": contrib,
        "cross_total": cross_total,
        "share_of_cross_total_pct": {nm: float(100.0 * v / cross_total) for nm, v in contrib.items()}
        if cross_total != 0 else {nm: None for nm in contrib},
    }


def interval_block(fn, grad_fn, point, Sigma, reps, reps_ind=None):
    """One composite, one covariance, one replicate matrix -> the full interval set."""
    C_point = float(fn(point[None, :])[0])
    g = grad_fn(point)
    var_meas = float(g @ Sigma @ g)
    var_decl = float(np.sum(g ** 2 * np.diag(Sigma)))
    ci_meas = [C_point - Z * math.sqrt(var_meas), C_point + Z * math.sqrt(var_meas)]
    ci_decl = [C_point - Z * math.sqrt(var_decl), C_point + Z * math.sqrt(var_decl)]
    w_meas = ci_meas[1] - ci_meas[0]
    w_decl = ci_decl[1] - ci_decl[0]
    Cb = np.asarray(fn(reps), dtype=float)
    out = {
        "C_point": C_point,
        "gradient_at_point": [float(x) for x in g],
        "var_delta_measured_covariance": var_meas,
        "var_delta_cross_terms_declared_zero": var_decl,
        "cross_covariance_share_of_variance_pct": float(100.0 * (1.0 - var_decl / var_meas))
        if var_meas > 0 else 0.0,
        "ci95_delta_measured_covariance": [float(ci_meas[0]), float(ci_meas[1])],
        "ci95_delta_cross_terms_declared_zero": [float(ci_decl[0]), float(ci_decl[1])],
        "width_delta_measured": float(w_meas),
        "width_delta_declared_zero": float(w_decl),
        "understatement_of_width_pct": float(100.0 * (1.0 - w_decl / w_meas)) if w_meas > 0 else 0.0,
        "ci95_paired_bootstrap_percentile": [float(np.percentile(Cb, QLO)),
                                             float(np.percentile(Cb, QHI))],
        "width_paired_bootstrap": float(np.percentile(Cb, QHI) - np.percentile(Cb, QLO)),
        "sd_paired_bootstrap": float(Cb.std(ddof=1)),
        "sd_delta_measured": float(math.sqrt(var_meas)),
        "mean_paired_bootstrap": float(Cb.mean()),
        "plugin_minus_bootstrap_mean": float(C_point - Cb.mean()),
        "cross_term_decomposition": cross_term_decomposition(g, Sigma),
    }
    if reps_ind is not None:
        Ci = np.asarray(fn(reps_ind), dtype=float)
        lo, hi = float(np.percentile(Ci, QLO)), float(np.percentile(Ci, QHI))
        out["ci95_independent_stream_bootstrap"] = [lo, hi]
        out["width_independent_stream_bootstrap"] = hi - lo
        out["sd_independent_stream_bootstrap"] = float(Ci.std(ddof=1))
        out["empirical_understatement_of_width_pct"] = float(
            100.0 * (1.0 - (hi - lo) / out["width_paired_bootstrap"]))
    return out


# ---------------------------------------------------------------- replicate generation
def shared_precompute(X_te, models):
    """Per-row prediction vectors, the deposited standard-normal perturbation matrix and
    the paired index stream. Cheap (a few seconds) and ALWAYS recomputed: the noise sweep
    of the non-smoothness section needs these, so caching them away would silently reduce
    the study to its smooth half on any rerun."""
    n = X_te.shape[0]
    d = X_te.shape[1]
    col_std = X_te.std(axis=0, keepdims=True)
    col_mean = X_te.mean(axis=0)

    # ONE global stream, consumed in dict order logit-then-rf, exactly as the
    # deposited run does. normal(0, s, size) == s * standard_normal(size) on the
    # same stream state, so we draw the standard normals and scale; that makes the
    # noise level a one-parameter family THROUGH the deposited draw.
    rng_global = np.random.default_rng(SEED)
    pre, Zmat = {}, {}
    t0 = time.time()
    for tag in TAGS:
        model = models[tag]
        p_full = model.predict_proba(X_te)[:, 1]
        p_masked = np.empty((d, n))
        for j in range(d):
            Xm = X_te.copy()
            Xm[:, j] = col_mean[j]
            p_masked[j] = model.predict_proba(Xm)[:, 1]
        Zmat[tag] = rng_global.standard_normal(X_te.shape)
        p_pert = model.predict_proba(X_te + NOISE_SD * Zmat[tag] * col_std)[:, 1]
        pre[tag] = (p_full, p_masked, p_pert)
    say("[precompute] %.1fs" % (time.time() - t0))

    # paired index stream: re-instantiated per model, so both models see the same indices
    IDX = np.empty((B, n), dtype=np.int64)
    boot_rng = np.random.default_rng(SEED + 1)
    for b in range(B):
        IDX[b] = boot_rng.integers(0, n, n)
    return pre, IDX, Zmat, col_std


def build_replicates(X_te, y_te, models, pre, IDX, col_std):
    """Regenerate the deposited replicate matrices. Nothing is persisted by the
    deposited runs, so they are re-derived here from data + seeds; the recipe is
    the deposited one and the continuity check below proves it bit for bit."""
    n = len(y_te)
    reps = {}
    for tag in TAGS:
        t1 = time.time()
        point = metric_vector(y_te, *pre[tag], np.arange(n))
        paired = np.empty((B, 3))
        for b in range(B):
            paired[b] = metric_vector(y_te, *pre[tag], IDX[b])
        say("[%s] fixed-draw paired pass %.1fs" % (tag, time.time() - t1))

        # independent (wrong-way) streams: one index stream per metric
        t2 = time.time()
        ind_rngs = [np.random.default_rng(SEED + 10 + k) for k in range(3)]
        ind = np.empty((B, 3))
        for b in range(B):
            for k in range(3):
                ind[b, k] = metric_vector(y_te, *pre[tag], ind_rngs[k].integers(0, n, n))[k]
        say("[%s] independent-stream pass %.1fs" % (tag, time.time() - t2))

        # redraw scheme: fresh perturbation per replicate; RGA and RGE are
        # deterministic in the resampled rows, so only the RGR column moves
        t3 = time.time()
        base = REDRAW_BASE[tag]
        p_full = pre[tag][0]
        redraw = paired.copy()
        for b in range(B):
            rb = np.random.default_rng(base + b)
            pert_b = rb.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
            p_pert_b = models[tag].predict_proba(X_te + pert_b)[:, 1]
            redraw[b, 2] = rgr_score(p_full[IDX[b]], p_pert_b[IDX[b]])
        say("[%s] redraw pass %.1fs" % (tag, time.time() - t3))

        reps[tag] = {"point": point, "paired": paired, "ind": ind, "redraw": redraw}
    return reps


def cached_replicates(X_te, y_te, models, pre, IDX, col_std):
    """The cache holds the replicate MATRICES only. Everything the non-smoothness
    sweep needs is recomputed on every run, so a cached rerun produces the same JSON
    as a cold one, minus the cost of the redraw passes."""
    os.makedirs(CACHE, exist_ok=True)
    names = ["point", "paired", "ind", "redraw"]
    have = all(os.path.exists(os.path.join(CACHE, "%s_%s_B%d.npy" % (t, nm, B)))
               for t in TAGS for nm in names)
    if have:
        say("[cache] replicate matrices loaded from the local cache")
        return {t: {nm: np.load(os.path.join(CACHE, "%s_%s_B%d.npy" % (t, nm, B)))
                    for nm in names} for t in TAGS}, True
    reps = build_replicates(X_te, y_te, models, pre, IDX, col_std)
    for t in TAGS:
        for nm in names:
            np.save(os.path.join(CACHE, "%s_%s_B%d.npy" % (t, nm, B)), reps[t][nm])
    return reps, False


def replicate_diagnostics(reps):
    """Where the replicate cloud sits, and how often each metric is the minimum. Both
    claims in the non-smoothness section rest on these counts, so they are recorded.

    The two distances are the ones the TOPSIS smoothness argument needs. The closeness
    coefficient d_minus / (d_minus + d_plus) is non-differentiable at BOTH of its
    reference points: at the anti-ideal (0,0,0), where d_minus = ||v|| has its cone
    point, and at the ideal (1,1,1), where d_plus = ||v - 1|| has its own. Quoting the
    distance to the anti-ideal alone would leave the nearer of the two singularities
    unmeasured, and on bounded [0,1] metrics sitting close to 1 it is the ideal that is
    nearer. We record both margins so the smoothness claim is checkable."""
    out = {}
    for tag in TAGS:
        for nm in ["paired", "ind", "redraw"]:
            M = reps[tag][nm]
            am = np.argmin(M, axis=1)
            out["%s/%s" % (tag, nm)] = {
                "min_component_anywhere": float(M.min()),
                "max_component_anywhere": float(M.max()),
                "min_distance_to_ideal_point": float(np.linalg.norm(M - 1.0, axis=1).min()),
                "min_distance_to_anti_ideal_point": float(np.linalg.norm(M, axis=1).min()),
                "argmin_counts": {m: int((am == k).sum()) for k, m in enumerate(METRICS)},
            }
    per = [v for k, v in out.items() if isinstance(v, dict)]
    out["min_component_over_all_matrices"] = float(min(v["min_component_anywhere"] for v in per))
    out["max_component_over_all_matrices"] = float(max(v["max_component_anywhere"] for v in per))
    out["min_distance_to_ideal_over_all_matrices"] = float(
        min(v["min_distance_to_ideal_point"] for v in per))
    out["min_distance_to_anti_ideal_over_all_matrices"] = float(
        min(v["min_distance_to_anti_ideal_point"] for v in per))
    out["argmin_counts_over_all_matrices"] = {
        m: int(sum(v["argmin_counts"][m] for v in per)) for m in METRICS}
    out["n_matrices_with_argmin_RGA_in_every_replicate"] = int(sum(
        1 for v in per if v["argmin_counts"]["RGA"] == sum(v["argmin_counts"].values())))
    return out


def weight_sensitivity(Sigma, step=0.02, floor=0.10):
    """How far can the cross-term diagnostic move if the weights of a weighted sum move?
    Three hand-picked profiles do not answer that. The composite is linear, so the whole
    simplex is a closed-form scan over the SAME covariance: no resampling needed."""
    grid = []
    m = int(round(1.0 / step))
    for i in range(m + 1):
        for j in range(m + 1 - i):
            w = np.array([i * step, j * step, 1.0 - i * step - j * step])
            w = np.clip(w, 0.0, 1.0)
            var_meas = float(w @ Sigma @ w)
            if var_meas <= 0:
                continue
            var_decl = float(np.sum(w ** 2 * np.diag(Sigma)))
            share = 100.0 * (1.0 - var_decl / var_meas)
            und = 100.0 * (1.0 - math.sqrt(var_decl / var_meas))
            grid.append((share, und, w))
    by_share = sorted(grid, key=lambda t: t[0])
    interior = [t for t in grid if (t[2] >= floor - 1e-12).all()]
    by_share_int = sorted(interior, key=lambda t: t[0])

    def rec(t):
        return {"weights": [float(x) for x in t[2]],
                "cross_covariance_share_of_variance_pct": float(t[0]),
                "understatement_of_width_pct": float(t[1])}

    named = {}
    for nm, w in (("wsum_equal", W_EQUAL), ("wsum_accuracy_led", W_ACC_LED),
                  ("wsum_accuracy_light", W_ACC_LIGHT)):
        var_meas = float(w @ Sigma @ w)
        var_decl = float(np.sum(w ** 2 * np.diag(Sigma)))
        named[nm] = {"weights": [float(x) for x in w],
                     "cross_covariance_share_of_variance_pct": float(100.0 * (1.0 - var_decl / var_meas)),
                     "understatement_of_width_pct": float(100.0 * (1.0 - math.sqrt(var_decl / var_meas)))}
    return {
        "grid_step": step,
        "n_weight_vectors": len(grid),
        "min_over_simplex": rec(by_share[0]),
        "max_over_simplex": rec(by_share[-1]),
        "interior_floor": floor,
        "min_over_interior": rec(by_share_int[0]),
        "max_over_interior": rec(by_share_int[-1]),
        "at_declared_profiles": named,
    }


# ---------------------------------------------------------------- non-smoothness sweep
def nonsmoothness_sweep(tag, y_te, X_te, models, pre, IDX, Zmat, col_std):
    """The min-type composite has a kink where the argmin changes. At the deposited
    operating point the argmin is frozen, so we move ALONG a one-parameter family of
    noise levels through that point until the argmin actually crosses, and watch what
    the delta method does. Changing the noise level changes the ESTIMAND: these numbers
    are a methodological demonstration and are not a correction to anything deposited."""
    n = len(y_te)
    p_full, p_masked, _ = pre[tag]
    model = models[tag]

    def level(s):
        p_pert_s = model.predict_proba(X_te + s * Zmat[tag] * col_std)[:, 1]
        pt = metric_vector(y_te, p_full, p_masked, p_pert_s, np.arange(n))
        col = np.empty(B)
        for b in range(B):
            col[b] = rgr_score(p_full[IDX[b]], p_pert_s[IDX[b]])
        return pt, col, p_pert_s

    return level


def sweep_record(s, point, M):
    """Everything we report at one noise level."""
    Sigma = np.cov(M, rowvar=False)
    argmin_pt = int(np.argmin(point))
    amins = np.argmin(M, axis=1)
    counts = [int((amins == k).sum()) for k in range(3)]
    flip = float((amins != argmin_pt).mean())
    # min composite
    C_min_pt = float(point.min())
    g = np.zeros(3)
    g[argmin_pt] = 1.0
    var_meas = float(g @ Sigma @ g)
    var_decl = float(np.sum(g ** 2 * np.diag(Sigma)))
    Cb = M.min(axis=1)
    ci_delta = [C_min_pt - Z * math.sqrt(var_meas), C_min_pt + Z * math.sqrt(var_meas)]
    ci_boot = [float(np.percentile(Cb, QLO)), float(np.percentile(Cb, QHI))]
    # smooth control at the same level
    C_geo_pt = float(c_geomean(point[None, :])[0])
    gg = g_geomean(point)
    var_geo = float(gg @ Sigma @ gg)
    Cg = c_geomean(M)
    ci_geo_delta = [C_geo_pt - Z * math.sqrt(var_geo), C_geo_pt + Z * math.sqrt(var_geo)]
    ci_geo_boot = [float(np.percentile(Cg, QLO)), float(np.percentile(Cg, QHI))]
    return {
        "noise_sd": float(s),
        "point": {m: float(point[k]) for k, m in enumerate(METRICS)},
        "argmin_at_point": METRICS[argmin_pt],
        "argmin_counts_over_replicates": dict(zip(METRICS, counts)),
        "fraction_of_replicates_whose_argmin_differs_from_the_point": flip,
        "fraction_RGR_below_RGA": float((M[:, 2] < M[:, 0]).mean()),
        "min_composite": {
            "C_plugin": C_min_pt,
            "sd_delta": float(math.sqrt(var_meas)),
            "sd_bootstrap": float(Cb.std(ddof=1)),
            "sd_ratio_delta_over_bootstrap": float(math.sqrt(var_meas) / Cb.std(ddof=1)),
            "bootstrap_mean": float(Cb.mean()),
            "plugin_minus_bootstrap_mean": float(C_min_pt - Cb.mean()),
            "ci95_delta": [float(ci_delta[0]), float(ci_delta[1])],
            "ci95_paired_bootstrap": ci_boot,
            "width_delta": float(ci_delta[1] - ci_delta[0]),
            "width_bootstrap": float(ci_boot[1] - ci_boot[0]),
            "cross_covariance_share_of_variance_pct":
                float(100.0 * (1.0 - var_decl / var_meas)) if var_meas > 0 else 0.0,
            "lower_bound_gap_delta_minus_bootstrap": float(ci_delta[0] - ci_boot[0]),
        },
        "geomean_control": {
            "C_plugin": C_geo_pt,
            "sd_delta": float(math.sqrt(var_geo)),
            "sd_bootstrap": float(Cg.std(ddof=1)),
            "sd_ratio_delta_over_bootstrap": float(math.sqrt(var_geo) / Cg.std(ddof=1)),
            "ci95_delta": [float(ci_geo_delta[0]), float(ci_geo_delta[1])],
            "ci95_paired_bootstrap": ci_geo_boot,
            "lower_bound_gap_delta_minus_bootstrap": float(ci_geo_delta[0] - ci_geo_boot[0]),
        },
    }


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    X, y, dataset_name = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    n = len(y_te)
    say("[data] %s: train=%d, test=%d, d=%d, B=%d" % (dataset_name, len(y_tr), n, X.shape[1], B))

    models = {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for tag in TAGS:
        models[tag].fit(X_tr, y_tr)

    pre, IDX, Zmat, col_std = shared_precompute(X_te, models)
    reps, from_cache = cached_replicates(X_te, y_te, models, pre, IDX, col_std)

    results = {
        "study": "aggregation layer for the composed rank-graduation score: one interval "
                 "construction, several composites",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset_name,
        "n_test": int(n),
        "B": B,
        "alpha": ALPHA,
        "noise_sd": NOISE_SD,
        "seed": SEED,
        "z_two_sided_95": Z,
        "safeai_commit": subprocess.check_output(
            ["git", "-C", SAFEAI_REPO, "rev-parse", "HEAD"]).decode().strip(),
        "conventions": {
            "understatement_of_width_pct": "100 * (1 - width_declared_zero / width_measured); "
                                           "the denominator is the wider, honest width, matching "
                                           "the deposited chain convention",
            "cross_covariance_share_of_variance_pct": "100 * (1 - var_declared_zero / var_measured)",
            "delta_centre_under_redraw": "the fixed-draw point vector is reused as the "
                                         "linearisation centre under the redraw scheme, as in the "
                                         "deposited redraw run, which stores point_fixed_draw only",
            "independent_stream": "fixed draw only; the empirical counterpart of declaring the "
                                  "cross-terms zero",
        },
        "composites": {},
        "models": {},
    }

    # ---- Shapley efficiency identity, verified rather than asserted
    eff = {}
    for tag in TAGS:
        M = reps[tag]["paired"]
        phi_min = shapley_matrix(M, _game_min)
        phi_prod = shapley_matrix(M, _game_prod)
        eff[tag] = {
            "max_abs_sum_shapley_min_minus_min": float(np.abs(phi_min.sum(axis=1) - M.min(axis=1)).max()),
            "max_abs_sum_shapley_prod_minus_prod": float(np.abs(phi_prod.sum(axis=1) - M.prod(axis=1)).max()),
            "shapley_values_at_point_min_game": {
                m: float(v) for m, v in zip(METRICS, shapley_matrix(reps[tag]["point"][None, :], _game_min)[0])},
            "shapley_share_at_point_min_game_pct": {
                m: float(100.0 * v / reps[tag]["point"].min())
                for m, v in zip(METRICS, shapley_matrix(reps[tag]["point"][None, :], _game_min)[0])},
        }
    results["shapley_efficiency_check"] = {
        "statement": "the Shapley values are efficient, so their sum equals the grand-coalition "
                     "value v(N) by construction; with v(S) = min over S the composite therefore "
                     "IS the minimum of the three metrics, and with v(S) = prod over S it IS the "
                     "product. The Shapley step reallocates credit among the metrics without "
                     "changing the total. We verify the identity numerically instead of asserting it.",
        "per_model": eff,
    }
    for tag in TAGS:
        say("[shapley] %s efficiency residual: min-game %.3e, product-game %.3e"
            % (tag, eff[tag]["max_abs_sum_shapley_min_minus_min"],
               eff[tag]["max_abs_sum_shapley_prod_minus_prod"]))

    # ---- per model, per composite, per scheme
    for name, fn, gfn, desc in ARMS:
        results["composites"][name] = {"description": desc}
    results["composites"]["wsum_equal"]["weights"] = W_EQUAL.tolist()
    results["composites"]["wsum_accuracy_led"]["weights"] = W_ACC_LED.tolist()
    results["composites"]["wsum_accuracy_light"]["weights"] = W_ACC_LIGHT.tolist()
    results["composites"]["topsis"]["instantiation"] = (
        "ideal point fixed at (1,1,1), anti-ideal at (0,0,0), Euclidean distance, closeness "
        "coefficient. A data-dependent ideal would make the composite depend on the other "
        "alternatives and would require the joint covariance ACROSS models: out of scope here.")
    results["composites"]["shapley_min"]["instantiation"] = (
        "v(S) = min over S, v(empty) = 0; composite = sum of the three Shapley values. ONE "
        "defensible instantiation, not a canonical one; the Shapley-Lorenz line of the Pavia "
        "group is a different construction and this is a placeholder until their composite is known.")
    results["composites"]["shapley_prod"]["instantiation"] = (
        "v(S) = prod over S, v(empty) = 0; the smooth counterpart of shapley_min.")

    for tag in TAGS:
        point = reps[tag]["point"]
        S_fixed = np.cov(reps[tag]["paired"], rowvar=False)
        S_redraw = np.cov(reps[tag]["redraw"], rowvar=False)
        block = {
            "point_fixed_draw": {m: float(point[k]) for k, m in enumerate(METRICS)},
            "Sigma_fixed_draw": S_fixed.tolist(),
            "Sigma_redraw": S_redraw.tolist(),
            "Corr_fixed_draw": np.corrcoef(reps[tag]["paired"], rowvar=False).tolist(),
            "Corr_redraw": np.corrcoef(reps[tag]["redraw"], rowvar=False).tolist(),
            "mean_RGR_redraw_replicates": float(reps[tag]["redraw"][:, 2].mean()),
            "arms": {},
        }
        for name, fn, gfn, desc in ARMS:
            g_an = gfn(point)
            g_nu = numerical_grad(fn, point)
            arm = {
                "gradient_analytic": [float(v) for v in g_an],
                "gradient_numerical_central_difference": [float(v) for v in g_nu],
                "gradient_max_abs_difference": float(np.abs(g_an - g_nu).max()),
                "fixed_draw": interval_block(fn, gfn, point, S_fixed,
                                             reps[tag]["paired"], reps[tag]["ind"]),
                "redraw": interval_block(fn, gfn, point, S_redraw, reps[tag]["redraw"]),
            }
            block["arms"][name] = arm
            fd = arm["fixed_draw"]
            say("[%s/%s] C=%.4f  redraw CI=[%.4f, %.4f]  cross-var share %.2f%%  "
                "width understatement %.2f%%"
                % (tag, name, fd["C_point"],
                   arm["redraw"]["ci95_paired_bootstrap_percentile"][0],
                   arm["redraw"]["ci95_paired_bootstrap_percentile"][1],
                   arm["redraw"]["cross_covariance_share_of_variance_pct"],
                   arm["redraw"]["understatement_of_width_pct"]))
        results["models"][tag] = block

    # ---- continuity check against the deposited numbers
    dep = json.load(open(DEPOSIT_FIXED))
    depr = json.load(open(DEPOSIT_REDRAW))
    cont = {"note": "the deposited runs persist no replicate matrix, so this study regenerates "
                    "the replicates from data + seeds; these residuals show the regeneration is "
                    "exact and that the geometric-mean arm reproduces the deposited composite."}
    for tag in TAGS:
        d = dep["models"][tag]
        geo = results["models"][tag]["arms"]["geomean"]
        cont[tag] = {
            "point_vector_max_abs_diff": float(np.abs(
                reps[tag]["point"] - np.array([d["point"]["RGA"], d["point"]["RGE"], d["point"]["RGR"]])).max()),
            "C_geomean_abs_diff": abs(geo["fixed_draw"]["C_point"] - d["point"]["C_geomean"]),
            "Sigma_fixed_draw_max_abs_diff": float(np.abs(
                np.array(results["models"][tag]["Sigma_fixed_draw"]) - np.array(d["Sigma_paired"])).max()),
            "ci_paired_percentile_max_abs_diff": float(np.abs(
                np.array(geo["fixed_draw"]["ci95_paired_bootstrap_percentile"])
                - np.array(d["ci_C_paired_percentile"])).max()),
            "ci_independent_percentile_max_abs_diff": float(np.abs(
                np.array(geo["fixed_draw"]["ci95_independent_stream_bootstrap"])
                - np.array(d["ci_C_independent_percentile"])).max()),
            "ci_delta_max_abs_diff": float(np.abs(
                np.array(geo["fixed_draw"]["ci95_delta_measured_covariance"])
                - np.array(d["ci_C_delta"])).max()),
            "sd_C_paired_abs_diff": abs(geo["fixed_draw"]["sd_paired_bootstrap"] - d["sd_C_paired"]),
            "sd_C_independent_abs_diff": abs(
                geo["fixed_draw"]["sd_independent_stream_bootstrap"] - d["sd_C_independent"]),
            "Sigma_redraw_max_abs_diff": float(np.abs(
                np.array(results["models"][tag]["Sigma_redraw"])
                - np.array(depr["models"][tag]["redraw"]["Sigma"])).max()),
            "ci_redraw_percentile_max_abs_diff": float(np.abs(
                np.array(geo["redraw"]["ci95_paired_bootstrap_percentile"])
                - np.array(depr["models"][tag]["redraw"]["ci_C_paired_percentile"])).max()),
            "deposited_cross_variance_share_pct_recomputed":
                geo["fixed_draw"]["cross_covariance_share_of_variance_pct"],
        }
        say("[continuity] %s: point %.1e, C %.1e, Sigma_fixed %.1e, Sigma_redraw %.1e, "
            "paired CI %.1e, independent CI %.1e"
            % (tag, cont[tag]["point_vector_max_abs_diff"], cont[tag]["C_geomean_abs_diff"],
               cont[tag]["Sigma_fixed_draw_max_abs_diff"], cont[tag]["Sigma_redraw_max_abs_diff"],
               cont[tag]["ci_paired_percentile_max_abs_diff"],
               cont[tag]["ci_independent_percentile_max_abs_diff"]))
    results["continuity_check"] = cont

    # ---- delta is exact for the linear arms: check it
    lin = {}
    for tag in TAGS:
        lin[tag] = {}
        for name, w in (("wsum_equal", W_EQUAL), ("wsum_accuracy_led", W_ACC_LED),
                        ("wsum_accuracy_light", W_ACC_LIGHT)):
            for scheme, key in (("fixed_draw", "paired"), ("redraw", "redraw")):
                M = reps[tag][key]
                Sig = np.cov(M, rowvar=False)
                var_delta = float(w @ Sig @ w)
                Cb = M @ w
                var_boot = float(Cb.var(ddof=1))
                arm = results["models"][tag]["arms"][name][scheme]
                lin[tag]["%s/%s" % (name, scheme)] = {
                    "var_delta": var_delta,
                    "var_bootstrap_replicates": var_boot,
                    "relative_difference": float(abs(var_delta - var_boot) / var_boot),
                    "sd_delta": math.sqrt(var_delta),
                    "sd_bootstrap": math.sqrt(var_boot),
                    "ci95_delta": arm["ci95_delta_measured_covariance"],
                    "ci95_paired_bootstrap_percentile": arm["ci95_paired_bootstrap_percentile"],
                    "max_abs_endpoint_gap": float(np.abs(
                        np.array(arm["ci95_delta_measured_covariance"])
                        - np.array(arm["ci95_paired_bootstrap_percentile"])).max()),
                }
    results["linear_exactness_check"] = {
        "statement": "for a linear composite the delta-method variance grad' Sigma grad and the "
                     "sample variance of the pushed-through replicates are the SAME number, since "
                     "Sigma is the sample covariance of those replicates. Any residual is floating "
                     "point. The percentile interval still differs from the symmetric delta "
                     "interval by the skew of the replicate distribution, which is reported as the "
                     "endpoint gap.",
        "per_model": lin,
    }
    for tag in TAGS:
        r = lin[tag]["wsum_equal/redraw"]
        say("[linear] %s wsum_equal/redraw: var rel.diff %.2e, endpoint gap %.5f"
            % (tag, r["relative_difference"], r["max_abs_endpoint_gap"]))

    # ---- how far the weight profile can move the diagnostic
    results["weight_sensitivity"] = {
        "statement": "the three declared weight profiles are hand-picked. Because a weighted "
                     "sum is linear, the cross-term diagnostic over the WHOLE weight simplex is "
                     "a closed-form scan on the same covariance matrix, so the range is reported "
                     "instead of three points. The minimum over the closed simplex is attained at "
                     "a vertex, where the composite is a single metric and there is no cross-term "
                     "to drop; the interior scan floors every weight to keep the composite a "
                     "genuine composition of all three metrics.",
        "scheme": "redraw covariance (the primary scheme)",
        "per_model": {tag: weight_sensitivity(np.array(results["models"][tag]["Sigma_redraw"]))
                      for tag in TAGS},
    }
    for tag in TAGS:
        ws = results["weight_sensitivity"]["per_model"][tag]
        say("[weights] %s: cross-var share over the simplex %.2f%% to %.2f%% (max at w=%s); "
            "interior (all w >= %.2f) %.2f%% to %.2f%%"
            % (tag, ws["min_over_simplex"]["cross_covariance_share_of_variance_pct"],
               ws["max_over_simplex"]["cross_covariance_share_of_variance_pct"],
               np.round(ws["max_over_simplex"]["weights"], 2).tolist(), ws["interior_floor"],
               ws["min_over_interior"]["cross_covariance_share_of_variance_pct"],
               ws["max_over_interior"]["cross_covariance_share_of_variance_pct"]))

    # ---- where the replicate cloud sits, and how often each metric is the minimum
    results["replicate_diagnostics"] = replicate_diagnostics(reps)
    rd = results["replicate_diagnostics"]
    say("[diagnostics] single components over the six replicate matrices: smallest %.4f, "
        "largest %.4f" % (rd["min_component_over_all_matrices"], rd["max_component_over_all_matrices"]))
    say("[diagnostics] closest approach of the replicate cloud: to the ideal (1,1,1) %.4f, "
        "to the anti-ideal (0,0,0) %.4f"
        % (rd["min_distance_to_ideal_over_all_matrices"],
           rd["min_distance_to_anti_ideal_over_all_matrices"]))
    for k in ["logit/paired", "logit/ind", "logit/redraw", "rf/paired", "rf/ind", "rf/redraw"]:
        say("[diagnostics] %-12s argmin counts %s" % (k, rd[k]["argmin_counts"]))
    say("[diagnostics] matrices in which RGA is the argmin in EVERY replicate: %d of 6"
        % rd["n_matrices_with_argmin_RGA_in_every_replicate"])

    # ---- non-smoothness demonstration
    results["nonsmoothness"] = run_nonsmoothness(reps, pre, IDX, Zmat, col_std,
                                                 X_te, y_te, models)
    results["replicates_loaded_from_cache"] = bool(from_cache)

    results["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(HERE, "results-aggregation.json"), "w") as f:
        json.dump(results, f, indent=2)
    say("[done] results-aggregation.json, run-aggregation.log; total %.1fs"
        % results["runtime_seconds"])
    with open(os.path.join(HERE, "run-aggregation.log"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")


def run_nonsmoothness(reps, pre, IDX, Zmat, col_std, X_te, y_te, models):
    out = {
        "question": "where does the delta method break and the paired bootstrap not? The "
                    "min-type Shapley composite has a kink where the argmin changes. At the "
                    "deposited operating point the argmin is frozen, so we walk a one-parameter "
                    "family of noise levels THROUGH that point until the argmin crosses.",
        "caveat": "changing the noise level changes the estimand. These numbers are a "
                  "methodological demonstration and are not a correction to any deposited figure. "
                  "The family is exact at the deposited point: the perturbation is s * Z * "
                  "per-feature-sd with Z the same standard normal matrix the deposited run drew, "
                  "so s = 0.5 reproduces the deposited perturbation.",
        "scheme": "fixed draw; the redraw scheme would cost about five minutes per noise level "
                  "and would not change the geometry being demonstrated.",
        "models": {},
    }
    for tag in TAGS:
        t0 = time.time()
        level = nonsmoothness_sweep(tag, y_te, X_te, models, pre, IDX, Zmat, col_std)
        base_cols = reps[tag]["paired"][:, :2]          # RGA, RGE do not depend on the noise level

        def at(s):
            pt, col, _ = level(s)
            M = np.column_stack([base_cols, col])
            return sweep_record(s, pt, M)

        grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        coarse = [at(s) for s in grid]
        fracs = [r["fraction_RGR_below_RGA"] for r in coarse]
        say("[nonsmooth/%s] coarse sweep (fixed draw):" % tag)
        for s, r, fr in zip(grid, coarse, fracs):
            mc = r["min_composite"]
            say("    s=%5.2f  RGA=%.4f RGR=%.4f  argmin=%-3s frac(RGR<RGA)=%.4f  "
                "sd_delta=%.5f sd_boot=%.5f ratio=%.3f"
                % (s, r["point"]["RGA"], r["point"]["RGR"], r["argmin_at_point"], fr,
                   mc["sd_delta"], mc["sd_bootstrap"], mc["sd_ratio_delta_over_bootstrap"]))

        # bracket the 0.5 crossing and bisect
        lo = hi = None
        for i in range(len(grid) - 1):
            if (fracs[i] - 0.5) * (fracs[i + 1] - 0.5) <= 0:
                lo, hi = grid[i], grid[i + 1]
                break
        bisection = []
        chosen = None
        if lo is None:
            say("[nonsmooth/%s] no crossing of the 0.5 flip fraction inside the grid" % tag)
        else:
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                rec = at(mid)
                bisection.append({"noise_sd": mid, "fraction_RGR_below_RGA": rec["fraction_RGR_below_RGA"]})
                if rec["fraction_RGR_below_RGA"] < 0.5:
                    lo = mid
                else:
                    hi = mid
                chosen = rec
            say("[nonsmooth/%s] chosen level s=%.4f: frac(RGR<RGA)=%.4f, argmin counts %s"
                % (tag, chosen["noise_sd"], chosen["fraction_RGR_below_RGA"],
                   chosen["argmin_counts_over_replicates"]))
            mc = chosen["min_composite"]
            say("[nonsmooth/%s]   min composite: sd_delta=%.5f sd_boot=%.5f ratio=%.3f  "
                "delta CI=[%.4f, %.4f]  boot CI=[%.4f, %.4f]"
                % (tag, mc["sd_delta"], mc["sd_bootstrap"], mc["sd_ratio_delta_over_bootstrap"],
                   mc["ci95_delta"][0], mc["ci95_delta"][1],
                   mc["ci95_paired_bootstrap"][0], mc["ci95_paired_bootstrap"][1]))
            gc = chosen["geomean_control"]
            say("[nonsmooth/%s]   geomean control at the same level: sd ratio %.3f, "
                "lower-bound gap %.5f"
                % (tag, gc["sd_ratio_delta_over_bootstrap"], gc["lower_bound_gap_delta_minus_bootstrap"]))
        out["models"][tag] = {
            "coarse_sweep": coarse,
            "bisection_trace": bisection,
            "chosen_level": chosen,
            "deposited_level": coarse[0],
            "sweep_seconds": round(time.time() - t0, 1),
        }
        say("[nonsmooth/%s] sweep %.1fs" % (tag, time.time() - t0))
    return out


if __name__ == "__main__":
    main()
