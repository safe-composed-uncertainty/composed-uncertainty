#!/usr/bin/env python3
"""
The second dataset for the composed-uncertainty article: UCI Default of Credit
Card Clients (Taiwan), OpenML data id 42477.

Item A11 of UPGRADE-PLAN-2026-07-27.md, executed against the pre-registration
written on 27 July 2026 and stored beside this file as PRE-REGISTRATION.md.

The question. The article's headline mechanism is that the TAIL-SWAP
perturbation family anti-correlates the robustness curve with the accuracy and
explainability curves deep in the severity sweep, and that this is why declaring
the cross-terms zero barely moves the volume composite's interval while it
understates a two-link chain's interval by 23.6 per cent. That rests on one
dataset: Statlog German credit, n_test = 300, d = 20. This run puts the same
machinery on a second, thirty-times-larger tabular credit dataset and applies
the verdict rules that were fixed before any of it was computed.

What is reused, not reinvented. Every estimator, every seed and every stream is
copied from the pinned drivers:

  ../redraw-2026-07-24/redraw_experiment.py            the scalar triple, the two
                                                       conditioning schemes, the
                                                       two-link chain
  ../pavia-composite-2026-07-25/pavia_composite_experiment.py
                                                       the three curves on the
                                                       common severity grid, the
                                                       tail swap, the volume
                                                       composite in three
                                                       variants, TOPSIS
  ../pavia-composite-2026-07-25/verify_cross_terms.py  the six-rung estimator
                                                       bridge

Neither of those directories is written to. They are opened read-only, once, to
read back the German-credit anchors so that the comparison table cannot drift
from the deposit.

Substrate: koleso500/safeai pinned at 39768fcd5264c881f7174268bbffda52b298ae89,
the vendored read-only clone and the inert import stubs of the redraw run.

Anton Sokolov, Tyche Institute, Tallinn. 28 July 2026.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
EXPDIR = os.path.dirname(HERE)
REDRAW_DIR = os.path.join(EXPDIR, "redraw-2026-07-24")
PAVIA_DIR = os.path.join(EXPDIR, "pavia-composite-2026-07-25")
SAFEAI_REPO = os.path.join(REDRAW_DIR, "safeai-src")
STUBS = os.path.join(REDRAW_DIR, "_stubs")

for _p, _what in ((SAFEAI_REPO, "vendored safeai clone"), (STUBS, "import stubs")):
    if not os.path.isdir(_p):
        raise SystemExit("missing %s: this study reuses the pinned substrate of the "
                         "redraw run and neither clones nor writes one." % _what)

sys.path.insert(0, STUBS)
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

# ---------------------------------------------------------------- constants
# All of these are inherited verbatim from the pinned drivers. Where a number
# follows from d it is DERIVED, not chosen; see PRE-REGISTRATION.md section 4.
SEED = 20260723
TEST_SIZE = 0.3
ALPHA = 0.05
Z = 1.959963984540054
QLO, QHI = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
NOISE_SD = 0.5
CLASS_ORDER = np.array([0, 1])
TAGS = ["logit", "rf"]
CURVES = ["RGA", "RGE", "RGR"]
CONST_TOL = 1e-12
W_SENS = np.array([0.50, 0.20, 0.30])
REDRAW_BASE = {"logit": SEED + 100_000, "rf": SEED + 200_000}

B = int(os.environ.get("SD_B", "2000"))
B_FULL = {"logit": int(os.environ.get("SD_B_FULL_LOGIT", "300")),
          "rf": int(os.environ.get("SD_B_FULL_RF", "30"))}
# Output suffix. Empty for the primary run; the smoke run that exercises every code
# path at a tiny B writes beside it under its own name so that no primary output is
# ever silently overwritten.
TAG_OUT = os.environ.get("SD_TAG", "")
MC_COND = 2000
MC_R = 400
MC_SEED = SEED + 777                 # pinned: shares (MC_SEED), empirical (+1),
                                     # conditioning band (+2 = SEED + 779)
MC_SEED_CORR = MC_SEED + 3           # ADDED, see design_departures below
MC_SEED_CHAIN = MC_SEED + 4          # ADDED, see design_departures below

# German-credit anchors, frozen in PRE-REGISTRATION.md section 6 and re-read from
# the deposits at run time so that a drift in either would abort the comparison.
GERMAN_FROZEN = {
    "P1_corr_curvemean_RGE_RGR": {"logit": -0.1372, "rf": -0.3102},
    "P2_corr_curvemean_RGA_RGR": {"logit": -0.0583, "rf": -0.0800},
    "P3_delta_understatement_pct": {
        "logit": {"arithmetic": -0.91, "geometric": -1.86, "rms": 1.26, "topsis": -1.50},
        "rf": {"arithmetic": -3.08, "geometric": -3.09, "rms": -1.50, "topsis": -3.11}},
    "P4_empirical_understatement_pct": {
        "logit": {"arithmetic": 0.89, "geometric": 1.45, "rms": -1.41, "topsis": -3.10},
        "rf": {"arithmetic": -6.49, "geometric": -5.21, "rms": -4.86, "topsis": -7.56}},
    "P5_chain": {"fixed_draw": {"rho": 0.7222, "understatement_pct": 23.79},
                 "redraw": {"rho": 0.7120, "understatement_pct": 23.57}},
    "P6_bridge_RGE_RGR": {
        "logit": [0.5967, 0.6144, 0.4249, -0.2083, 0.5354, -0.1372],
        "rf": [0.3415, 0.3437, 0.2879, -0.1529, 0.4711, -0.3102]},
    "secondary_scalar_corr_fixed_draw": {
        "logit": {"RGA_RGE": 0.2858, "RGA_RGR": 0.1596, "RGE_RGR": 0.5967},
        "rf": {"RGA_RGE": 0.2922, "RGA_RGR": 0.1530, "RGE_RGR": 0.3415}},
}

BRIDGE_LABELS = [
    "scalar RGE (bare)      x scalar RGR (bare)      [deposit pair]",
    "scalar RGE (class_ord) x scalar RGR (class_ord) [convention only]",
    "curve-mean RGE         x scalar RGR (class_ord) [RGE estimator swapped]",
    "scalar RGE (class_ord) x curve-mean RGR (swap)  [RGR estimator swapped]",
    "curve-mean RGE         x noise-sweep RGR mean   [sweep, same family]",
    "curve-mean RGE         x curve-mean RGR (swap)  [study pair]",
]

LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------- data
def load_taiwan():
    """OpenML data id 42477. The ARFF ships anonymous names x1..x23; probe 1
    asserted the UCI dictionary mapping against measured column statistics and
    all eight checks passed. y = 1 is default, the UCI dictionary's own target
    and a MINORITY class at 22.1 per cent -- declared in the pre-registration,
    and the coding sensitivity of section 4.3 runs unconditionally below."""
    ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                      parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "1").astype(int)
    return X, y, list(ds.feature_names), \
        "uci-default-of-credit-card-clients-taiwan (OpenML id 42477 v1)"


# ---------------------------------------------------------------- the three curves
def rga_vector(y_true, p_full, grid):
    res = rga_curve(y_true, p_full, curve_method="partial", n_segments=grid,
                    normalize_to_perfect=False)
    return np.asarray(res["curve"], dtype=float)


def rge_vector_package(model, X, feature_names, grid, baseline="mean"):
    res = rge_curve(model, X, method="tabular", feature_names=feature_names,
                    masking_method="greedy", baseline=baseline, n_steps=grid,
                    verbose=False)
    order = [feature_names.index(nm) for nm in res["removed_features"]]
    return np.asarray(res["rge_scores"], dtype=float), order


def tail_swap(X, p):
    """The article's eq.-(9) perturbation. This is the vectorised form of
    verify_cross_terms.py; probe 2 checked it against the loop form of
    pavia_composite_experiment.py at p = 0.30 and found max abs diff 0.0e+00."""
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


# ---------------------------------------------------------------- composites
def tensor_mean(a, e, r, kind):
    A, E, R = a[:, None, None], e[None, :, None], r[None, None, :]
    if kind == "arithmetic":
        M = (A + E + R) / 3.0
    elif kind == "geometric":
        M = np.cbrt(A * E * R)
    elif kind == "rms":
        M = np.sqrt((A ** 2 + E ** 2 + R ** 2) / 3.0)
    else:
        raise ValueError(kind)
    return float(M.mean())


def real_cbrt(x):
    """The real cube root, which is what numpy's cbrt computes and what the
    definition means. Python's ** operator returns nan for a negative base at a
    fractional exponent; see brute_force_triple_sum below for why that matters
    here and did not matter on German credit."""
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def brute_force_triple_sum(a, e, r, kind, cube_root="pinned"):
    """Three explicit Python loops, no broadcasting: the check that the vectorised
    tensor is the triple sum the definition writes.

    DECLARED DIFFERENCE FROM THE GERMAN-CREDIT RUN. The accumulated partial RGA
    curve ends at a structural zero that floating-point arithmetic delivers as a
    residue of order 1e-16. On German credit that residue happened to be POSITIVE
    (+1.11e-16), so the pinned expression (a*e*r) ** (1/3) evaluated. Here it is
    NEGATIVE (-1.11e-16 for the logistic model, -2.22e-16 for the forest), and
    Python's ** returns nan for a negative base at a fractional exponent, while
    numpy's cbrt -- which the vectorised tensor and the closed form both use --
    returns the real cube root. The two forms are therefore run side by side:
    cube_root='pinned' reproduces the pinned expression verbatim, and
    cube_root='real' evaluates the same triple sum with the real cube root. Only
    the second can be compared with the tensor, and the comparison is reported as
    such rather than quietly repaired."""
    L = len(a)
    total = 0.0
    if kind == "arithmetic":
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    total += (a[i] + e[j] + r[k]) / 3.0
    elif kind == "geometric":
        cb = (lambda x: x ** (1.0 / 3.0)) if cube_root == "pinned" else real_cbrt
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    total += cb(a[i] * e[j] * r[k])
    elif kind == "rms":
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    total += math.sqrt((a[i] ** 2 + e[j] ** 2 + r[k] ** 2) / 3.0)
    else:
        raise ValueError(kind)
    return total / float(L ** 3)


def closed_form(a, e, r, kind):
    if kind == "arithmetic":
        return float((a.mean() + e.mean() + r.mean()) / 3.0)
    if kind == "geometric":
        return float(np.mean(np.cbrt(a)) * np.mean(np.cbrt(e)) * np.mean(np.cbrt(r)))
    return None


def v_arith(A, E, R):
    return (A.mean(1) + E.mean(1) + R.mean(1)) / 3.0


def v_geom(A, E, R):
    return np.cbrt(A).mean(1) * np.cbrt(E).mean(1) * np.cbrt(R).mean(1)


def v_rms(A, E, R, chunk=100):
    out = np.empty(A.shape[0], dtype=float)
    for s in range(0, A.shape[0], chunk):
        a = A[s:s + chunk][:, :, None, None]
        e = E[s:s + chunk][:, None, :, None]
        r = R[s:s + chunk][:, None, None, :]
        out[s:s + chunk] = np.sqrt((a * a + e * e + r * r) / 3.0).mean(axis=(1, 2, 3))
    return out


def v_arith_weighted(A, E, R, w=W_SENS):
    return w[0] * A.mean(1) + w[1] * E.mean(1) + w[2] * R.mean(1)


def v_arith_diag(A, E, R):
    return ((A + E + R) / 3.0).mean(1)


def v_geom_diag(A, E, R):
    return np.cbrt(A * E * R).mean(1)


def v_rms_diag(A, E, R):
    return np.sqrt((A ** 2 + E ** 2 + R ** 2) / 3.0).mean(1)


def topsis_reference(L):
    pis_a = np.ones(L)
    pis_e = np.concatenate([[1.0], np.zeros(L - 1)])
    pis_r = np.ones(L)
    nis_a = np.zeros(L)
    nis_e = np.linspace(1.0, 0.5, L)
    nis_r = np.concatenate([[1.0], np.zeros(L - 1)])
    return (np.concatenate([pis_a, pis_e, pis_r]),
            np.concatenate([nis_a, nis_e, nis_r]))


def topsis_cc(A, E, R, pis, nis, wvec):
    V = np.concatenate([A, E, R], axis=1)
    w2 = wvec ** 2
    sp = np.sqrt(((V - pis) ** 2 * w2).sum(axis=1))
    sm = np.sqrt(((V - nis) ** 2 * w2).sum(axis=1))
    return sm / (sm + sp)


# ---------------------------------------------------------------- gradients
def grad_rms_entries(a, e, r):
    L = len(a)
    M = np.sqrt((a[:, None, None] ** 2 + e[None, :, None] ** 2
                 + r[None, None, :] ** 2) / 3.0)
    inv = 1.0 / M
    c = 1.0 / (3.0 * L ** 3)
    return (a * inv.sum(axis=(1, 2)) * c,
            e * inv.sum(axis=(0, 2)) * c,
            r * inv.sum(axis=(0, 1)) * c)


def grad_topsis_entries(a, e, r, pis, nis, wvec):
    v = np.concatenate([a, e, r])
    w2 = wvec ** 2
    sp = math.sqrt(float(((v - pis) ** 2 * w2).sum()))
    sm = math.sqrt(float(((v - nis) ** 2 * w2).sum()))
    dsp = w2 * (v - pis) / sp
    dsm = w2 * (v - nis) / sm
    g = (sp * dsm - sm * dsp) / (sm + sp) ** 2
    L = len(a)
    return g[:L], g[L:2 * L], g[2 * L:]


# ---------------------------------------------------------------- interval blocks
def cross_term_decomposition(g, Sigma):
    pairs = {"RGA_RGE": (0, 1), "RGA_RGR": (0, 2), "RGE_RGR": (1, 2)}
    contrib = {nm: float(2.0 * g[i] * g[j] * Sigma[i, j]) for nm, (i, j) in pairs.items()}
    cross_total = float(sum(contrib.values()))
    diag = {m: float(g[k] ** 2 * Sigma[k, k]) for k, m in enumerate(CURVES)}
    diag_total = float(sum(diag.values()))
    absg = float(np.abs(g).sum())
    tot = diag_total + cross_total
    return {
        "gradient_share_of_L1_pct": {m: float(100.0 * abs(g[k]) / absg) if absg > 0 else 0.0
                                     for k, m in enumerate(CURVES)},
        "diagonal_contributions_to_variance": diag,
        "diagonal_share_of_variance_pct": ({m: float(100.0 * diag[m] / tot) for m in CURVES}
                                           if tot > 0 else {}),
        "cross_contributions_to_variance": contrib,
        "cross_total": cross_total,
        "share_of_cross_total_pct": ({nm: float(100.0 * v / cross_total)
                                      for nm, v in contrib.items()}
                                     if cross_total != 0 else {nm: None for nm in contrib}),
    }


def percentile_block(Vb):
    lo, hi = float(np.percentile(Vb, QLO)), float(np.percentile(Vb, QHI))
    return {"ci95_percentile": [lo, hi], "width": hi - lo,
            "sd": float(Vb.std(ddof=1)), "mean": float(Vb.mean())}


def mc_std(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return float(v.std(ddof=1)) if v.size > 1 else float("nan")


def delta_share_mc_se(Sb, g_fn, S0):
    rng = np.random.default_rng(MC_SEED)
    Bn = Sb.shape[0]
    shares, unders = [], []
    for _ in range(MC_R):
        Sr = Sb[rng.integers(0, Bn, Bn)]
        Sg = np.cov(Sr, rowvar=False)
        g = g_fn(S0)
        vm = float(g @ Sg @ g)
        vd = float(np.sum(g ** 2 * np.diag(Sg)))
        if vm > 0 and vd > 0:
            shares.append(100.0 * (1.0 - vd / vm))
            unders.append(100.0 * (1.0 - math.sqrt(vd) / math.sqrt(vm)))
    return mc_std(shares), mc_std(unders)


def empirical_under_mc_se(Vb, Vi):
    rng = np.random.default_rng(MC_SEED + 1)
    Bn = len(Vb)
    out = []
    for _ in range(MC_R):
        vb = Vb[rng.integers(0, Bn, Bn)]
        vi = Vi[rng.integers(0, Bn, Bn)]
        wb = np.percentile(vb, QHI) - np.percentile(vb, QLO)
        wi = np.percentile(vi, QHI) - np.percentile(vi, QLO)
        if wb > 0:
            out.append(100.0 * (1.0 - wi / wb))
    return mc_std(out)


def corr_mc_se(u, v, seed=MC_SEED_CORR):
    """ADDED for this run, and declared. The pinned drivers carry the second-level
    resampling block for the width shares but not for the correlations. The
    pre-registered verdict rule calls a sign only where its absolute value exceeds
    twice its Monte Carlo standard error, so the same block -- MC_R resamples of
    the B replicate rows -- is applied to the correlations as well. It changes no
    estimate; it only qualifies one."""
    rng = np.random.default_rng(seed)
    Bn = len(u)
    out = []
    for _ in range(MC_R):
        pick = rng.integers(0, Bn, Bn)
        uu, vv = u[pick], v[pick]
        if uu.std() > 0 and vv.std() > 0:
            out.append(float(np.corrcoef(uu, vv)[0, 1]))
    return mc_std(out)


def arm_block(name, describe, fn, curves_point, reps_paired, reps_ind,
              summary_fn=None, grad_fn=None, exact_in_summaries=None):
    a0, e0, r0 = curves_point
    V0 = float(fn(a0[None, :], e0[None, :], r0[None, :])[0])
    Vb = np.asarray(fn(*reps_paired), dtype=float)
    Vi = np.asarray(fn(*reps_ind), dtype=float)
    pb, pi = percentile_block(Vb), percentile_block(Vi)
    out = {
        "description": describe,
        "V_point": V0,
        "paired_bootstrap": pb,
        "independent_stream_bootstrap": pi,
        "empirical_understatement_of_width_pct":
            float(100.0 * (1.0 - pi["width"] / pb["width"])) if pb["width"] > 0 else None,
        "empirical_understatement_mc_se_pct": empirical_under_mc_se(Vb, Vi),
        "plugin_minus_bootstrap_mean": float(V0 - pb["mean"]),
        "delta_method": None,
    }
    if summary_fn is not None:
        S0 = summary_fn(a0[None, :], e0[None, :], r0[None, :])[0]
        Sb = summary_fn(*reps_paired)
        Sigma = np.cov(Sb, rowvar=False)
        Corr = np.corrcoef(Sb, rowvar=False)
        g = grad_fn(S0)
        var_meas = float(g @ Sigma @ g)
        var_decl = float(np.sum(g ** 2 * np.diag(Sigma)))
        w_meas = 2.0 * Z * math.sqrt(var_meas)
        w_decl = 2.0 * Z * math.sqrt(var_decl)
        out["delta_method"] = {
            "summaries_are_exact": bool(exact_in_summaries),
            "summary_point": [float(x) for x in S0],
            "gradient_at_point": [float(x) for x in g],
            "Sigma_of_summaries": [[float(x) for x in row] for row in np.atleast_2d(Sigma)],
            "Corr_of_summaries": [[float(x) for x in row] for row in np.atleast_2d(Corr)],
            "var_measured_covariance": var_meas,
            "var_cross_terms_declared_zero": var_decl,
            "cross_covariance_share_of_variance_pct":
                float(100.0 * (1.0 - var_decl / var_meas)) if var_meas > 0 else 0.0,
            "ci95_delta_measured_covariance": [V0 - Z * math.sqrt(var_meas),
                                               V0 + Z * math.sqrt(var_meas)],
            "ci95_delta_cross_terms_declared_zero": [V0 - Z * math.sqrt(var_decl),
                                                     V0 + Z * math.sqrt(var_decl)],
            "width_delta_measured": w_meas,
            "width_delta_declared_zero": w_decl,
            "understatement_of_width_pct":
                float(100.0 * (1.0 - w_decl / w_meas)) if w_meas > 0 else 0.0,
            "understatement_factor": float(w_meas / w_decl) if w_decl > 0 else None,
            "cross_term_decomposition": cross_term_decomposition(g, np.atleast_2d(Sigma)),
        }
        se_share, se_under = delta_share_mc_se(Sb, grad_fn, S0)
        out["delta_method"]["cross_share_mc_se_pct"] = se_share
        out["delta_method"]["understatement_mc_se_pct"] = se_under
    out["_name"] = name
    return out


def geo_mean(v):
    return float(np.exp(np.mean(np.log(np.clip(v, 1e-12, None)))))


def corr_dict(M):
    C = np.corrcoef(M, rowvar=False)
    return {"matrix": C.tolist(), "corr_RGA_RGE": float(C[0, 1]),
            "corr_RGA_RGR": float(C[0, 2]), "corr_RGE_RGR": float(C[1, 2])}


# ---------------------------------------------------------------- German anchors
def read_german_anchors():
    """Read the deposited German-credit artefacts READ-ONLY and check them against
    the values frozen in PRE-REGISTRATION.md section 6. If any of them has moved,
    the comparison is invalid and the run aborts."""
    with open(os.path.join(PAVIA_DIR, "results-pavia-composite.json")) as fh:
        pav = json.load(fh)
    with open(os.path.join(REDRAW_DIR, "results-redraw.json")) as fh:
        red = json.load(fh)
    with open(os.path.join(PAVIA_DIR, "verify-cross-terms.log")) as fh:
        vlog = fh.read()

    g = {"source_files": [
        os.path.join(PAVIA_DIR, "results-pavia-composite.json"),
        os.path.join(REDRAW_DIR, "results-redraw.json"),
        os.path.join(PAVIA_DIR, "verify-cross-terms.log")],
        "n_test": pav["n_test"], "d": pav["d"], "curve_length_L": pav["curve_length_L"],
        "B": pav["B"], "B_conditioning_check": pav["B_conditioning_check"],
        "models": {}}
    for tag in TAGS:
        m = pav["models"][tag]
        C = m["curve_summaries"]["mean_of_curve"]["Corr"]
        arms = m["arms"]
        g["models"][tag] = {
            "corr_curve_means": {"RGA_RGE": C[0][1], "RGA_RGR": C[0][2], "RGE_RGR": C[1][2]},
            "per_severity_index_correlation":
                m["curve_summaries"]["per_severity_index_correlation"],
            "mean_per_index_correlation": m["curve_summaries"]["mean_per_index_correlation"],
            "delta_understatement_pct": {
                k: arms[k]["delta_method"]["understatement_of_width_pct"]
                for k in ("arithmetic", "geometric", "rms", "topsis")},
            "delta_understatement_mc_se_pct": {
                k: arms[k]["delta_method"]["understatement_mc_se_pct"]
                for k in ("arithmetic", "geometric", "rms", "topsis")},
            "empirical_understatement_pct": {
                k: arms[k]["empirical_understatement_of_width_pct"]
                for k in ("arithmetic", "geometric", "rms", "topsis")},
            "V_point": {k: arms[k]["V_point"] for k in
                        ("arithmetic", "geometric", "rms", "topsis")},
            "paired_ci_width": {k: arms[k]["paired_bootstrap"]["width"] for k in
                                ("arithmetic", "geometric", "rms", "topsis")},
            "scalar_corr_fixed_draw": red["models"][tag]["fixed_draw"]["Corr"],
            "conditioning_sd_ratio": {
                k: m["conditioning_check"]["arms"][k]["sd_ratio_full_over_frozen"]
                for k in ("arithmetic", "geometric", "rms")},
        }
    g["chain"] = {s: {"rho": red["chain"][s]["cross_link_correlation"],
                      "understatement_pct": red["chain"][s]["understatement_of_width_pct"],
                      "width_measured": red["chain"][s]["width_measured"],
                      "width_declared_zero": red["chain"][s]["width_declared_zero"],
                      "h_point": red["chain"][s]["h_point"]}
                  for s in ("fixed_draw", "redraw")}

    # the bridge lives in the verification log of the independent rerun
    blocks = re.split(r"^=+$", vlog, flags=re.M)
    bridge = {}
    for tag in TAGS:
        idx = vlog.find("\n%s\n" % tag)
        seg = vlog[idx:idx + 4000] if idx >= 0 else ""
        vals = re.findall(r"^\s+([+-]\d\.\d{4})\s{3}(?:scalar|curve)", seg, flags=re.M)
        bridge[tag] = [float(v) for v in vals[:6]]
    g["bridge_RGE_RGR"] = bridge
    del blocks

    # --- assertions against the frozen pre-registration values
    checks = []

    def chk(label, got, want, tol=5e-5):
        ok = abs(float(got) - float(want)) <= tol
        checks.append({"label": label, "read_back": float(got), "frozen": float(want),
                       "ok": bool(ok)})
        return ok

    for tag in TAGS:
        chk("P1 %s" % tag, g["models"][tag]["corr_curve_means"]["RGE_RGR"],
            GERMAN_FROZEN["P1_corr_curvemean_RGE_RGR"][tag])
        chk("P2 %s" % tag, g["models"][tag]["corr_curve_means"]["RGA_RGR"],
            GERMAN_FROZEN["P2_corr_curvemean_RGA_RGR"][tag])
        for k in ("arithmetic", "geometric", "rms", "topsis"):
            chk("P3 %s %s" % (tag, k), g["models"][tag]["delta_understatement_pct"][k],
                GERMAN_FROZEN["P3_delta_understatement_pct"][tag][k], tol=6e-3)
            chk("P4 %s %s" % (tag, k), g["models"][tag]["empirical_understatement_pct"][k],
                GERMAN_FROZEN["P4_empirical_understatement_pct"][tag][k], tol=6e-3)
        for i, v in enumerate(GERMAN_FROZEN["P6_bridge_RGE_RGR"][tag]):
            chk("P6 %s rung %d" % (tag, i + 1), bridge[tag][i], v, tol=1e-4)
        for nm, v in GERMAN_FROZEN["secondary_scalar_corr_fixed_draw"][tag].items():
            chk("scalar %s %s" % (tag, nm),
                g["models"][tag]["scalar_corr_fixed_draw"]["corr_" + nm], v)
    for s in ("fixed_draw", "redraw"):
        chk("P5 %s rho" % s, g["chain"][s]["rho"], GERMAN_FROZEN["P5_chain"][s]["rho"])
        chk("P5 %s understatement" % s, g["chain"][s]["understatement_pct"],
            GERMAN_FROZEN["P5_chain"][s]["understatement_pct"], tol=6e-3)
    g["frozen_value_checks"] = checks
    g["all_frozen_values_reproduced"] = bool(all(c["ok"] for c in checks))
    return g


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    results = {
        "study": "The second dataset for the composed-uncertainty article: "
                 "UCI Default of Credit Card Clients (Taiwan)",
        "plan_item": "A11 of UPGRADE-PLAN-2026-07-27.md",
        "pre_registration": os.path.join(HERE, "PRE-REGISTRATION.md"),
        "author": "Anton Sokolov, Tyche Institute, Tallinn",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "safeai_commit": subprocess.check_output(
            ["git", "-C", SAFEAI_REPO, "rev-parse", "HEAD"]).decode().strip(),
        "numpy": np.__version__,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "seed": SEED, "alpha": ALPHA, "z": Z, "B": B, "B_conditioning_check": B_FULL,
        "design_departures_from_the_german_credit_study": [
            "L = d + 1 = 24 and GRID = d = 23 (German credit: 21 and 20). The RULE is "
            "unchanged; the number follows from d because the greedy removal curve has "
            "one knot per feature plus the zero-removal anchor. Consequences: the "
            "composite averages 13,824 tensor cells instead of 9,261, and the tail-swap "
            "severity spacing is 0.5/23 = 0.021739 instead of 0.5/20 = 0.025.",
            "n_test = 9,000 instead of 300. The split rule is unchanged (stratified, "
            "test_size = 0.3); only the sample it is applied to is larger.",
            "The positive class is a minority (y = 1 = default, 22.1 per cent) where "
            "German credit's y = 1 = good is the majority (70.0 per cent). Declared in "
            "advance, with an unconditional coding sensitivity under y' = 1 - y.",
            "ADDED and declared: a Monte Carlo standard error for the correlations and "
            "for the chain understatement, by the same second-level resampling block "
            "(MC_R = %d resamples of the B replicate rows) that the pinned driver "
            "already applies to the width shares, seeded at MC_SEED + 3 and MC_SEED + 4. "
            "The pre-registered verdict rule calls a sign only against twice its Monte "
            "Carlo standard error, and the pinned drivers do not carry that block for "
            "the correlations. It qualifies estimates; it changes none." % MC_R,
            "The estimator bridge is computed inside the same paired loop as the "
            "composite rather than in a second script. The index stream is identical "
            "(default_rng(SEED + 1)), so the numbers are the ones the separate script "
            "would produce; the separate independent recomputation is "
            "verify_second_dataset.py, run afterwards.",
            "The two length-d truncation readings are at L = 23, not L = 20.",
            "FOUND BY THE SMOKE RUN AND DECLARED, not a choice: the terminal knot of "
            "the accumulated partial RGA curve is a structural zero delivered as a "
            "floating-point residue, and on Taiwan that residue is NEGATIVE (-1.11e-16 "
            "logit, -2.22e-16 rf) where on German credit it was positive (+1.11e-16). "
            "The pinned geometric brute-force check writes (a*e*r) ** (1/3), and "
            "Python's ** returns nan for a negative base at a fractional exponent, so "
            "that one check cannot evaluate here. The vectorised tensor and the closed "
            "form both use numpy's cbrt, which returns the real cube root, and are "
            "unaffected. Both brute-force forms are run and both are reported.",
        ],
        "unchanged_from_the_german_credit_study": [
            "safeai pinned at 39768fcd5264c881f7174268bbffda52b298ae89, vendored "
            "read-only, with the inert torch/art import stubs.",
            "SEED = 20260723 and every derived stream: paired default_rng(SEED+1); "
            "independent default_rng(SEED+10+k), k = 0,1,2; chain shared "
            "default_rng(SEED+99); Gaussian sweep default_rng(SEED) per model; "
            "scheme-B redraw bases SEED+100000 (logit) and SEED+200000 (rf) consumed "
            "as base+b; Monte Carlo seeds SEED+777 and SEED+779.",
            "B = 2000 paired and 2000 independent-stream replicates; alpha = 0.05; "
            "z = 1.959963984540054; percentile endpoints 2.5 and 97.5.",
            "Untuned models: StandardScaler + LogisticRegression(max_iter=2000) and "
            "RandomForestClassifier(n_estimators=300).",
            "The tail swap verbatim, on the same p-range [0, 0.5].",
            "The fixed Gaussian draw at 0.5 column sd and the Gaussian sweep at "
            "sigma_t = t/GRID column sd.",
            "Three volume variants as the mean of the L x L x L tensor, the "
            "matched-severity diagonal readings, TOPSIS with the thesis ideal and "
            "anti-ideal vectors read at the run's L, the weighted sensitivity arm at "
            "w = (0.50, 0.20, 0.30) still declared a departure from the source authors.",
            "The chain h = C_A * C_B with C the geometric mean of the three SCALAR "
            "metrics, link A = logit, link B = rf.",
            "Both conditioning senses: the scalar driver's fixed draw against the "
            "per-replicate redraw, and the curve driver's frozen fast path against a "
            "full re-search of the greedy order on the resampled rows, at "
            "B_full = 300 (logit) and 30 (rf).",
            "class_order = [0, 1] wherever the pinned drivers pass it.",
        ],
    }

    say("=" * 78)
    say("The second dataset for the composed-uncertainty article")
    say("UCI Default of Credit Card Clients (Taiwan), OpenML id 42477")
    say("Anton Sokolov, Tyche Institute, Tallinn -- run of %s"
        % results["generated_utc"])
    say("=" * 78)

    # ---- German anchors, read back read-only
    say("")
    say("[anchors] reading the deposited German-credit artefacts, read-only")
    german = read_german_anchors()
    bad = [c for c in german["frozen_value_checks"] if not c["ok"]]
    say("[anchors] %d of %d frozen pre-registered values reproduced from the deposits"
        % (len(german["frozen_value_checks"]) - len(bad),
           len(german["frozen_value_checks"])))
    if bad:
        for c in bad:
            say("[anchors] MISMATCH %s: read %.6f, frozen %.6f"
                % (c["label"], c["read_back"], c["frozen"]))
        raise SystemExit("German-credit anchors have drifted; the comparison would be "
                         "invalid. Aborting before any second-dataset number is computed.")
    results["german_credit_anchors"] = german

    # ---- data and models
    X, y, names, dataset = load_taiwan()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=SEED, stratify=y)
    n, d = X_te.shape
    GRID = d
    L = GRID + 1
    results.update({"dataset": dataset, "n_train": int(len(y_tr)), "n_test": int(n),
                    "d": int(d), "grid": int(GRID), "curve_length_L": int(L),
                    "positive_share_test": float(y_te.mean()),
                    "positives_in_test": int(y_te.sum()),
                    "target_coding": "y = 1 is default (the UCI dictionary's own "
                                     "target), a MINORITY class"})
    say("[data]  %s: train = %d, test = %d, d = %d, positive share %.4f"
        % (dataset, len(y_tr), n, d, y_te.mean()))
    say("[grid]  severity grid t/%d, t = 0..%d  ->  curve length L = %d, tensor cells %d"
        % (GRID, GRID, L, L ** 3))
    say("[boot]  B = %d paired and %d independent-stream replicates; B_full = %d (logit) "
        "/ %d (rf)" % (B, B, B_FULL["logit"], B_FULL["rf"]))
    say("[subst] safeai @ %s (vendored, read-only)" % results["safeai_commit"][:12])

    models = {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    say("[fit]   both models fitted on %d rows, untuned" % len(y_tr))

    # ---- index streams, recipe inherited verbatim
    t0 = time.time()
    IDX = np.empty((B, n), dtype=np.int64)
    boot_rng = np.random.default_rng(SEED + 1)
    for b in range(B):
        IDX[b] = boot_rng.integers(0, n, n)
    IND = np.empty((3, B, n), dtype=np.int64)
    for k in range(3):
        rk = np.random.default_rng(SEED + 10 + k)
        for b in range(B):
            IND[k, b] = rk.integers(0, n, n)
    say("[streams] paired + three independent index streams drawn in %.1f s "
        "(%.0f MB int64)" % (time.time() - t0, 4 * B * n * 8 / 1e6))

    ps = 0.5 * np.arange(L) / GRID
    pis, nis = topsis_reference(L)
    wvec = np.concatenate([np.full(L, 1.0 / 3.0)] * 3)
    cc_pis = float(topsis_cc(pis[:L][None, :], pis[L:2 * L][None, :],
                             pis[2 * L:][None, :], pis, nis, wvec)[0])
    cc_nis = float(topsis_cc(nis[:L][None, :], nis[L:2 * L][None, :],
                             nis[2 * L:][None, :], pis, nis, wvec)[0])
    say("[topsis] reference check at L = %d: Ci(best) = %.6f, Ci(worst) = %.6f "
        "(thesis: 1.0 and 0.0)" % (L, cc_pis, cc_nis))
    results["topsis_reference"] = {"L": int(L), "Ci_of_best_case": cc_pis,
                                   "Ci_of_worst_case": cc_nis}

    # the deposit's module-level rng, consumed logit first then rf, exactly as in
    # redraw_experiment.py's precompute loop
    dep_rng = np.random.default_rng(SEED)

    results["models"] = {}
    KEEP = {}
    STORE = {}
    for tag in TAGS:
        model = models[tag]
        say("")
        say("#" * 78)
        say("### %s" % tag)
        say("#" * 78)
        mres = {}
        p_full = model.predict_proba(X_te)[:, 1]
        col_mean = X_te.mean(axis=0)
        col_std = X_te.std(axis=0, keepdims=True)

        # ---- point curves
        t0 = time.time()
        a0 = rga_vector(y_te, p_full, GRID)
        e0, order = rge_vector_package(model, X_te, names, GRID, "mean")
        r0_swap = np.empty(L)
        for t in range(L):
            r0_swap[t] = rgr_score(p_full,
                                   model.predict_proba(tail_swap(X_te, ps[t]))[:, 1],
                                   class_order=CLASS_ORDER)
        t_point = time.time() - t0
        say("[curves] point estimate in %.1f s" % t_point)
        for nm, v in (("RGA", a0), ("RGE", e0), ("RGR", r0_swap)):
            say("  %s  len=%d  [%s ... %s]"
                % (nm, L, " ".join("%.4f" % x for x in v[:4]),
                   " ".join("%.4f" % x for x in v[-2:])))

        rga_scalar = float(rga_score(y_te, p_full))
        mres["anchor_consistency"] = {
            "rga_anchor_curve": float(a0[0]), "rga_score_scalar": rga_scalar,
            "abs_gap_anchor_vs_scalar": abs(float(a0[0]) - rga_scalar)}
        say("[anchor] RGA curve anchor %.12f vs the scalar rga_score %.12f -> |gap| %.2e"
            % (a0[0], rga_scalar, abs(float(a0[0]) - rga_scalar)))

        # ---- the composite at the point estimate, three ways
        say("[composite] point estimate, three ways")
        comp = {}
        for kind in ("arithmetic", "geometric", "rms"):
            t1 = time.time()
            v_bf = brute_force_triple_sum(a0, e0, r0_swap, kind, cube_root="pinned")
            t_bf = time.time() - t1
            v_bf_real = (brute_force_triple_sum(a0, e0, r0_swap, kind, cube_root="real")
                         if kind == "geometric" else v_bf)
            v_tn = tensor_mean(a0, e0, r0_swap, kind)
            v_cf = closed_form(a0, e0, r0_swap, kind)
            comp[kind] = {
                "V_brute_force_triple_sum_pinned_expression": v_bf,
                "pinned_expression_is_finite": bool(np.isfinite(v_bf)),
                "V_brute_force_triple_sum_real_cube_root": v_bf_real,
                "V_vectorised_tensor": v_tn, "V_closed_form": v_cf,
                "abs_residual_tensor_vs_brute_force": abs(v_tn - v_bf_real),
                "abs_residual_closed_form_vs_brute_force":
                    (abs(v_cf - v_bf_real) if v_cf is not None else None),
                "brute_force_seconds": t_bf}
            say("  %-10s brute(pinned)=%.12f  brute(real cbrt)=%.12f  tensor=%.12f  "
                "closed=%s  |tensor-brute|=%.2e"
                % (kind, v_bf, v_bf_real, v_tn,
                   ("%.12f" % v_cf) if v_cf is not None else "none        ",
                   abs(v_tn - v_bf_real)))
        if not comp["geometric"]["pinned_expression_is_finite"]:
            say("  NOTE: the pinned geometric brute-force expression is not finite here "
                "because the terminal RGA knot is a NEGATIVE floating-point residue "
                "(%.2e) and Python's ** returns nan for a negative base at a fractional "
                "exponent. On German credit the same residue was positive. The tensor, "
                "the closed form and the real-cube-root brute force all agree; the "
                "pinned expression is reported as it stands." % a0[-1])
        mres["factorisation_check"] = comp

        ga, ge, gr = grad_rms_entries(a0, e0, r0_swap)
        euler = float(ga @ a0 + ge @ e0 + gr @ r0_swap)
        mres["rms_gradient_euler_identity"] = {
            "sum_of_projections": euler, "V_rms": comp["rms"]["V_vectorised_tensor"],
            "abs_gap": abs(euler - comp["rms"]["V_vectorised_tensor"])}
        say("  rms gradient check (Euler): %.12f vs V = %.12f, |gap| = %.2e"
            % (euler, comp["rms"]["V_vectorised_tensor"],
               abs(euler - comp["rms"]["V_vectorised_tensor"])))

        # ---- precompute, curve path
        t0 = time.time()
        p_mask = np.empty((L, n))
        p_mask[0] = p_full
        for t in range(1, L):
            Xm = X_te.copy()
            cols = order[:t]
            Xm[:, cols] = col_mean[cols]
            p_mask[t] = model.predict_proba(Xm)[:, 1]
        p_swap = np.empty((L, n))
        for t in range(L):
            p_swap[t] = model.predict_proba(tail_swap(X_te, ps[t]))[:, 1]
        rng_noise = np.random.default_rng(SEED)
        p_noise = np.empty((L, n))
        for t in range(L):
            sigma = t / float(GRID)
            Xp = (X_te + rng_noise.normal(0.0, sigma, size=X_te.shape) * col_std
                  if sigma > 0 else X_te)
            p_noise[t] = model.predict_proba(Xp)[:, 1]
        t_pre = time.time() - t0

        # ---- precompute, the deposited scalar path (single-feature masks and the
        #      ONE fixed Gaussian perturbation, drawn from the shared SEED stream in
        #      the order logit then rf, exactly as redraw_experiment.py does)
        t0 = time.time()
        p_mask_single = np.empty((d, n))
        for j in range(d):
            Xm = X_te.copy()
            Xm[:, j] = col_mean[j]
            p_mask_single[j] = model.predict_proba(Xm)[:, 1]
        pert_fixed = dep_rng.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
        p_pert_fixed = model.predict_proba(X_te + pert_fixed)[:, 1]
        t_pre2 = time.time() - t0
        say("[precompute] curve path %.1f s (%d masked + %d swapped + %d noise), "
            "scalar path %.1f s (%d single masks + 1 fixed draw)"
            % (t_pre, L, L, L, t_pre2, d))

        # ---- the fast path must reproduce the package curves at the point estimate
        def triple(idx_a, idx_e, idx_r):
            pa, pe, pr = p_full[idx_a], p_full[idx_e], p_full[idx_r]
            a = rga_vector(y_te[idx_a], pa, GRID)
            e = np.empty(L)
            e[0] = 1.0
            for t in range(1, L):
                e[t] = rge_score(pe, p_mask[t][idx_e], class_order=CLASS_ORDER)
            r = np.empty(L)
            for t in range(L):
                r[t] = rgr_score(pr, p_swap[t][idx_r], class_order=CLASS_ORDER)
            return a, e, r

        full_idx = np.arange(n)
        a_chk, e_chk, r_chk = triple(full_idx, full_idx, full_idx)
        fast_gaps = {"RGA": float(np.max(np.abs(a_chk - a0))),
                     "RGE": float(np.max(np.abs(e_chk - e0))),
                     "RGR": float(np.max(np.abs(r_chk - r0_swap)))}
        say("[fastpath] max abs gap vs the package curves: RGA %.2e  RGE %.2e  RGR %.2e"
            % (fast_gaps["RGA"], fast_gaps["RGE"], fast_gaps["RGR"]))
        mres["fast_path_vs_package_max_abs_gap"] = fast_gaps

        # ---- which scoring convention the package curve builders use
        e_with = np.array([1.0] + [rge_score(p_full, p_mask[t], class_order=CLASS_ORDER)
                                   for t in range(1, L)])
        e_without = np.array([1.0] + [rge_score(p_full, p_mask[t]) for t in range(1, L)])
        r_swap_1d = np.array([rgr_score(p_full, p_swap[t]) for t in range(L)])
        conv = {
            "rge_max_abs_gap_vs_package_curve_with_class_order":
                float(np.max(np.abs(e_with - e0))),
            "rge_max_abs_gap_vs_package_curve_without_class_order":
                float(np.max(np.abs(e_without - e0))),
            "rgr_tail_swap_matrix_convention_last": float(r0_swap[-1]),
            "rgr_tail_swap_vector_convention_last": float(r_swap_1d[-1]),
            "rgr_tail_swap_max_abs_gap_between_conventions":
                float(np.max(np.abs(r0_swap - r_swap_1d))),
            "V_under_vector_convention": {
                k: tensor_mean(a0, e_without, r_swap_1d, k)
                for k in ("arithmetic", "geometric", "rms")},
        }
        say("[convention] RGE fast path vs package curve: with class_order %.2e, "
            "without %.2e"
            % (conv["rge_max_abs_gap_vs_package_curve_with_class_order"],
               conv["rge_max_abs_gap_vs_package_curve_without_class_order"]))
        say("[convention] RGR at p = 0.5: matrix convention %.4f vs bare vector %.4f "
            "(max gap over the curve %.4f)"
            % (conv["rgr_tail_swap_matrix_convention_last"],
               conv["rgr_tail_swap_vector_convention_last"],
               conv["rgr_tail_swap_max_abs_gap_between_conventions"]))
        mres["scoring_convention"] = conv

        # ---- the paired pass: curves, the Gaussian sweep, and BOTH scalar triples
        #      on one and the same index stream, so that the estimator bridge and the
        #      composite are read off the same replicates.
        t0 = time.time()
        Ab = np.empty((B, L)); Eb = np.empty((B, L)); Rb = np.empty((B, L))
        Rb_noise = np.empty((B, L))
        SC = np.empty((B, 3))        # bare-vector convention: the deposit's estimators
        SC_co = np.empty((B, 2))     # the same RGE/RGR scalars with class_order
        for b in range(B):
            ib = IDX[b]
            a, e, r = triple(ib, ib, ib)
            Ab[b], Eb[b], Rb[b] = a, e, r
            pb = p_full[ib]
            yb = y_te[ib]
            for t in range(L):
                Rb_noise[b, t] = rgr_score(pb, p_noise[t][ib], class_order=CLASS_ORDER)
            SC[b, 0] = rga_score(yb, pb)
            SC[b, 1] = np.mean([rge_score(pb, p_mask_single[j][ib]) for j in range(d)])
            SC[b, 2] = rgr_score(pb, p_pert_fixed[ib])
            SC_co[b, 0] = np.mean([rge_score(pb, p_mask_single[j][ib],
                                             class_order=CLASS_ORDER) for j in range(d)])
            SC_co[b, 1] = rgr_score(pb, p_pert_fixed[ib], class_order=CLASS_ORDER)
            if (b + 1) % 500 == 0:
                say("  [paired] %d/%d replicates, %.0f s elapsed" % (b + 1, B,
                                                                    time.time() - t0))
        t_paired = time.time() - t0
        say("[bootstrap] paired pass, B = %d, %.1f s (%.1f ms a replicate)"
            % (B, t_paired, 1000.0 * t_paired / B))

        t0 = time.time()
        Ai = np.empty((B, L)); Ei = np.empty((B, L)); Ri = np.empty((B, L))
        for b in range(B):
            a, e, r = triple(IND[0, b], IND[1, b], IND[2, b])
            Ai[b], Ei[b], Ri[b] = a, e, r
        t_ind = time.time() - t0
        say("[bootstrap] independent-stream pass, B = %d, %.1f s" % (B, t_ind))

        reps_paired = (Ab, Eb, Rb)
        reps_ind = (Ai, Ei, Ri)
        curves_point = (a0, e0, r0_swap)

        # ---- curve summaries and the per-severity-index correlations
        Smean = np.column_stack([Ab.mean(1), Eb.mean(1), Rb.mean(1)])
        Scbrt = np.column_stack([np.cbrt(Ab).mean(1), np.cbrt(Eb).mean(1),
                                 np.cbrt(Rb).mean(1)])
        marg = {}
        for lab, S, pt in (("mean_of_curve", Smean,
                            [a0.mean(), e0.mean(), r0_swap.mean()]),
                           ("mean_of_cube_root_of_curve", Scbrt,
                            [np.cbrt(a0).mean(), np.cbrt(e0).mean(),
                             np.cbrt(r0_swap).mean()])):
            marg[lab] = {
                "point": [float(x) for x in pt],
                "bootstrap_sd": [float(x) for x in S.std(axis=0, ddof=1)],
                "ci95_percentile": [[float(np.percentile(S[:, k], QLO)),
                                     float(np.percentile(S[:, k], QHI))]
                                    for k in range(3)],
                "Corr": [[float(x) for x in row] for row in np.corrcoef(S, rowvar=False)],
            }
        marg["mean_of_curve"]["Corr_mc_se"] = {
            "RGA_RGE": corr_mc_se(Smean[:, 0], Smean[:, 1]),
            "RGA_RGR": corr_mc_se(Smean[:, 0], Smean[:, 2]),
            "RGE_RGR": corr_mc_se(Smean[:, 1], Smean[:, 2])}
        per_index = {"RGA_RGE": [], "RGA_RGR": [], "RGE_RGR": []}
        for t in range(L):
            cols = {"RGA": Ab[:, t], "RGE": Eb[:, t], "RGR": Rb[:, t]}
            for nm, (u, v) in (("RGA_RGE", ("RGA", "RGE")),
                               ("RGA_RGR", ("RGA", "RGR")),
                               ("RGE_RGR", ("RGE", "RGR"))):
                x, yv = cols[u], cols[v]
                if x.std() <= CONST_TOL or yv.std() <= CONST_TOL:
                    per_index[nm].append(None)
                else:
                    per_index[nm].append(float(np.corrcoef(x, yv)[0, 1]))
        marg["per_severity_index_correlation"] = per_index
        marg["mean_per_index_correlation"] = {
            nm: float(np.mean([v for v in vals if v is not None]))
            for nm, vals in per_index.items()}
        # the deep-sweep reading the article's mechanism is about: the second half of
        # the grid, where the tail swap actually bites
        deep = {}
        for nm, vals in per_index.items():
            second = [v for v in vals[L // 2:] if v is not None]
            first = [v for v in vals[:L // 2] if v is not None]
            deep[nm] = {"mean_first_half": float(np.mean(first)) if first else None,
                        "mean_second_half": float(np.mean(second)) if second else None}
        marg["half_grid_means"] = deep
        mres["curve_summaries"] = marg
        C = marg["mean_of_curve"]["Corr"]
        se = marg["mean_of_curve"]["Corr_mc_se"]
        say("[summaries] corr of the three curve means: RGA-RGE %+.4f (mc se %.4f)  "
            "RGA-RGR %+.4f (mc se %.4f)  RGE-RGR %+.4f (mc se %.4f)"
            % (C[0][1], se["RGA_RGE"], C[0][2], se["RGA_RGR"], C[1][2], se["RGE_RGR"]))
        say("[summaries] mean correlation at matched severity index: RGA-RGE %+.4f  "
            "RGA-RGR %+.4f  RGE-RGR %+.4f"
            % (marg["mean_per_index_correlation"]["RGA_RGE"],
               marg["mean_per_index_correlation"]["RGA_RGR"],
               marg["mean_per_index_correlation"]["RGE_RGR"]))
        say("[summaries] per-index RGE-RGR: %s"
            % " ".join("na" if v is None else "%+.3f" % v for v in per_index["RGE_RGR"]))
        say("[summaries] per-index RGA-RGR: %s"
            % " ".join("na" if v is None else "%+.3f" % v for v in per_index["RGA_RGR"]))
        say("[summaries] half-grid means RGE-RGR: first %.4f, second %.4f"
            % (deep["RGE_RGR"]["mean_first_half"], deep["RGE_RGR"]["mean_second_half"]))

        mres["last_rga_knot"] = {
            "point_value": float(a0[-1]),
            "max_abs_over_replicates": float(np.max(np.abs(Ab[:, -1]))),
            "note": "the accumulated partial RGA curve ends at zero by construction"}

        # ---- arms
        def s_mean(A, E, R):
            return np.column_stack([A.mean(1), E.mean(1), R.mean(1)])

        def s_cbrt_mean(A, E, R):
            return np.column_stack([np.cbrt(A).mean(1), np.cbrt(E).mean(1),
                                    np.cbrt(R).mean(1)])

        def make_projection_summary(gA, gE, gR):
            def s(A, E, R):
                return np.column_stack([A @ gA, E @ gE, R @ gR])
            return s

        gt_a, gt_e, gt_r = grad_topsis_entries(a0, e0, r0_swap, pis, nis, wvec)

        arms = []
        arms.append(arm_block(
            "arithmetic",
            "Compliance Score, arithmetic mean variant; closed form "
            "(mean RGA + mean RGE + mean RGR)/3",
            v_arith, curves_point, reps_paired, reps_ind,
            summary_fn=s_mean, grad_fn=lambda s: np.array([1 / 3, 1 / 3, 1 / 3]),
            exact_in_summaries=True))
        arms.append(arm_block(
            "geometric",
            "Compliance Score, geometric mean variant; closed form "
            "mean(RGA^1/3) * mean(RGE^1/3) * mean(RGR^1/3)",
            v_geom, curves_point, reps_paired, reps_ind,
            summary_fn=s_cbrt_mean,
            grad_fn=lambda s: np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]]),
            exact_in_summaries=True))
        arms.append(arm_block(
            "rms",
            "Compliance Score, root mean square variant; no closed form, the full "
            "L^3 tensor",
            v_rms, curves_point, reps_paired, reps_ind,
            summary_fn=make_projection_summary(ga, ge, gr),
            grad_fn=lambda s: np.ones(3), exact_in_summaries=False))
        arms.append(arm_block(
            "topsis",
            "TOPSIS closeness coefficient with the hand-built ideal and anti-ideal "
            "vectors of the Kolesnikov MSc thesis sec. 3.2.3, equal weights",
            lambda A, E, R: topsis_cc(A, E, R, pis, nis, wvec),
            curves_point, reps_paired, reps_ind,
            summary_fn=make_projection_summary(gt_a, gt_e, gt_r),
            grad_fn=lambda s: np.ones(3), exact_in_summaries=False))
        arms.append(arm_block(
            "arithmetic_weighted_SENSITIVITY",
            "SENSITIVITY CHECK ONLY and a declared departure from the authors' stated "
            "position that they do not assign weights: w = (0.50, 0.20, 0.30)",
            v_arith_weighted, curves_point, reps_paired, reps_ind,
            summary_fn=s_mean, grad_fn=lambda s: W_SENS.copy(),
            exact_in_summaries=True))
        for nm, fn_ in (("arithmetic_diagonal", v_arith_diag),
                        ("geometric_diagonal", v_geom_diag),
                        ("rms_diagonal", v_rms_diag)):
            arms.append(arm_block(
                nm, "matched-severity diagonal reading, i = j = k only",
                fn_, curves_point, reps_paired, reps_ind))

        mres["arms"] = {}
        say("")
        say("[arms] point value, paired bootstrap interval, and the cross-term cost")
        for arm in arms:
            nm = arm.pop("_name")
            mres["arms"][nm] = arm
            dm = arm["delta_method"]
            say("  %-32s V = %.6f  boot95 = [%.6f, %.6f]  w = %.6f%s"
                % (nm, arm["V_point"], arm["paired_bootstrap"]["ci95_percentile"][0],
                   arm["paired_bootstrap"]["ci95_percentile"][1],
                   arm["paired_bootstrap"]["width"],
                   ("  delta w = %.6f, declared-zero w = %.6f, understated by "
                    "%+.2f%% (mc se %.2f)"
                    % (dm["width_delta_measured"], dm["width_delta_declared_zero"],
                       dm["understatement_of_width_pct"],
                       dm["understatement_mc_se_pct"])) if dm else ""))
            say("  %-32s   empirical (independent streams) width = %.6f, "
                "understated by %+.2f%% (mc se %.2f)"
                % ("", arm["independent_stream_bootstrap"]["width"],
                   arm["empirical_understatement_of_width_pct"],
                   arm["empirical_understatement_mc_se_pct"]))
            if dm:
                Cs = dm["Corr_of_summaries"]
                ct = dm["cross_term_decomposition"]["cross_contributions_to_variance"]
                say("  %-32s   corr(summaries) RGA-RGE %+.4f  RGA-RGR %+.4f  "
                    "RGE-RGR %+.4f | 2 g g Sigma: %+.3e %+.3e %+.3e | cross share "
                    "%+.2f%% (mc se %.2f)"
                    % ("", Cs[0][1], Cs[0][2], Cs[1][2], ct["RGA_RGE"], ct["RGA_RGR"],
                       ct["RGE_RGR"], dm["cross_covariance_share_of_variance_pct"],
                       dm["cross_share_mc_se_pct"]))

        # ---- product space against the matched-severity diagonal
        say("")
        say("[diagonal] product space against the matched-severity diagonal")
        diag = {}
        for kind, fp, fd in (("arithmetic", v_arith, v_arith_diag),
                             ("geometric", v_geom, v_geom_diag),
                             ("rms", v_rms, v_rms_diag)):
            Vp0 = float(fp(a0[None, :], e0[None, :], r0_swap[None, :])[0])
            Vd0 = float(fd(a0[None, :], e0[None, :], r0_swap[None, :])[0])
            Dp = np.asarray(fd(*reps_paired)) - np.asarray(fp(*reps_paired))
            lo, hi = float(np.percentile(Dp, QLO)), float(np.percentile(Dp, QHI))
            diag[kind] = {
                "V_product_space": Vp0, "V_diagonal": Vd0,
                "diagonal_minus_product_space": Vd0 - Vp0,
                "ci95_of_difference_paired_bootstrap": [lo, hi],
                "difference_excludes_zero": bool(lo > 0 or hi < 0),
                "sd_of_difference": float(Dp.std(ddof=1)),
                "ratio_of_difference_to_composite_ci_width":
                    float(abs(Vd0 - Vp0) / mres["arms"][kind]["paired_bootstrap"]["width"])}
            say("  %-10s product = %.6f  diagonal = %.6f  diff = %+.6f  "
                "boot95 of diff = [%+.6f, %+.6f]  excludes zero: %s"
                % (kind, Vp0, Vd0, Vd0 - Vp0, lo, hi,
                   diag[kind]["difference_excludes_zero"]))
        mres["product_space_vs_diagonal"] = diag

        # ---- power-mean ordering
        Va, Vg, Vr = v_arith(*reps_paired), v_geom(*reps_paired), v_rms(*reps_paired)
        ordering = {
            "point": {"geometric": float(v_geom(a0[None], e0[None], r0_swap[None])[0]),
                      "arithmetic": float(v_arith(a0[None], e0[None], r0_swap[None])[0]),
                      "rms": float(v_rms(a0[None], e0[None], r0_swap[None])[0])},
            "replicates_satisfying_geometric_le_arithmetic": int(np.sum(Vg <= Va)),
            "replicates_satisfying_arithmetic_le_rms": int(np.sum(Va <= Vr)),
            "replicates_total": int(B)}
        ordering["point_ordering_holds"] = bool(
            ordering["point"]["geometric"] <= ordering["point"]["arithmetic"]
            <= ordering["point"]["rms"])
        mres["ordering_check"] = ordering
        say("")
        say("[ordering] geometric %.6f <= arithmetic %.6f <= rms %.6f : %s "
            "(replicates: %d/%d and %d/%d)"
            % (ordering["point"]["geometric"], ordering["point"]["arithmetic"],
               ordering["point"]["rms"], ordering["point_ordering_holds"],
               ordering["replicates_satisfying_geometric_le_arithmetic"], B,
               ordering["replicates_satisfying_arithmetic_le_rms"], B))

        # ---- sensitivities: the two length-d truncations, the Gaussian RGR arm,
        #      and TOPSIS at the literal thesis length
        sens = {}
        key_head = "anchor_dropped_L%d" % GRID
        key_tail = "terminal_knot_dropped_L%d" % GRID
        A2, E2, R2 = Ab[:, 1:], Eb[:, 1:], Rb[:, 1:]
        Ai2, Ei2, Ri2 = Ai[:, 1:], Ei[:, 1:], Ri[:, 1:]
        sens[key_head] = {}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            V0 = float(fn_(a0[None, 1:], e0[None, 1:], r0_swap[None, 1:])[0])
            pbk = percentile_block(np.asarray(fn_(A2, E2, R2)))
            pik = percentile_block(np.asarray(fn_(Ai2, Ei2, Ri2)))
            sens[key_head][kind] = {
                "V_point": V0, "paired_bootstrap": pbk,
                "independent_stream_bootstrap": pik,
                "empirical_understatement_of_width_pct":
                    float(100.0 * (1.0 - pik["width"] / pbk["width"])),
                "shift_from_primary": V0 - mres["arms"][kind]["V_point"]}
        A2t, E2t, R2t = Ab[:, :GRID], Eb[:, :GRID], Rb[:, :GRID]
        Ai2t, Ei2t, Ri2t = Ai[:, :GRID], Ei[:, :GRID], Ri[:, :GRID]
        sens[key_tail] = {}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            V0 = float(fn_(a0[None, :GRID], e0[None, :GRID], r0_swap[None, :GRID])[0])
            pbk = percentile_block(np.asarray(fn_(A2t, E2t, R2t)))
            pik = percentile_block(np.asarray(fn_(Ai2t, Ei2t, Ri2t)))
            sens[key_tail][kind] = {
                "V_point": V0, "paired_bootstrap": pbk,
                "independent_stream_bootstrap": pik,
                "empirical_understatement_of_width_pct":
                    float(100.0 * (1.0 - pik["width"] / pbk["width"])),
                "shift_from_primary": V0 - mres["arms"][kind]["V_point"]}
        sens["length_d_truncation_ambiguity"] = {}
        for kind in ("arithmetic", "geometric", "rms"):
            v_head = sens[key_head][kind]["V_point"]
            v_tail = sens[key_tail][kind]["V_point"]
            w_prim = mres["arms"][kind]["paired_bootstrap"]["width"]
            sens["length_d_truncation_ambiguity"][kind] = {
                "V_anchor_dropped": v_head, "V_terminal_knot_dropped": v_tail,
                "spread_between_the_two_readings": v_tail - v_head,
                "spread_as_fraction_of_primary_ci_width":
                    float(abs(v_tail - v_head) / w_prim) if w_prim > 0 else None,
                "shifts_have_opposite_signs": bool(
                    (v_head - mres["arms"][kind]["V_point"])
                    * (v_tail - mres["arms"][kind]["V_point"]) < 0)}
        r0_noise = np.empty(L)
        for t in range(L):
            r0_noise[t] = rgr_score(p_full, p_noise[t], class_order=CLASS_ORDER)
        sens["rgr_scaled_gaussian_noise"] = {
            "note": "RGR rebuilt as a sweep of Gaussian noise at sigma_t = (t/d) column "
                    "standard deviations; the fixed draw of the scalar study sits at "
                    "sigma = 0.5 column sd",
            "curve_point": r0_noise.tolist(), "arms": {}}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            V0 = float(fn_(a0[None], e0[None], r0_noise[None])[0])
            pbk = percentile_block(np.asarray(fn_(Ab, Eb, Rb_noise)))
            sens["rgr_scaled_gaussian_noise"]["arms"][kind] = {
                "V_point": V0, "paired_bootstrap": pbk,
                "shift_from_primary": V0 - mres["arms"][kind]["V_point"]}
        pisG, nisG = topsis_reference(GRID)
        wvecG = np.concatenate([np.full(GRID, 1.0 / 3.0)] * 3)
        ccG_point = float(topsis_cc(a0[None, :GRID], e0[None, :GRID], r0_swap[None, :GRID],
                                    pisG, nisG, wvecG)[0])
        ccG_b = topsis_cc(Ab[:, :GRID], Eb[:, :GRID], Rb[:, :GRID], pisG, nisG, wvecG)
        ccGh_point = float(topsis_cc(a0[None, 1:], e0[None, 1:], r0_swap[None, 1:],
                                     pisG, nisG, wvecG)[0])
        ccGh_b = topsis_cc(Ab[:, 1:], Eb[:, 1:], Rb[:, 1:], pisG, nisG, wvecG)
        sens["topsis_length_d_readings"] = {
            "note": "the thesis reference vectors keep the anchor, so the reading that "
                    "matches them drops the LAST knot; the anchor-dropped reading is "
                    "printed next to it. At d = %d these are length-%d readings, the "
                    "same rule the German-credit run applied at length 20." % (d, GRID),
            "Ci_point_terminal_dropped": ccG_point,
            "paired_bootstrap_terminal_dropped": percentile_block(np.asarray(ccG_b)),
            "Ci_point_anchor_dropped": ccGh_point,
            "paired_bootstrap_anchor_dropped": percentile_block(np.asarray(ccGh_b)),
            "spread_between_the_two_readings": ccG_point - ccGh_point,
            "spread_as_fraction_of_primary_ci_width":
                float(abs(ccG_point - ccGh_point)
                      / mres["arms"]["topsis"]["paired_bootstrap"]["width"])}
        mres["sensitivities"] = sens
        say("")
        say("[sens] anchor dropped, L = %d: arithmetic %+.6f, geometric %+.6f, rms %+.6f"
            % (GRID, sens[key_head]["arithmetic"]["shift_from_primary"],
               sens[key_head]["geometric"]["shift_from_primary"],
               sens[key_head]["rms"]["shift_from_primary"]))
        say("[sens] terminal knot dropped, L = %d: arithmetic %+.6f, geometric %+.6f, "
            "rms %+.6f"
            % (GRID, sens[key_tail]["arithmetic"]["shift_from_primary"],
               sens[key_tail]["geometric"]["shift_from_primary"],
               sens[key_tail]["rms"]["shift_from_primary"]))
        say("[sens] scaled Gaussian noise RGR: arithmetic %+.6f, geometric %+.6f, "
            "rms %+.6f"
            % (sens["rgr_scaled_gaussian_noise"]["arms"]["arithmetic"]["shift_from_primary"],
               sens["rgr_scaled_gaussian_noise"]["arms"]["geometric"]["shift_from_primary"],
               sens["rgr_scaled_gaussian_noise"]["arms"]["rms"]["shift_from_primary"]))
        say("[sens] TOPSIS at length %d: terminal-dropped %.6f, anchor-dropped %.6f, "
            "spread %.6f = %.2f of the primary interval width"
            % (GRID, ccG_point, ccGh_point,
               sens["topsis_length_d_readings"]["spread_between_the_two_readings"],
               sens["topsis_length_d_readings"]["spread_as_fraction_of_primary_ci_width"]))

        # ---- the estimator bridge, six rungs, on this same paired stream
        def cc_(u, v):
            return float(np.corrcoef(u, v)[0, 1])

        rge_cm, rgr_cm = Smean[:, 1], Smean[:, 2]
        Rb_noise_mean = Rb_noise.mean(1)
        bridge_vals = [
            cc_(SC[:, 1], SC[:, 2]),
            cc_(SC_co[:, 0], SC_co[:, 1]),
            cc_(rge_cm, SC_co[:, 1]),
            cc_(SC_co[:, 0], rgr_cm),
            cc_(rge_cm, Rb_noise_mean),
            cc_(rge_cm, rgr_cm),
        ]
        bridge_pairs = [(SC[:, 1], SC[:, 2]), (SC_co[:, 0], SC_co[:, 1]),
                        (rge_cm, SC_co[:, 1]), (SC_co[:, 0], rgr_cm),
                        (rge_cm, Rb_noise_mean), (rge_cm, rgr_cm)]
        bridge_se = [corr_mc_se(u, v) for u, v in bridge_pairs]
        mres["estimator_bridge_RGE_RGR"] = {
            "note": "the same six rungs, in the same order, as verify_cross_terms.py "
                    "on German credit",
            "rungs": [{"rung": i + 1, "construction": BRIDGE_LABELS[i],
                       "correlation": bridge_vals[i], "mc_se": bridge_se[i],
                       "sign_resolved_at_2_mc_se": bool(abs(bridge_vals[i])
                                                        > 2 * bridge_se[i])}
                      for i in range(6)]}
        say("")
        say("[bridge] RGE-RGR correlation, one estimator change at a time:")
        for i in range(6):
            say("      %+.4f  (mc se %.4f)  %s" % (bridge_vals[i], bridge_se[i],
                                                   BRIDGE_LABELS[i]))
        ends = {
            "RGA_RGE_scalars": cc_(SC[:, 0], SC[:, 1]),
            "RGA_RGE_curve_means": cc_(Smean[:, 0], rge_cm),
            "RGA_RGR_scalars": cc_(SC[:, 0], SC[:, 2]),
            "RGA_RGR_curve_means": cc_(Smean[:, 0], rgr_cm)}
        mres["estimator_bridge_RGA_endpoints"] = ends
        say("[bridge] RGA-RGE: scalars %+.4f -> curve means %+.4f"
            % (ends["RGA_RGE_scalars"], ends["RGA_RGE_curve_means"]))
        say("[bridge] RGA-RGR: scalars %+.4f -> curve means %+.4f"
            % (ends["RGA_RGR_scalars"], ends["RGA_RGR_curve_means"]))

        # the deterministic knots cannot be the mechanism
        det_dropped = np.column_stack([Ab[:, :-1].mean(1), Eb[:, 1:].mean(1),
                                       Rb[:, 1:].mean(1)])
        Cd = np.corrcoef(det_dropped, rowvar=False)
        Call = np.corrcoef(Smean, rowvar=False)
        mres["deterministic_knot_invariance"] = {
            "max_abs_corr_difference": float(np.max(np.abs(Cd - Call))),
            "replicate_sd_of_deterministic_knots": {
                "RGA_last": float(Ab[:, -1].std()), "RGE_first": float(Eb[:, 0].std()),
                "RGR_first": float(Rb[:, 0].std())}}
        say("[invariance] max |corr(all %d knots) - corr(deterministic knot deleted)| "
            "= %.2e" % (L, mres["deterministic_knot_invariance"]["max_abs_corr_difference"]))

        # ---- conditioning check, sense (b): full re-search of the greedy order
        bf = B_FULL[tag]
        say("")
        say("[conditioning] reduced-B check, B_full = %d (%s), on the same index stream"
            % (bf, tag))
        t0 = time.time()
        Af = np.empty((bf, L)); Ef = np.empty((bf, L)); Rf = np.empty((bf, L))
        for b in range(bf):
            ib = IDX[b]
            Xb, yb = X_te[ib], y_te[ib]
            pb = model.predict_proba(Xb)[:, 1]
            Af[b] = rga_vector(yb, pb, GRID)
            eb, _ = rge_vector_package(model, Xb, names, GRID, "mean")
            Ef[b] = eb
            for t in range(L):
                Rf[b, t] = rgr_score(pb, model.predict_proba(tail_swap(Xb, ps[t]))[:, 1],
                                     class_order=CLASS_ORDER)
            if (b + 1) % 50 == 0 or (bf <= 30 and (b + 1) % 10 == 0):
                say("  [conditioning] %d/%d replicates, %.0f s elapsed"
                    % (b + 1, bf, time.time() - t0))
        t_full = time.time() - t0
        cond = {"B_full": bf, "gross_check_only": bool(bf < 100), "seconds": t_full,
                "seconds_per_replicate": t_full / bf, "arms": {}}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            Vfroz = np.asarray(fn_(Ab[:bf], Eb[:bf], Rb[:bf]))
            Vfull = np.asarray(fn_(Af, Ef, Rf))
            rng_c = np.random.default_rng(MC_SEED + 2)
            ratios = np.empty(MC_COND)
            for m_ in range(MC_COND):
                pick = rng_c.integers(0, bf, bf)
                sd_f = Vfroz[pick].std(ddof=1)
                ratios[m_] = (Vfull[pick].std(ddof=1) / sd_f) if sd_f > 0 else np.nan
            ratios = ratios[np.isfinite(ratios)]
            band = [float(np.percentile(ratios, QLO)), float(np.percentile(ratios, QHI))]
            cond["arms"][kind] = {
                "mean_frozen": float(Vfroz.mean()), "mean_full": float(Vfull.mean()),
                "sd_frozen": float(Vfroz.std(ddof=1)), "sd_full": float(Vfull.std(ddof=1)),
                "max_abs_paired_difference": float(np.max(np.abs(Vfull - Vfroz))),
                "mean_abs_paired_difference": float(np.mean(np.abs(Vfull - Vfroz))),
                "sd_ratio_full_over_frozen": float(Vfull.std(ddof=1) / Vfroz.std(ddof=1)),
                "sd_ratio_mc_band_95": band}
            c = cond["arms"][kind]
            say("  %-10s sd frozen %.6f  sd full %.6f  ratio %.3f  mc band [%.3f, %.3f]  "
                "mean |paired diff| %.6f"
                % (kind, c["sd_frozen"], c["sd_full"], c["sd_ratio_full_over_frozen"],
                   band[0], band[1], c["mean_abs_paired_difference"]))
        say("  cost: %.1f s for %d replicates (%.2f s a replicate)"
            % (t_full, bf, t_full / bf))
        mres["conditioning_check_curve_recompute"] = cond

        mres["curves_point"] = {
            "RGA_partial": a0.tolist(),
            "RGE_greedy_mean_baseline": e0.tolist(),
            "RGE_greedy_removal_order": [names[i] for i in order],
            "RGR_tail_swap": r0_swap.tolist(),
            "RGR_tail_swap_p_grid": ps.tolist(),
            "RGR_scaled_gaussian_noise": r0_noise.tolist()}
        mres["timings_s"] = {
            "point_curves": t_point, "precompute_curve": t_pre,
            "precompute_scalar": t_pre2, "paired_pass": t_paired,
            "independent_stream_pass": t_ind, "conditioning_check": t_full}
        results["models"][tag] = mres
        KEEP[tag] = {"point": np.concatenate([a0, e0, r0_swap]), "paired": (Ab, Eb, Rb)}
        STORE[tag] = {"Ab": Ab, "Eb": Eb, "Rb": Rb, "Rb_noise": Rb_noise,
                      "SC": SC, "SC_co": SC_co, "a0": a0, "e0": e0, "r0": r0_swap,
                      "p_full": p_full, "p_mask_single": p_mask_single,
                      "p_pert_fixed": p_pert_fixed, "col_std": col_std}

    # ---------------------------------------------------------------- TOPSIS step 2
    say("")
    say("[topsis-norm] thesis step 2 (column normalisation), measured")
    alt_sets = {"models_and_references": [KEEP[t]["point"] for t in TAGS] + [pis, nis],
                "models_only": [KEEP[t]["point"] for t in TAGS]}
    for t in TAGS:
        alt_sets["one_model_%s_and_references" % t] = [KEEP[t]["point"], pis, nis]
    tn = {"note": "column norms are held fixed at the point-estimate decision matrix",
          "alternative_sets": {}}
    for nm, alts in alt_sets.items():
        Mx = np.vstack(alts)
        nrm = np.sqrt((Mx ** 2).sum(axis=0))
        entry = {"n_alternatives": int(Mx.shape[0]), "min_column_norm": float(nrm.min()),
                 "n_columns_with_norm_below_1e-12": int((nrm < CONST_TOL).sum()),
                 "models": {}}
        safe = np.where(nrm > 0, nrm, 1.0)
        pn, qn = pis / safe, nis / safe
        for t in TAGS:
            ccn = float(topsis_cc((KEEP[t]["point"][:L] / safe[:L])[None, :],
                                  (KEEP[t]["point"][L:2 * L] / safe[L:2 * L])[None, :],
                                  (KEEP[t]["point"][2 * L:] / safe[2 * L:])[None, :],
                                  pn, qn, wvec)[0])
            base = results["models"][t]["arms"]["topsis"]["V_point"]
            w_t = results["models"][t]["arms"]["topsis"]["paired_bootstrap"]["width"]
            entry["models"][t] = {"Ci_normalised": ccn, "Ci_unnormalised": base,
                                  "shift": ccn - base,
                                  "shift_as_fraction_of_ci_width": float(abs(ccn - base) / w_t)}
        tn["alternative_sets"][nm] = entry
        say("  %-34s alternatives=%d  min column norm %.2e  %s"
            % (nm, entry["n_alternatives"], entry["min_column_norm"],
               "  ".join("%s %.6f (%+.6f)" % (t, entry["models"][t]["Ci_normalised"],
                                              entry["models"][t]["shift"]) for t in TAGS)))
    results["topsis_column_normalisation_sensitivity"] = tn

    # ---------------------------------------------------------------- scalar driver
    # redraw_experiment.py on Taiwan: scheme A (the fixed draw, already computed as SC
    # on the same stream) against scheme B (a fresh perturbation per replicate), then
    # the two-link chain under both schemes.
    say("")
    say("#" * 78)
    say("### the scalar construction: two conditioning schemes and the two-link chain")
    say("#" * 78)
    METRICS = ["RGA", "RGE", "RGR"]
    t0 = time.time()
    p_pert_redraw = {}
    for tag in TAGS:
        model = models[tag]
        mat = np.empty((B, n))
        base = REDRAW_BASE[tag]
        col_std = STORE[tag]["col_std"]
        for b in range(B):
            rb = np.random.default_rng(base + b)
            pert_b = rb.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
            mat[b] = model.predict_proba(X_te + pert_b)[:, 1]
        p_pert_redraw[tag] = mat
        say("[redraw] %s: %d fresh perturbation draws + predictions, %.1f s cumulative"
            % (tag, B, time.time() - t0))

    results["scalar_construction"] = {
        "redraw_seed_rule": "per-replicate seed = REDRAW_BASE[model] + b; REDRAW_BASE = %s"
                            % REDRAW_BASE,
        "models": {}}
    for tag in TAGS:
        SC = STORE[tag]["SC"]
        p_full = STORE[tag]["p_full"]
        paired_A = SC
        paired_Bs = np.empty((B, 3))
        for b in range(B):
            ib = IDX[b]
            rgr_b = rgr_score(p_full[ib], p_pert_redraw[tag][b][ib])
            paired_Bs[b] = [SC[b, 0], SC[b, 1], rgr_b]
        point = np.array([rga_score(y_te, p_full),
                          float(np.mean([rge_score(p_full, STORE[tag]["p_mask_single"][j])
                                         for j in range(d)])),
                          rgr_score(p_full, STORE[tag]["p_pert_fixed"])])
        CA_ = np.array([geo_mean(v) for v in paired_A])
        CB_ = np.array([geo_mean(v) for v in paired_Bs])
        results["scalar_construction"]["models"][tag] = {
            "point_fixed_draw": dict(zip(METRICS + ["C_geomean"],
                                         [float(v) for v in point] + [geo_mean(point)])),
            "fixed_draw": {
                "Corr": corr_dict(paired_A),
                "sd": {m: float(paired_A[:, k].std(ddof=1)) for k, m in enumerate(METRICS)},
                "ci_C_paired_percentile": [float(np.percentile(CA_, QLO)),
                                           float(np.percentile(CA_, QHI))],
                "sd_C_paired": float(CA_.std(ddof=1))},
            "redraw": {
                "Corr": corr_dict(paired_Bs),
                "sd": {m: float(paired_Bs[:, k].std(ddof=1)) for k, m in enumerate(METRICS)},
                "ci_C_paired_percentile": [float(np.percentile(CB_, QLO)),
                                           float(np.percentile(CB_, QHI))],
                "sd_C_paired": float(CB_.std(ddof=1)),
                "mean_RGR": float(paired_Bs[:, 2].mean())},
        }
        cA = results["scalar_construction"]["models"][tag]["fixed_draw"]["Corr"]
        cB = results["scalar_construction"]["models"][tag]["redraw"]["Corr"]
        say("[%s] fixed-draw corr: RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f"
            % (tag, cA["corr_RGA_RGE"], cA["corr_RGA_RGR"], cA["corr_RGE_RGR"]))
        say("[%s] redraw     corr: RGA-RGE %+.4f  RGA-RGR %+.4f  RGE-RGR %+.4f"
            % (tag, cB["corr_RGA_RGE"], cB["corr_RGA_RGR"], cB["corr_RGE_RGR"]))
        STORE[tag]["point_scalar"] = point

    # ---- the chain, shared index stream, both schemes
    t0 = time.time()
    shared_rng = np.random.default_rng(SEED + 99)
    CA_fix = np.empty(B); CB_fix = np.empty(B)
    CA_red = np.empty(B); CB_red = np.empty(B)
    for b in range(B):
        idx = shared_rng.integers(0, n, n)
        for tag, arr_fix, arr_red in (("logit", CA_fix, CA_red), ("rf", CB_fix, CB_red)):
            p_full = STORE[tag]["p_full"]
            pb = p_full[idx]
            vec = np.array([rga_score(y_te[idx], pb),
                            float(np.mean([rge_score(pb, STORE[tag]["p_mask_single"][j][idx])
                                           for j in range(d)])),
                            rgr_score(pb, STORE[tag]["p_pert_fixed"][idx])])
            arr_fix[b] = geo_mean(vec)
            rgr_b = rgr_score(pb, p_pert_redraw[tag][b][idx])
            arr_red[b] = geo_mean([vec[0], vec[1], rgr_b])
        if (b + 1) % 500 == 0:
            say("  [chain] %d/%d replicates, %.0f s elapsed" % (b + 1, B, time.time() - t0))

    def chain_block(CA, CB, point_A, point_B, seed):
        boot_A, boot_B = float(CA.mean()), float(CB.mean())
        S = np.cov(np.vstack([CA, CB]))
        rho = float(np.corrcoef(CA, CB)[0, 1])
        h_point = float(point_A * point_B)
        grad = np.array([point_B, point_A])
        var_measured = float(grad @ S @ grad)
        var_declared0 = float(grad[0] ** 2 * S[0, 0] + grad[1] ** 2 * S[1, 1])
        ci_meas = [h_point - Z * np.sqrt(var_measured), h_point + Z * np.sqrt(var_measured)]
        ci_decl = [h_point - Z * np.sqrt(var_declared0), h_point + Z * np.sqrt(var_declared0)]
        w_meas, w_decl = ci_meas[1] - ci_meas[0], ci_decl[1] - ci_decl[0]
        # Monte Carlo band on the understatement and on rho, by the same second-level
        # resampling block the pinned driver applies to the width shares
        rng = np.random.default_rng(seed)
        us, rs = [], []
        for _ in range(MC_R):
            pick = rng.integers(0, len(CA), len(CA))
            Sr = np.cov(np.vstack([CA[pick], CB[pick]]))
            vm = float(grad @ Sr @ grad)
            vd = float(grad[0] ** 2 * Sr[0, 0] + grad[1] ** 2 * Sr[1, 1])
            if vm > 0 and vd > 0:
                us.append(100.0 * (1.0 - math.sqrt(vd) / math.sqrt(vm)))
            rs.append(float(np.corrcoef(CA[pick], CB[pick])[0, 1]))
        return {
            "estimator": "plug-in product of the two full-sample link scores",
            "link_point_values": {"A": float(point_A), "B": float(point_B)},
            "cross_link_correlation": rho, "cross_link_correlation_mc_se": mc_std(rs),
            "link_covariance": S.tolist(), "h_point": h_point,
            "bootstrap_link_means": {"A": boot_A, "B": boot_B},
            "ci95_measured_covariance": [float(v) for v in ci_meas],
            "ci95_cross_term_declared_zero": [float(v) for v in ci_decl],
            "width_measured": float(w_meas), "width_declared_zero": float(w_decl),
            "understatement_of_width_pct": float(100 * (1 - w_decl / w_meas)),
            "understatement_mc_se_pct": mc_std(us),
        }

    point_A = geo_mean(STORE["logit"]["point_scalar"])
    point_B = geo_mean(STORE["rf"]["point_scalar"])
    results["chain"] = {
        "links": {"A": "logit", "B": "rf"},
        "chain_functional": "h = C_A * C_B (both-must-hold), C the geometric mean of "
                            "the three SCALAR metrics",
        "fixed_draw": chain_block(CA_fix, CB_fix, point_A, point_B, MC_SEED_CHAIN),
        "redraw": chain_block(CA_red, CB_red, point_A, point_B, MC_SEED_CHAIN + 1),
    }
    for scheme in ("fixed_draw", "redraw"):
        ch = results["chain"][scheme]
        say("[chain/%s] rho = %+.4f (mc se %.4f)  h = %.4f  measured width %.6f  "
            "declared-zero width %.6f  understatement %+.2f%% (mc se %.2f)"
            % (scheme, ch["cross_link_correlation"], ch["cross_link_correlation_mc_se"],
               ch["h_point"], ch["width_measured"], ch["width_declared_zero"],
               ch["understatement_of_width_pct"], ch["understatement_mc_se_pct"]))
    say("[chain] pass in %.1f s" % (time.time() - t0))

    # ---------------------------------------------------------------- coding sensitivity
    # Pre-registered in section 4.3 and run UNCONDITIONALLY, so that it cannot serve as
    # a post-hoc rescue. Both models are refitted on y' = 1 - y and the point curves,
    # the curve-summary covariance and one paired pass are recomputed.
    say("")
    say("#" * 78)
    say("### pre-registered coding sensitivity: y' = 1 - y (y' = 1 is now 'no default')")
    say("#" * 78)
    t_cs = time.time()
    y_tr_f, y_te_f = 1 - y_tr, 1 - y_te
    models_f = {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for m in models_f.values():
        m.fit(X_tr, y_tr_f)
    cs = {"note": "models refitted on y' = 1 - y; the same index stream "
                  "default_rng(SEED + 1) and the same B", "models": {}}
    for tag in TAGS:
        model = models_f[tag]
        p_full = model.predict_proba(X_te)[:, 1]
        col_mean = X_te.mean(axis=0)
        a0f = rga_vector(y_te_f, p_full, GRID)
        e0f, orderf = rge_vector_package(model, X_te, names, GRID, "mean")
        r0f = np.empty(L)
        for t in range(L):
            r0f[t] = rgr_score(p_full, model.predict_proba(tail_swap(X_te, ps[t]))[:, 1],
                               class_order=CLASS_ORDER)
        p_mask_f = np.empty((L, n)); p_mask_f[0] = p_full
        for t in range(1, L):
            Xm = X_te.copy()
            cols = orderf[:t]
            Xm[:, cols] = col_mean[cols]
            p_mask_f[t] = model.predict_proba(Xm)[:, 1]
        p_swap_f = np.empty((L, n))
        for t in range(L):
            p_swap_f[t] = model.predict_proba(tail_swap(X_te, ps[t]))[:, 1]
        p_mask_single_f = np.empty((d, n))
        for j in range(d):
            Xm = X_te.copy()
            Xm[:, j] = col_mean[j]
            p_mask_single_f[j] = model.predict_proba(Xm)[:, 1]
        Af = np.empty((B, L)); Ef = np.empty((B, L)); Rf = np.empty((B, L))
        SCf = np.empty((B, 2))
        for b in range(B):
            ib = IDX[b]
            pb = p_full[ib]
            Af[b] = rga_vector(y_te_f[ib], pb, GRID)
            Ef[b, 0] = 1.0
            for t in range(1, L):
                Ef[b, t] = rge_score(pb, p_mask_f[t][ib], class_order=CLASS_ORDER)
            for t in range(L):
                Rf[b, t] = rgr_score(pb, p_swap_f[t][ib], class_order=CLASS_ORDER)
            SCf[b, 0] = np.mean([rge_score(pb, p_mask_single_f[j][ib],
                                           class_order=CLASS_ORDER) for j in range(d)])
        Sm = np.column_stack([Af.mean(1), Ef.mean(1), Rf.mean(1)])
        Cf = np.corrcoef(Sm, rowvar=False)
        rung4 = float(np.corrcoef(SCf[:, 0], Sm[:, 2])[0, 1])
        rung6 = float(Cf[1, 2])
        per_index_f = []
        for t in range(L):
            u, v = Ef[:, t], Rf[:, t]
            per_index_f.append(None if (u.std() <= CONST_TOL or v.std() <= CONST_TOL)
                               else float(np.corrcoef(u, v)[0, 1]))
        cs["models"][tag] = {
            "corr_curve_means": {"RGA_RGE": float(Cf[0][1]), "RGA_RGR": float(Cf[0][2]),
                                 "RGE_RGR": rung6},
            "corr_curve_means_mc_se": {"RGE_RGR": corr_mc_se(Sm[:, 1], Sm[:, 2]),
                                       "RGA_RGR": corr_mc_se(Sm[:, 0], Sm[:, 2])},
            "bridge_rung4_scalarRGE_x_curvemeanRGR": rung4,
            "bridge_rung4_mc_se": corr_mc_se(SCf[:, 0], Sm[:, 2]),
            "bridge_rung6_curvemeans": rung6,
            "per_severity_index_correlation_RGE_RGR": per_index_f,
            "curves_point": {"RGA": a0f.tolist(), "RGE": e0f.tolist(),
                             "RGR_tail_swap": r0f.tolist()},
            "V_point": {k: tensor_mean(a0f, e0f, r0f, k)
                        for k in ("arithmetic", "geometric", "rms")},
        }
        say("[coding] %s under y' = 1 - y: curve-mean RGE-RGR %+.4f (primary %+.4f), "
            "RGA-RGR %+.4f (primary %+.4f), rung 4 %+.4f (primary %+.4f)"
            % (tag, rung6,
               results["models"][tag]["curve_summaries"]["mean_of_curve"]["Corr"][1][2],
               float(Cf[0][2]),
               results["models"][tag]["curve_summaries"]["mean_of_curve"]["Corr"][0][2],
               rung4,
               results["models"][tag]["estimator_bridge_RGE_RGR"]["rungs"][3]["correlation"]))
    cs["seconds"] = time.time() - t_cs
    results["coding_sensitivity"] = cs
    say("[coding] pass in %.1f s" % cs["seconds"])

    # ---------------------------------------------------------------- the verdict
    say("")
    say("#" * 78)
    say("### the pre-registered verdict, applied")
    say("#" * 78)

    def rung(tag, i):
        return results["models"][tag]["estimator_bridge_RGE_RGR"]["rungs"][i - 1]

    dm_und = {t: {k: results["models"][t]["arms"][k]["delta_method"]
                  ["understatement_of_width_pct"]
                  for k in ("arithmetic", "geometric", "rms", "topsis")} for t in TAGS}
    dm_se = {t: {k: results["models"][t]["arms"][k]["delta_method"]
                 ["understatement_mc_se_pct"]
                 for k in ("arithmetic", "geometric", "rms", "topsis")} for t in TAGS}
    vol = ("arithmetic", "geometric", "rms")

    unresolved = []
    for t in TAGS:
        for i in (4, 5, 6):
            r_ = rung(t, i)
            if not r_["sign_resolved_at_2_mc_se"]:
                unresolved.append("bridge rung %d on %s (%.4f, mc se %.4f)"
                                  % (i, t, r_["correlation"], r_["mc_se"]))
    for s in ("fixed_draw", "redraw"):
        ch = results["chain"][s]
        if abs(ch["understatement_of_width_pct"]) <= 2 * ch["understatement_mc_se_pct"]:
            unresolved.append("chain understatement under %s (%.2f, mc se %.2f)"
                              % (s, ch["understatement_of_width_pct"],
                                 ch["understatement_mc_se_pct"]))
        if abs(ch["cross_link_correlation"]) <= 2 * ch["cross_link_correlation_mc_se"]:
            unresolved.append("chain rho under %s" % s)

    cond_a = all(rung(t, 4)["correlation"] < 0 for t in TAGS)
    cond_b = all(rung(t, 5)["correlation"] > 0 for t in TAGS)
    cond_c = all(rung(t, 6)["correlation"] < 0 for t in TAGS)
    cond_d = all(abs(dm_und[t][k]) <= 6.0 for t in TAGS for k in vol)
    cond_e = all(results["chain"][s]["understatement_of_width_pct"] >= 10.0
                 and results["chain"][s]["cross_link_correlation"] > 0
                 for s in ("fixed_draw", "redraw"))
    fail_a = all(rung(t, 4)["correlation"] > 0 for t in TAGS)
    fail_b = all(rung(t, 6)["correlation"] > 0 for t in TAGS)
    fail_c = all(max(dm_und[t][k] for k in vol) > 15.0 for t in TAGS)
    fail_d = all((results["chain"][s]["understatement_of_width_pct"] < 5.0)
                 for s in ("fixed_draw", "redraw"))
    disagree4 = (np.sign(rung("logit", 4)["correlation"])
                 != np.sign(rung("rf", 4)["correlation"]))
    disagree6 = (np.sign(rung("logit", 6)["correlation"])
                 != np.sign(rung("rf", 6)["correlation"]))
    coding_flip = any(
        np.sign(results["coding_sensitivity"]["models"][t]["bridge_rung6_curvemeans"])
        != np.sign(results["models"][t]["curve_summaries"]["mean_of_curve"]["Corr"][1][2])
        for t in TAGS)
    coding_flip4 = any(
        np.sign(results["coding_sensitivity"]["models"][t]
                ["bridge_rung4_scalarRGE_x_curvemeanRGR"])
        != np.sign(rung(t, 4)["correlation"]) for t in TAGS)

    if unresolved or disagree4 or disagree6 or coding_flip or coding_flip4:
        verdict = "INCONCLUSIVE"
    elif cond_a and cond_b and cond_c and cond_d and cond_e:
        verdict = "GENERALISES"
    elif fail_a or fail_b or fail_c or fail_d:
        verdict = "FAILS TO GENERALISE"
    else:
        verdict = "INCONCLUSIVE"

    results["verdict"] = {
        "verdict": verdict,
        "rule_source": "PRE-REGISTRATION.md section 7, applied without modification",
        "generalises_conditions": {
            "(a) rung 4 negative on both models": bool(cond_a),
            "(b) rung 5 positive on both models": bool(cond_b),
            "(c) rung 6 negative on both models": bool(cond_c),
            "(d) delta understatement of the three volume variants within 6 pp of zero "
            "on both models": bool(cond_d),
            "(e) chain understatement >= 10 pp and rho > 0 under both schemes": bool(cond_e),
        },
        "failure_conditions": {
            "(a') rung 4 positive on both models": bool(fail_a),
            "(b') rung 6 positive on both models": bool(fail_b),
            "(c') volume delta understatement exceeds +15 pp on both models "
            "(read as: the largest of the three variants exceeds +15 on each model)":
                bool(fail_c),
            "(d') chain understatement below 5 pp under both schemes": bool(fail_d),
        },
        "inconclusive_triggers": {
            "quantities_inside_two_mc_standard_errors_of_zero": unresolved,
            "models_disagree_in_sign_on_rung_4": bool(disagree4),
            "models_disagree_in_sign_on_rung_6": bool(disagree6),
            "coding_sensitivity_flips_rung_6": bool(coding_flip),
            "coding_sensitivity_flips_rung_4": bool(coding_flip4),
        },
        "quantities": {
            "bridge": {t: [rung(t, i)["correlation"] for i in range(1, 7)] for t in TAGS},
            "bridge_mc_se": {t: [rung(t, i)["mc_se"] for i in range(1, 7)] for t in TAGS},
            "delta_understatement_pct": dm_und,
            "delta_understatement_mc_se_pct": dm_se,
            "chain": {s: {"rho": results["chain"][s]["cross_link_correlation"],
                          "understatement_pct":
                              results["chain"][s]["understatement_of_width_pct"]}
                      for s in ("fixed_draw", "redraw")},
        },
    }
    say("[verdict] %s" % verdict)
    for k, v in results["verdict"]["generalises_conditions"].items():
        say("  generalises %-95s %s" % (k, v))
    for k, v in results["verdict"]["failure_conditions"].items():
        say("  failure     %-95s %s" % (k, v))
    for k, v in results["verdict"]["inconclusive_triggers"].items():
        say("  inconclusive %-94s %s" % (k, v))

    # ---------------------------------------------------------------- comparison
    comp = {"note": "German credit beside Taiwan, quantity by quantity, on the "
                    "pre-registered list only", "rows": []}
    for t in TAGS:
        comp["rows"].append({
            "quantity": "P1 corr(curve-mean RGE, curve-mean RGR), %s" % t,
            "german": german["models"][t]["corr_curve_means"]["RGE_RGR"],
            "taiwan": results["models"][t]["curve_summaries"]["mean_of_curve"]["Corr"][1][2],
            "taiwan_mc_se":
                results["models"][t]["curve_summaries"]["mean_of_curve"]["Corr_mc_se"]["RGE_RGR"]})
        comp["rows"].append({
            "quantity": "P2 corr(curve-mean RGA, curve-mean RGR), %s" % t,
            "german": german["models"][t]["corr_curve_means"]["RGA_RGR"],
            "taiwan": results["models"][t]["curve_summaries"]["mean_of_curve"]["Corr"][0][2],
            "taiwan_mc_se":
                results["models"][t]["curve_summaries"]["mean_of_curve"]["Corr_mc_se"]["RGA_RGR"]})
        for k in ("arithmetic", "geometric", "rms", "topsis"):
            comp["rows"].append({
                "quantity": "P3 delta understatement %% %s, %s" % (k, t),
                "german": german["models"][t]["delta_understatement_pct"][k],
                "taiwan": dm_und[t][k], "taiwan_mc_se": dm_se[t][k]})
            comp["rows"].append({
                "quantity": "P4 empirical understatement %% %s, %s" % (k, t),
                "german": german["models"][t]["empirical_understatement_pct"][k],
                "taiwan": results["models"][t]["arms"][k]
                          ["empirical_understatement_of_width_pct"],
                "taiwan_mc_se": results["models"][t]["arms"][k]
                                ["empirical_understatement_mc_se_pct"]})
        for i in range(6):
            comp["rows"].append({
                "quantity": "P6 bridge rung %d, %s" % (i + 1, t),
                "german": german["bridge_RGE_RGR"][t][i],
                "taiwan": rung(t, i + 1)["correlation"],
                "taiwan_mc_se": rung(t, i + 1)["mc_se"]})
    for s in ("fixed_draw", "redraw"):
        comp["rows"].append({"quantity": "P5 chain rho, %s" % s,
                             "german": german["chain"][s]["rho"],
                             "taiwan": results["chain"][s]["cross_link_correlation"],
                             "taiwan_mc_se": results["chain"][s]["cross_link_correlation_mc_se"]})
        comp["rows"].append({"quantity": "P5 chain understatement %%, %s" % s,
                             "german": german["chain"][s]["understatement_pct"],
                             "taiwan": results["chain"][s]["understatement_of_width_pct"],
                             "taiwan_mc_se": results["chain"][s]["understatement_mc_se_pct"]})
    results["comparison"] = comp
    say("")
    say("[compare] German credit beside Taiwan, pre-registered quantities only")
    say("  %-52s %12s %12s %10s" % ("quantity", "German", "Taiwan", "mc se"))
    for r in comp["rows"]:
        say("  %-52s %12.4f %12.4f %10.4f"
            % (r["quantity"], r["german"], r["taiwan"],
               r["taiwan_mc_se"] if r["taiwan_mc_se"] == r["taiwan_mc_se"] else float("nan")))

    # ---------------------------------------------------------------- persistence
    npz_path = os.path.join(HERE, "replicates-second-dataset%s.npz" % TAG_OUT)
    np.savez_compressed(
        npz_path,
        **{("%s_%s" % (t, k)): STORE[t][k]
           for t in TAGS for k in ("Ab", "Eb", "Rb", "Rb_noise", "SC", "SC_co",
                                   "a0", "e0", "r0")},
        chain_C_fixed_A=CA_fix, chain_C_fixed_B=CB_fix,
        chain_C_redraw_A=CA_red, chain_C_redraw_B=CB_red,
        y_test=y_te)
    say("")
    say("[replicates] %s" % npz_path)

    results["total_runtime_s"] = time.time() - t_start
    say("[done] total runtime %.1f s (%.1f min)"
        % (results["total_runtime_s"], results["total_runtime_s"] / 60.0))

    with open(os.path.join(HERE, "results-second-dataset%s.json" % TAG_OUT), "w") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(HERE, "run-second-dataset%s.log" % TAG_OUT), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
