#!/usr/bin/env python3
"""
An interval for the Pavia Compliance Score, computed on the construction that
defines it.

The Compliance Score of Giudici and Kolesnikov ("Integrating Safe AI Metrics",
SSRN 10.2139/ssrn.5362574, eq. (13)-(17); Kolesnikov MSc thesis, eq.
(2.13)-(2.17)) does not consume three scalars. It consumes the three VECTORS
from which the RGA, RGE and RGR curves are built. For every triplet of indices
(i, j, k) a mean of (RGA[i], RGE[j], RGR[k]) is taken -- arithmetic, geometric,
or root mean square -- filling an L x L x L tensor M, and the score is the
average height of that tensor,

    V = (1 / L^3) * sum_i sum_j sum_k M(i, j, k).

This study puts a confidence interval on V, in all three variants, on their own
construction. It also reproduces the TOPSIS case of the thesis, which does not
appear in the published article at all, with the hand-built ideal and anti-ideal
vectors that Kolesnikov specifies.

Substrate: safeai (github.com/koleso500/safeai) at pinned commit 39768fc, the
vendored clone and inert import stubs of the 2026-07-24 redraw run, reused
READ-ONLY. This script never clones, never writes into the substrate, and never
touches the deposited experiment directories.

Data, seed and split are inherited from the deposited runs so that every number
here is commensurable with the published ones.

Anton Sokolov, Tyche Institute, Tallinn. 25 July 2026.
"""
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

from safeai.rga import rga_curve, rga_score
from safeai.rge import rge_curve, rge_score
from safeai.rgr import rgr_score

# ---------------------------------------------------------------- constants
SEED = 20260723                 # identical to the deposited experiment and the redraw
TEST_SIZE = 0.3
ALPHA = 0.05
Z = 1.959963984540054           # two-sided 95%, as hardcoded in both deposited runs
QLO, QHI = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)

B = int(os.environ.get("PAVIA_B", "2000"))          # primary bootstrap size
# Conditioning check, reduced B, per model. A replicate of this arm costs a fraction
# of a second on the logistic model and tens of seconds on the forest, so the two
# cannot carry the same B. The sd ratio the check reports is itself a noisy quantity
# at small B, so the logistic arm is run at a size where its Monte Carlo band is
# informative, and the forest arm is left at 30 and labelled a gross check. The band
# of each arm is measured (MC_COND paired resamples of the replicate pairs) and
# reported next to the ratio it qualifies.
B_FULL = {"logit": int(os.environ.get("PAVIA_B_FULL_LOGIT", "300")),
          "rf": int(os.environ.get("PAVIA_B_FULL_RF", "30"))}
MC_COND = 2000                  # resamples for the Monte Carlo band on that ratio
CONST_TOL = 1e-12               # below this a replicate column is structurally fixed
CLASS_ORDER = np.array([0, 1])
TAGS = ["logit", "rf"]
CURVES = ["RGA", "RGE", "RGR"]

# One weighted arm only, and it is a declared departure from their stated position.
# Article, immediately after eq. (17): "We remark that our perspective is agnostic,
# that is, we do not assign weights to the different three dimensions, reflecting
# different risk appetites. Rather, we consider all possible combinations of
# perturbation types and take simple averages the performance of each model in the
# three dimensions." The profile below is the accuracy-led profile of the companion
# aggregation study, carried over for continuity and for no other reason.
W_SENS = np.array([0.50, 0.20, 0.30])

# Continuity anchors, read once from the deposited results.json of the published
# experiment and hardcoded here so the check is against the deposit, not against a
# rerun of it. These are the scalar RGA point estimates of the deposited run.
DEPOSITED_RGA = {"logit": 0.8052154195011337, "rf": 0.8272108843537415}

LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def f(x, p=6):
    return float(np.round(float(x), p))


# ---------------------------------------------------------------- data and models
def load_data():
    ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "good").astype(int)
    return X, y, list(ds.feature_names), "statlog-german-credit (OpenML credit-g v1)"


# ---------------------------------------------------------------- the three curves
def rga_vector(y_true, p_full, grid):
    """Article eq. (5)-(6): partial RGA contributions accumulated in reverse, from
    the full RGA down to zero. curve_method='partial' must be asked for by name;
    'auto' resolves to 'removal' in this build despite the module docstring."""
    res = rga_curve(y_true, p_full, curve_method="partial", n_segments=grid,
                    normalize_to_perfect=False)
    return np.asarray(res["curve"], dtype=float)


def rge_vector_package(model, X, feature_names, grid, baseline="mean"):
    """Article eq. (12) and the thesis AURGE procedure: greedy monotone removal.
    Returns the curve and the greedy removal order (as column indices)."""
    res = rge_curve(model, X, method="tabular", feature_names=feature_names,
                    masking_method="greedy", baseline=baseline, n_steps=grid,
                    verbose=False)
    order = [feature_names.index(nm) for nm in res["removed_features"]]
    return np.asarray(res["rge_scores"], dtype=float), order


def tail_swap(X, p):
    """The article's own perturbation, eq. (9) and the text around it: for each
    p in [0, 0.5] the lowest p-quantile of observations is swapped with the highest
    p-quantile, column by column. Rank i exchanges with rank n+1-i for the
    outermost floor(p*n) ranks on each side; at p = 0.5 the column is reversed.
    The pinned package ships no tail-swap, so this is rebuilt on top of its
    published rgr_score."""
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


# ---------------------------------------------------------------- the composite
def tensor_mean(a, e, r, kind):
    """The composite exactly as written: fill the L x L x L tensor and average it."""
    A = a[:, None, None]
    E = e[None, :, None]
    R = r[None, None, :]
    if kind == "arithmetic":
        M = (A + E + R) / 3.0
    elif kind == "geometric":
        M = np.cbrt(A * E * R)
    elif kind == "rms":
        M = np.sqrt((A ** 2 + E ** 2 + R ** 2) / 3.0)
    else:
        raise ValueError(kind)
    return float(M.mean())


def brute_force_triple_sum(a, e, r, kind):
    """The same object with three explicit Python loops and no array broadcasting.
    Slow on purpose: it is the check that the vectorised tensor is the triple sum
    the definition writes, and not a rearrangement of it."""
    L = len(a)
    total = 0.0
    if kind == "arithmetic":
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    total += (a[i] + e[j] + r[k]) / 3.0
    elif kind == "geometric":
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    total += (a[i] * e[j] * r[k]) ** (1.0 / 3.0)
    elif kind == "rms":
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    total += math.sqrt((a[i] ** 2 + e[j] ** 2 + r[k] ** 2) / 3.0)
    else:
        raise ValueError(kind)
    return total / float(L ** 3)


def closed_form(a, e, r, kind):
    """The factorisations. None where the triple sum does not separate."""
    if kind == "arithmetic":
        return float((a.mean() + e.mean() + r.mean()) / 3.0)
    if kind == "geometric":
        return float(np.mean(np.cbrt(a)) * np.mean(np.cbrt(e)) * np.mean(np.cbrt(r)))
    return None


# ---- vectorised composites over replicate curve matrices, shape (B, L)
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


# ---- TOPSIS, Kolesnikov MSc thesis sec. 3.2.3, brought forward from unpublished work
def topsis_reference(L):
    """The hand-built ideal and anti-ideal vectors, verbatim from the thesis:

      best case  -- RGA [1, 1, ..., 1]; RGE [1, 0, 0, ..., 0]; RGR [1, 1, ..., 1]
      worst case -- RGA [0, 0, ..., 0]; RGE [1, 0.9737, 0.9474, ..., 0.5];
                    RGR [1, 0, 0, ..., 0]

    The printed worst-case RGE is numpy.linspace(1, 0.5, 20) to four decimals; we
    keep the construction (a linear decay from 1 to the 0.5 random baseline) and
    read it at the length our curves actually have."""
    pis_a = np.ones(L)
    pis_e = np.concatenate([[1.0], np.zeros(L - 1)])
    pis_r = np.ones(L)
    nis_a = np.zeros(L)
    nis_e = np.linspace(1.0, 0.5, L)
    nis_r = np.concatenate([[1.0], np.zeros(L - 1)])
    return (np.concatenate([pis_a, pis_e, pis_r]),
            np.concatenate([nis_a, nis_e, nis_r]))


def topsis_cc(A, E, R, pis, nis, wvec):
    """Thesis eq. (2.18)-(2.19): Euclidean distances to the positive and negative
    ideal solutions, closeness Ci = S_minus / (S_plus + S_minus)."""
    V = np.concatenate([A, E, R], axis=1)
    w2 = wvec ** 2
    sp = np.sqrt(((V - pis) ** 2 * w2).sum(axis=1))
    sm = np.sqrt(((V - nis) ** 2 * w2).sum(axis=1))
    return sm / (sm + sp)


# ---------------------------------------------------------------- gradients
def grad_rms_entries(a, e, r):
    """Gradient of the root-mean-square volume with respect to each curve entry.
    dV/da_i = (a_i / (3 L^3)) * sum_{j,k} 1 / M(i,j,k). Finite everywhere M > 0.
    M is homogeneous of degree one in (a, e, r) jointly, so ga.a + ge.e + gr.r = V
    exactly; that identity is checked in main()."""
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
    """Which pair of curve summaries carries the cross-covariance, and how much
    gradient weight sits on each curve. Same block as the companion aggregation
    study, so the numbers are comparable across the two."""
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


MC_R = 400          # second-level resamples, for the Monte Carlo error of a share
MC_SEED = SEED + 777


def mc_std(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return float(v.std(ddof=1)) if v.size > 1 else float("nan")


def delta_share_mc_se(Sb, g_fn, S0):
    """Monte Carlo error of the cross-covariance share and of the width
    understatement, obtained by resampling the B replicate rows. Without this the
    small shares reported below cannot be told from bootstrap noise."""
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
    """Monte Carlo error of the empirical (independent-stream) width
    understatement, by resampling both replicate streams."""
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


def arm_block(name, describe, fn, curves_point, reps_paired, reps_ind,
              summary_fn=None, grad_fn=None, exact_in_summaries=None):
    """One composite arm -> point value, bootstrap interval, and, where the
    composite admits a three-summary reduction, the delta interval with the
    measured covariance between the three curve summaries against the same
    interval with the cross-terms declared zero."""
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


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    results = {
        "study": "An interval for the Pavia Compliance Score, on the curve construction",
        "author": "Anton Sokolov, Tyche Institute, Tallinn",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "safeai_commit": subprocess.check_output(
            ["git", "-C", SAFEAI_REPO, "rev-parse", "HEAD"]).decode().strip(),
        "numpy": np.__version__,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "seed": SEED, "alpha": ALPHA, "z": Z, "B": B, "B_conditioning_check": B_FULL,
    }

    # The deposited experiment of the published run, read READ-ONLY, so that the
    # contrast between the scalar construction and the curve construction is in the
    # artifact rather than quoted from a note.
    with open(os.path.join(os.path.dirname(HERE), "results.json")) as fh:
        deposited = json.load(fh)
    results["deposited_scalar_contrast"] = {
        "source": "results.json of the deposited experiment, opened read-only",
        "point": {t: deposited["models"][t]["point"] for t in TAGS},
        "Corr_paired_of_the_three_scalar_metrics":
            {t: deposited["models"][t]["Corr_paired"] for t in TAGS},
    }

    X, y, names, dataset = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    n, d = X_te.shape
    GRID = d
    L = GRID + 1
    results.update({"dataset": dataset, "n_train": int(len(y_tr)), "n_test": int(n),
                    "d": int(d), "grid": int(GRID), "curve_length_L": int(L)})

    say("=" * 78)
    say("An interval for the Pavia Compliance Score, on the curve construction")
    say("Anton Sokolov, Tyche Institute, Tallinn -- run of %s" % results["generated_utc"])
    say("=" * 78)
    say("[data]  %s: train=%d, test=%d, d=%d" % (dataset, len(y_tr), n, d))
    say("[grid]  severity grid s_t = t/%d, t = 0..%d  ->  curve length L = %d"
        % (GRID, GRID, L))
    say("[boot]  B = %d paired replicates, %d independent-stream replicates, "
        "B_full = %d (logit) / %d (rf) for the conditioning check"
        % (B, B, B_FULL["logit"], B_FULL["rf"]))
    say("[subst] safeai @ %s (vendored, read-only)" % results["safeai_commit"][:12])

    models = {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)

    # index streams, recipe inherited from the deposited runs
    IDX = np.empty((B, n), dtype=np.int64)
    boot_rng = np.random.default_rng(SEED + 1)
    for b in range(B):
        IDX[b] = boot_rng.integers(0, n, n)
    IND = np.empty((3, B, n), dtype=np.int64)
    for k in range(3):
        rk = np.random.default_rng(SEED + 10 + k)
        for b in range(B):
            IND[k, b] = rk.integers(0, n, n)

    pis, nis = topsis_reference(L)
    wvec = np.concatenate([np.full(L, 1.0 / 3.0)] * 3)   # equal weights across the
    # three dimensions, as in the thesis Table 3.12; the closeness coefficient is
    # invariant to a common rescaling, so equal weights are the no-weight case.
    cc_pis = float(topsis_cc(pis[:L][None, :], pis[L:2 * L][None, :],
                             pis[2 * L:][None, :], pis, nis, wvec)[0])
    cc_nis = float(topsis_cc(nis[:L][None, :], nis[L:2 * L][None, :],
                             nis[2 * L:][None, :], pis, nis, wvec)[0])
    say("[topsis] reference check: Ci(best case) = %.6f, Ci(worst case) = %.6f "
        "(thesis: 1.0 and 0.0)" % (cc_pis, cc_nis))
    results["topsis_reference"] = {
        "source": "Kolesnikov MSc thesis sec. 3.2.3, not present in the published article",
        "pis_RGA": pis[:L].tolist(), "pis_RGE": pis[L:2 * L].tolist(),
        "pis_RGR": pis[2 * L:].tolist(),
        "nis_RGA": nis[:L].tolist(), "nis_RGE": nis[L:2 * L].tolist(),
        "nis_RGR": nis[2 * L:].tolist(),
        "weights_per_dimension": [1 / 3, 1 / 3, 1 / 3],
        "Ci_of_best_case": cc_pis, "Ci_of_worst_case": cc_nis,
        "thesis_printed_worst_RGE_head": [1.0, 0.9737, 0.9474],
        "linspace_1_to_0p5_over_20_head": np.linspace(1.0, 0.5, 20)[:3].tolist(),
    }

    results["models"] = {}
    # point curves and paired replicate matrices, retained across the model loop so
    # that the TOPSIS column-normalisation block below can build a decision matrix
    # whose rows are several models, which is what the thesis's step 2 requires.
    KEEP = {}
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

        # ---- point estimate: the three curves
        t0 = time.time()
        a0 = rga_vector(y_te, p_full, GRID)
        e0, order = rge_vector_package(model, X_te, names, GRID, "mean")
        r0_swap, ps = np.empty(L), 0.5 * np.arange(L) / GRID
        for t in range(L):
            r0_swap[t] = rgr_score(p_full, model.predict_proba(tail_swap(X_te, ps[t]))[:, 1],
                                   class_order=CLASS_ORDER)
        t_point = time.time() - t0
        say("[curves] point estimate in %.1f s" % t_point)
        say("  RGA  len=%d  [%s ... %s]" % (L, " ".join("%.4f" % v for v in a0[:4]),
                                            " ".join("%.4f" % v for v in a0[-2:])))
        say("  RGE  len=%d  [%s ... %s]" % (L, " ".join("%.4f" % v for v in e0[:4]),
                                            " ".join("%.4f" % v for v in e0[-2:])))
        say("  RGR  len=%d  [%s ... %s]" % (L, " ".join("%.4f" % v for v in r0_swap[:4]),
                                            " ".join("%.4f" % v for v in r0_swap[-2:])))

        # ---- continuity with the deposited run: the RGA anchor is the deposited RGA
        rga_scalar = float(rga_score(y_te, p_full))
        cont = {"rga_anchor_curve": float(a0[0]),
                "rga_score_scalar": rga_scalar,
                "deposited_point_RGA": DEPOSITED_RGA[tag],
                "abs_gap_anchor_vs_scalar": abs(float(a0[0]) - rga_scalar),
                "abs_gap_anchor_vs_deposited": abs(float(a0[0]) - DEPOSITED_RGA[tag])}
        say("[continuity] RGA anchor %.16f vs deposited point RGA %.16f -> |gap| %.2e"
            % (cont["rga_anchor_curve"], cont["deposited_point_RGA"],
               cont["abs_gap_anchor_vs_deposited"]))
        mres["continuity_with_deposit"] = cont

        # ---- the composite at the point estimate: tensor, brute force, closed form
        say("[composite] point estimate, three ways")
        comp = {}
        for kind in ("arithmetic", "geometric", "rms"):
            t1 = time.time()
            v_bf = brute_force_triple_sum(a0, e0, r0_swap, kind)
            t_bf = time.time() - t1
            v_tn = tensor_mean(a0, e0, r0_swap, kind)
            v_cf = closed_form(a0, e0, r0_swap, kind)
            comp[kind] = {
                "V_brute_force_triple_sum": v_bf,
                "V_vectorised_tensor": v_tn,
                "V_closed_form": v_cf,
                "abs_residual_tensor_vs_brute_force": abs(v_tn - v_bf),
                "abs_residual_closed_form_vs_brute_force":
                    (abs(v_cf - v_bf) if v_cf is not None else None),
                "abs_residual_closed_form_vs_tensor":
                    (abs(v_cf - v_tn) if v_cf is not None else None),
                "brute_force_seconds": t_bf,
            }
            say("  %-10s brute=%.12f  tensor=%.12f  closed=%s  "
                "|tensor-brute|=%.2e  |closed-brute|=%s"
                % (kind, v_bf, v_tn,
                   ("%.12f" % v_cf) if v_cf is not None else "none        ",
                   abs(v_tn - v_bf),
                   ("%.2e" % abs(v_cf - v_bf)) if v_cf is not None else "n/a"))
        mres["factorisation_check"] = comp

        # Euler identity for the root-mean-square gradient: it must reconstruct V
        ga, ge, gr = grad_rms_entries(a0, e0, r0_swap)
        euler = float(ga @ a0 + ge @ e0 + gr @ r0_swap)
        mres["rms_gradient_euler_identity"] = {
            "sum_of_projections": euler,
            "V_rms": comp["rms"]["V_vectorised_tensor"],
            "abs_gap": abs(euler - comp["rms"]["V_vectorised_tensor"]),
        }
        say("  rms gradient check (Euler): ga.a + ge.e + gr.r = %.12f vs V = %.12f, "
            "|gap| = %.2e" % (euler, comp["rms"]["V_vectorised_tensor"],
                              abs(euler - comp["rms"]["V_vectorised_tensor"])))

        # ---- precompute for the fast bootstrap path
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
        say("[precompute] %d nested masked + %d tail-swapped + %d scaled-noise "
            "prediction vectors in %.1f s" % (L, L, L, t_pre))

        # the fast path must reproduce the package curve exactly at the point estimate
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
        say("[fastpath] max abs gap vs the package curves at the point estimate: "
            "RGA %.2e  RGE %.2e  RGR %.2e"
            % (fast_gaps["RGA"], fast_gaps["RGE"], fast_gaps["RGR"]))
        mres["fast_path_vs_package_max_abs_gap"] = fast_gaps

        # ---- which scoring convention the package's own curve builders use.
        # rge_curve and rgr_curve resolve a class order from the fitted model and
        # score full probability MATRICES; rge_score and rgr_score called on a bare
        # 1D probability vector do not. rge_curve gives us a reference curve, so the
        # question can be settled by measurement rather than by reading.
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
            # the counterfactual composite scores EVERY convention-sensitive curve
            # on the bare vector: RGA takes y and a probability vector and has no
            # convention to choose, RGE and RGR are both rescored without
            # class_order. Mixing conventions between curves would answer no
            # question anyone asks.
            "V_under_vector_convention": {
                k: tensor_mean(a0, e_without, r_swap_1d, k)
                for k in ("arithmetic", "geometric", "rms")},
            "RGE_bare_vector_curve": e_without.tolist(),
            "RGR_tail_swap_bare_vector_curve": r_swap_1d.tolist(),
        }
        say("[convention] RGE fast path vs the package curve: with class_order %.2e, "
            "without %.2e" % (conv["rge_max_abs_gap_vs_package_curve_with_class_order"],
                              conv["rge_max_abs_gap_vs_package_curve_without_class_order"]))
        say("[convention] RGR at p = 0.5, probability-matrix convention %.4f vs bare "
            "1D vector %.4f (max gap over the curve %.4f)"
            % (conv["rgr_tail_swap_matrix_convention_last"],
               conv["rgr_tail_swap_vector_convention_last"],
               conv["rgr_tail_swap_max_abs_gap_between_conventions"]))
        mres["scoring_convention"] = conv

        # ---- the paired bootstrap
        t0 = time.time()
        Ab = np.empty((B, L)); Eb = np.empty((B, L)); Rb = np.empty((B, L))
        Rb_noise = np.empty((B, L))
        for b in range(B):
            ib = IDX[b]
            a, e, r = triple(ib, ib, ib)
            Ab[b], Eb[b], Rb[b] = a, e, r
            pb = p_full[ib]
            for t in range(L):
                Rb_noise[b, t] = rgr_score(pb, p_noise[t][ib], class_order=CLASS_ORDER)
        t_paired = time.time() - t0
        say("[bootstrap] paired pass, B = %d, %.1f s (%.2f ms a replicate)"
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

        # ---- the three curve summaries, their marginals, and where their
        #      dependence lives. The article's throughline is about the covariance
        #      BETWEEN these three objects, so it is recorded before any composite.
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
        # per severity index, the correlation between the three curves across
        # replicates. Index 0 is the zero-severity anchor, where RGE and RGR are the
        # constant 1 and the correlation is undefined; the last RGA knot is
        # structurally zero and its replicate column is floating-point residue of
        # the order of 1e-16, so a column whose spread is below CONST_TOL is
        # treated as fixed -- correlating residue against a real column would put
        # a meaningless number into the mean below.
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
        mres["curve_summaries"] = marg
        say("[summaries] corr of the three curve means: RGA-RGE %+.4f  RGA-RGR %+.4f  "
            "RGE-RGR %+.4f"
            % (marg["mean_of_curve"]["Corr"][0][1], marg["mean_of_curve"]["Corr"][0][2],
               marg["mean_of_curve"]["Corr"][1][2]))
        say("[summaries] mean correlation at matched severity index: RGA-RGE %+.4f  "
            "RGA-RGR %+.4f  RGE-RGR %+.4f"
            % (marg["mean_per_index_correlation"]["RGA_RGE"],
               marg["mean_per_index_correlation"]["RGA_RGR"],
               marg["mean_per_index_correlation"]["RGE_RGR"]))

        # the last RGA knot is structurally zero: record that it does not move
        mres["last_rga_knot"] = {
            "point_value": float(a0[-1]),
            "max_abs_over_replicates": float(np.max(np.abs(Ab[:, -1]))),
            "note": "the accumulated partial RGA curve ends at zero by construction, "
                    "so this coordinate carries no sampling variation",
        }

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
            "Compliance Score, arithmetic mean variant (article eq. 13 and 17); "
            "closed form (mean RGA + mean RGE + mean RGR)/3",
            v_arith, curves_point, reps_paired, reps_ind,
            summary_fn=s_mean, grad_fn=lambda s: np.array([1 / 3, 1 / 3, 1 / 3]),
            exact_in_summaries=True))
        arms.append(arm_block(
            "geometric",
            "Compliance Score, geometric mean variant (article eq. 14 and 17); "
            "closed form mean(RGA^1/3) * mean(RGE^1/3) * mean(RGR^1/3)",
            v_geom, curves_point, reps_paired, reps_ind,
            summary_fn=s_cbrt_mean,
            grad_fn=lambda s: np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]]),
            exact_in_summaries=True))
        arms.append(arm_block(
            "rms",
            "Compliance Score, root mean square variant (article eq. 15 and 17); "
            "no closed form, evaluated as the full L^3 tensor",
            v_rms, curves_point, reps_paired, reps_ind,
            summary_fn=make_projection_summary(ga, ge, gr),
            grad_fn=lambda s: np.ones(3), exact_in_summaries=False))
        arms.append(arm_block(
            "topsis",
            "TOPSIS closeness coefficient with the hand-built ideal and anti-ideal "
            "vectors of the Kolesnikov MSc thesis sec. 3.2.3, equal weights; "
            "brought forward from unpublished work, absent from the article",
            lambda A, E, R: topsis_cc(A, E, R, pis, nis, wvec),
            curves_point, reps_paired, reps_ind,
            summary_fn=make_projection_summary(gt_a, gt_e, gt_r),
            grad_fn=lambda s: np.ones(3), exact_in_summaries=False))
        arms.append(arm_block(
            "arithmetic_weighted_SENSITIVITY",
            "SENSITIVITY CHECK ONLY, and a declared departure from the authors' stated "
            "position that they do not assign weights: arithmetic variant with "
            "w = (0.50, 0.20, 0.30)",
            v_arith_weighted, curves_point, reps_paired, reps_ind,
            summary_fn=s_mean, grad_fn=lambda s: W_SENS.copy(),
            exact_in_summaries=True))
        # diagonal arms: no delta block. The diagonal geometric composite is not
        # differentiable at the structurally zero last RGA knot, so we do not
        # linearise the diagonal family at all and let the independent-stream
        # bootstrap carry the cross-term claim there.
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
                C = dm["Corr_of_summaries"]
                ct = dm["cross_term_decomposition"]["cross_contributions_to_variance"]
                say("  %-32s   corr(summaries) RGA-RGE %+.4f  RGA-RGR %+.4f  "
                    "RGE-RGR %+.4f | 2 g g Sigma: %+.3e %+.3e %+.3e | cross share "
                    "of variance %+.2f%% (mc se %.2f)"
                    % ("", C[0][1], C[0][2], C[1][2], ct["RGA_RGE"], ct["RGA_RGR"],
                       ct["RGE_RGR"], dm["cross_covariance_share_of_variance_pct"],
                       dm["cross_share_mc_se_pct"]))

        # ---- product space against the matched-severity diagonal, paired
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
                    float(abs(Vd0 - Vp0) / mres["arms"][kind]["paired_bootstrap"]["width"]),
            }
            say("  %-10s product = %.6f  diagonal = %.6f  diff = %+.6f  "
                "boot95 of diff = [%+.6f, %+.6f]  excludes zero: %s"
                % (kind, Vp0, Vd0, Vd0 - Vp0, lo, hi, diag[kind]["difference_excludes_zero"]))
        mres["product_space_vs_diagonal"] = diag

        # ---- ordering check, power-mean inequality
        Va, Vg, Vr = v_arith(*reps_paired), v_geom(*reps_paired), v_rms(*reps_paired)
        ordering = {
            "point": {"geometric": float(v_geom(a0[None], e0[None], r0_swap[None])[0]),
                      "arithmetic": float(v_arith(a0[None], e0[None], r0_swap[None])[0]),
                      "rms": float(v_rms(a0[None], e0[None], r0_swap[None])[0])},
            "point_ordering_holds": None,
            "replicates_satisfying_geometric_le_arithmetic": int(np.sum(Vg <= Va)),
            "replicates_satisfying_arithmetic_le_rms": int(np.sum(Va <= Vr)),
            "replicates_total": int(B),
        }
        ordering["point_ordering_holds"] = bool(
            ordering["point"]["geometric"] <= ordering["point"]["arithmetic"]
            <= ordering["point"]["rms"])
        mres["ordering_check"] = ordering
        say("")
        say("[ordering] geometric %.6f <= arithmetic %.6f <= rms %.6f : %s "
            "(replicates satisfying: %d/%d and %d/%d)"
            % (ordering["point"]["geometric"], ordering["point"]["arithmetic"],
               ordering["point"]["rms"], ordering["point_ordering_holds"],
               ordering["replicates_satisfying_geometric_le_arithmetic"], B,
               ordering["replicates_satisfying_arithmetic_le_rms"], B))

        # ---- sensitivities
        sens = {}
        # The article and the thesis both say the three vectors have "length n
        # (corresponding to the number of features)", but every curve they construct
        # carries n + 1 knots: eq. (6) writes the RGA vector as [RGA, ..., 0], the RGE
        # curve is said to run from 1 at p = 0 to the 0.5 random baseline at p = 1, and
        # the RGR sweep starts at p = 0. So the literal length-n reading is reached by
        # dropping ONE end knot, and the text does not say which. Both truncations are
        # reported below. They move the composite in OPPOSITE directions, so reporting
        # only one of them would understate the ambiguity.
        #
        # (a) drop the zero-severity anchor from all three curves: L = d = 20
        A20, E20, R20 = Ab[:, 1:], Eb[:, 1:], Rb[:, 1:]
        Ai20, Ei20, Ri20 = Ai[:, 1:], Ei[:, 1:], Ri[:, 1:]
        sens["anchor_dropped_L20"] = {}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            V0 = float(fn_(a0[None, 1:], e0[None, 1:], r0_swap[None, 1:])[0])
            pbk = percentile_block(np.asarray(fn_(A20, E20, R20)))
            pik = percentile_block(np.asarray(fn_(Ai20, Ei20, Ri20)))
            sens["anchor_dropped_L20"][kind] = {
                "V_point": V0, "paired_bootstrap": pbk,
                "independent_stream_bootstrap": pik,
                "empirical_understatement_of_width_pct":
                    float(100.0 * (1.0 - pik["width"] / pbk["width"])),
                "shift_from_primary": V0 - mres["arms"][kind]["V_point"],
            }

        # (a2) the other length-20 reading: drop the TERMINAL knot instead of the
        # anchor. This is the truncation the thesis's own length-20 TOPSIS reference
        # vectors imply, since those keep the anchor: their worst-case RGE is
        # [1, 0.9737, ..., 0.5], an anchor plus nineteen steps.
        A20t, E20t, R20t = Ab[:, :GRID], Eb[:, :GRID], Rb[:, :GRID]
        Ai20t, Ei20t, Ri20t = Ai[:, :GRID], Ei[:, :GRID], Ri[:, :GRID]
        sens["terminal_knot_dropped_L20"] = {}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            V0 = float(fn_(a0[None, :GRID], e0[None, :GRID], r0_swap[None, :GRID])[0])
            pbk = percentile_block(np.asarray(fn_(A20t, E20t, R20t)))
            pik = percentile_block(np.asarray(fn_(Ai20t, Ei20t, Ri20t)))
            sens["terminal_knot_dropped_L20"][kind] = {
                "V_point": V0, "paired_bootstrap": pbk,
                "independent_stream_bootstrap": pik,
                "empirical_understatement_of_width_pct":
                    float(100.0 * (1.0 - pik["width"] / pbk["width"])),
                "shift_from_primary": V0 - mres["arms"][kind]["V_point"],
            }

        # (a3) what the choice between the two length-20 readings costs
        sens["length20_truncation_ambiguity"] = {}
        for kind in ("arithmetic", "geometric", "rms"):
            v_head = sens["anchor_dropped_L20"][kind]["V_point"]
            v_tail = sens["terminal_knot_dropped_L20"][kind]["V_point"]
            w_prim = mres["arms"][kind]["paired_bootstrap"]["width"]
            sens["length20_truncation_ambiguity"][kind] = {
                "V_anchor_dropped": v_head,
                "V_terminal_knot_dropped": v_tail,
                "spread_between_the_two_readings": v_tail - v_head,
                "spread_as_fraction_of_primary_ci_width":
                    float(abs(v_tail - v_head) / w_prim) if w_prim > 0 else None,
                "shifts_have_opposite_signs": bool(
                    (v_head - mres["arms"][kind]["V_point"])
                    * (v_tail - mres["arms"][kind]["V_point"]) < 0),
            }
        # (b) the scaled Gaussian noise RGR arm, continuity with the deposits
        sens["rgr_scaled_gaussian_noise"] = {
            "note": "RGR rebuilt as a sweep of Gaussian noise at sigma_t = (t/d) column "
                    "standard deviations, scored with the package rgr_score; the "
                    "deposited fixed draw sits at t = 10, sigma = 0.5 column sd",
            "curve_point": None, "arms": {},
        }
        r0_noise = np.empty(L)
        for t in range(L):
            r0_noise[t] = rgr_score(p_full, p_noise[t], class_order=CLASS_ORDER)
        sens["rgr_scaled_gaussian_noise"]["curve_point"] = r0_noise.tolist()
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            V0 = float(fn_(a0[None], e0[None], r0_noise[None])[0])
            pbk = percentile_block(np.asarray(fn_(Ab, Eb, Rb_noise)))
            sens["rgr_scaled_gaussian_noise"]["arms"][kind] = {
                "V_point": V0, "paired_bootstrap": pbk,
                "shift_from_primary": V0 - mres["arms"][kind]["V_point"],
            }
        # (c) TOPSIS at the literal thesis length, L = 20. The thesis vectors keep the
        # anchor -- the worst-case RGE is printed as [1.0, 0.9737, 0.9474, ..., 0.5] --
        # so the reading that matches them drops the LAST knot of each curve. The
        # anchor-dropped reading is reported next to it so that the two length-20
        # conventions used in this study are visible side by side rather than buried.
        pis20, nis20 = topsis_reference(GRID)
        wvec20 = np.concatenate([np.full(GRID, 1.0 / 3.0)] * 3)
        cc20_point = float(topsis_cc(a0[None, :GRID], e0[None, :GRID], r0_swap[None, :GRID],
                                     pis20, nis20, wvec20)[0])
        cc20_b = topsis_cc(Ab[:, :GRID], Eb[:, :GRID], Rb[:, :GRID], pis20, nis20, wvec20)
        cc20h_point = float(topsis_cc(a0[None, 1:], e0[None, 1:], r0_swap[None, 1:],
                                      pis20, nis20, wvec20)[0])
        cc20h_b = topsis_cc(Ab[:, 1:], Eb[:, 1:], Rb[:, 1:], pis20, nis20, wvec20)
        sens["topsis_literal_thesis_length_L20"] = {
            "note": "the thesis reference vectors are length 20 on a 20-predictor "
                    "dataset and they keep the anchor, so the literal reading drops the "
                    "last knot of each curve; the worst-case RGE is then exactly the "
                    "printed [1.0, 0.9737, 0.9474, ..., 0.5]",
            "Ci_point": cc20_point,
            "paired_bootstrap": percentile_block(np.asarray(cc20_b)),
            "shift_from_primary": cc20_point - mres["arms"]["topsis"]["V_point"],
            "Ci_point_anchor_dropped_instead": cc20h_point,
            "paired_bootstrap_anchor_dropped_instead":
                percentile_block(np.asarray(cc20h_b)),
            "shift_from_primary_anchor_dropped_instead":
                cc20h_point - mres["arms"]["topsis"]["V_point"],
            "spread_between_the_two_length20_readings": cc20_point - cc20h_point,
            "spread_as_fraction_of_primary_ci_width":
                float(abs(cc20_point - cc20h_point)
                      / mres["arms"]["topsis"]["paired_bootstrap"]["width"]),
        }
        mres["sensitivities"] = sens
        say("")
        say("[sens] anchor dropped, L = %d: arithmetic %+.6f, geometric %+.6f, rms %+.6f "
            "against the primary"
            % (GRID, sens["anchor_dropped_L20"]["arithmetic"]["shift_from_primary"],
               sens["anchor_dropped_L20"]["geometric"]["shift_from_primary"],
               sens["anchor_dropped_L20"]["rms"]["shift_from_primary"]))
        say("[sens] scaled Gaussian noise RGR: arithmetic %+.6f, geometric %+.6f, "
            "rms %+.6f against the primary"
            % (sens["rgr_scaled_gaussian_noise"]["arms"]["arithmetic"]["shift_from_primary"],
               sens["rgr_scaled_gaussian_noise"]["arms"]["geometric"]["shift_from_primary"],
               sens["rgr_scaled_gaussian_noise"]["arms"]["rms"]["shift_from_primary"]))
        say("[sens] terminal knot dropped, L = %d: arithmetic %+.6f, geometric %+.6f, "
            "rms %+.6f against the primary"
            % (GRID, sens["terminal_knot_dropped_L20"]["arithmetic"]["shift_from_primary"],
               sens["terminal_knot_dropped_L20"]["geometric"]["shift_from_primary"],
               sens["terminal_knot_dropped_L20"]["rms"]["shift_from_primary"]))
        for kind in ("arithmetic", "geometric", "rms"):
            amb = sens["length20_truncation_ambiguity"][kind]
            say("[sens] length-20 ambiguity, %-10s anchor-dropped %.6f vs "
                "terminal-dropped %.6f, spread %+.6f = %.2f of the primary interval "
                "width, opposite signs: %s"
                % (kind, amb["V_anchor_dropped"], amb["V_terminal_knot_dropped"],
                   amb["spread_between_the_two_readings"],
                   amb["spread_as_fraction_of_primary_ci_width"],
                   amb["shifts_have_opposite_signs"]))
        say("[sens] TOPSIS at the literal thesis length L = %d: Ci = %.6f (%+.6f); "
            "anchor-dropped instead %.6f (%+.6f); spread %.6f = %.2f of the interval width"
            % (GRID, cc20_point, sens["topsis_literal_thesis_length_L20"]["shift_from_primary"],
               cc20h_point,
               sens["topsis_literal_thesis_length_L20"]["shift_from_primary_anchor_dropped_instead"],
               sens["topsis_literal_thesis_length_L20"]["spread_between_the_two_length20_readings"],
               sens["topsis_literal_thesis_length_L20"]["spread_as_fraction_of_primary_ci_width"]))

        # ---- conditioning check at reduced B: re-search the greedy order and
        #      recompute the perturbation on the resampled rows
        bf = B_FULL[tag]
        say("")
        say("[conditioning] reduced-B check, B_full = %d (%s), on the same index streams"
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
        t_full = time.time() - t0
        cond = {"B_full": bf,
                "gross_check_only": bool(bf < 100),
                "seconds": t_full,
                "seconds_per_replicate": t_full / bf, "arms": {}}
        for kind, fn_ in (("arithmetic", v_arith), ("geometric", v_geom), ("rms", v_rms)):
            Vfroz = np.asarray(fn_(Ab[:bf], Eb[:bf], Rb[:bf]))
            Vfull = np.asarray(fn_(Af, Ef, Rf))
            # Monte Carlo band on the sd ratio: paired resamples of the (frozen,
            # full-recompute) replicate pairs, same picks for every variant.
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
                "sd_ratio_full_over_frozen":
                    float(Vfull.std(ddof=1) / Vfroz.std(ddof=1)),
                "sd_ratio_mc_band_95": band,
            }
            c = cond["arms"][kind]
            say("  %-10s sd frozen %.6f  sd full-recompute %.6f  ratio %.3f  "
                "mc band [%.3f, %.3f]  mean |paired diff| %.6f"
                % (kind, c["sd_frozen"], c["sd_full"], c["sd_ratio_full_over_frozen"],
                   band[0], band[1], c["mean_abs_paired_difference"]))
        say("  cost: %.1f s for %d replicates (%.2f s a replicate)"
            % (t_full, bf, t_full / bf))
        mres["conditioning_check"] = cond

        mres["curves_point"] = {
            "RGA_partial": a0.tolist(),
            "RGE_greedy_mean_baseline": e0.tolist(),
            "RGE_greedy_removal_order": [names[i] for i in order],
            "RGR_tail_swap": r0_swap.tolist(),
            "RGR_tail_swap_p_grid": ps.tolist(),
            "RGR_scaled_gaussian_noise": r0_noise.tolist(),
        }
        mres["timings_s"] = {
            "point_curves": t_point, "precompute": t_pre,
            "paired_bootstrap": t_paired, "independent_stream_bootstrap": t_ind,
            "conditioning_check": t_full,
        }
        results["models"][tag] = mres
        KEEP[tag] = {"point": np.concatenate([a0, e0, r0_swap]),
                     "paired": (Ab, Eb, Rb)}

    # ---------------------------------------------------------------- cross-model
    # ---- TOPSIS step 2, measured rather than argued
    # The thesis's generic recipe normalises each column of the decision matrix by the
    # square root of the sum of squares over alternatives before weighting. This study
    # does not, because the closeness coefficient of a single model has to stand alone
    # if it is to carry an interval. That is a decision, not a fact, so it is measured
    # here on every alternative set the run actually has, rather than defended in prose.
    say("")
    say("[topsis-norm] thesis step 2 (column normalisation), measured")
    alt_sets = {"models_and_references": [KEEP[t]["point"] for t in TAGS] + [pis, nis],
                "models_only": [KEEP[t]["point"] for t in TAGS]}
    for t in TAGS:
        alt_sets["one_model_%s_and_references" % t] = [KEEP[t]["point"], pis, nis]
    tn = {"note": "column norms are held fixed at the point-estimate decision matrix; "
                  "resampling them as well would confound the normaliser's own sampling "
                  "variation with the model's, and would need the joint covariance "
                  "across models, which this study does not estimate",
          "alternative_sets": {}}
    for nm, alts in alt_sets.items():
        Mx = np.vstack(alts)
        nrm = np.sqrt((Mx ** 2).sum(axis=0))
        entry = {"n_alternatives": int(Mx.shape[0]),
                 "min_column_norm": float(nrm.min()),
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
            rec = {"Ci_normalised": ccn, "Ci_unnormalised": base,
                   "shift": ccn - base,
                   "shift_as_fraction_of_ci_width": float(abs(ccn - base) / w_t)}
            if nm == "models_and_references":
                Ab_, Eb_, Rb_ = KEEP[t]["paired"]
                ccn_b = topsis_cc(Ab_ / safe[:L], Eb_ / safe[L:2 * L],
                                  Rb_ / safe[2 * L:], pn, qn, wvec)
                rec["paired_bootstrap"] = percentile_block(np.asarray(ccn_b))
            entry["models"][t] = rec
        tn["alternative_sets"][nm] = entry
        say("  %-34s alternatives=%d  min column norm %.2e  %s"
            % (nm, entry["n_alternatives"], entry["min_column_norm"],
               "  ".join("%s %.6f (%+.6f)"
                         % (t, entry["models"][t]["Ci_normalised"],
                            entry["models"][t]["shift"]) for t in TAGS)))
    tn["comparison_set_dependence_of_the_shift"] = {
        t: {nm: tn["alternative_sets"][nm]["models"][t]["shift"]
            for nm in alt_sets if nm != "models_only"} for t in TAGS}
    results["topsis_column_normalisation_sensitivity"] = tn
    say("  the last RGA knot is structurally zero for every model by eq. (6) -- what "
        "the run stores is floating-point residue of the order of 1e-16 -- so on a "
        "decision matrix of models alone that column has no meaningful norm and step 2 "
        "is not defined; the models_only row above is what that degeneracy does.")

    results["published_table5_california_hmda"] = {
        "note": "Giudici and Kolesnikov, Table 5, California HMDA. Reproduced here ONLY "
                "to check the ordering the power-mean inequality forces. Their data are "
                "California HMDA and ours are German credit, so no numerical agreement "
                "is claimed or expected.",
        "rows": {
            "LR": [0.59695, 0.47965, 0.67482],
            "RF": [0.63218, 0.53011, 0.69337],
            "XGB": [0.62893, 0.53042, 0.68661],
            "SEM": [0.63151, 0.53229, 0.68968],
            "VEM": [0.63217, 0.53141, 0.69202],
            "NN": [0.62639, 0.52752, 0.68482],
            "Random": [0.76681, 0.61145, 0.84059],
        },
        "column_order": ["arithmetic", "geometric", "rms"],
    }
    t5 = results["published_table5_california_hmda"]["rows"]
    t5_ok = {k: bool(v[1] <= v[0] <= v[2]) for k, v in t5.items()}
    results["published_table5_california_hmda"]["ordering_holds_per_row"] = t5_ok
    results["published_table5_california_hmda"]["ordering_holds_everywhere"] = bool(
        all(t5_ok.values()))
    say("")
    say("[table5] the published table obeys geometric <= arithmetic <= rms in every "
        "row: %s" % results["published_table5_california_hmda"]["ordering_holds_everywhere"])

    # How the product-space-against-diagonal shift compares with the gaps between
    # models in their own table. The comparison is made WITHIN a variant: a geometric
    # shift is measured against the spread of the geometric column, not of some other
    # column. The five models compared are the ones other than logistic regression and
    # the random baseline, which are the two extremes of every column.
    mid = ["RF", "XGB", "SEM", "VEM", "NN"]
    spread = {}
    for c, kind in enumerate(results["published_table5_california_hmda"]["column_order"]):
        vals = [t5[m][c] for m in mid]
        spread[kind] = {"min": min(vals), "max": max(vals), "spread": max(vals) - min(vals)}
        for tag in TAGS:
            d = results["models"][tag]["product_space_vs_diagonal"][kind]
            spread[kind]["indexing_shift_over_spread_%s" % tag] = float(
                abs(d["diagonal_minus_product_space"]) / (max(vals) - min(vals))) \
                if max(vals) > min(vals) else None
    results["published_table5_california_hmda"]["middle_five_model_spread_per_column"] = {
        "models": mid, "per_column": spread}
    for kind in results["published_table5_california_hmda"]["column_order"]:
        s = spread[kind]
        say("[table5] %-10s spread over %s = %.5f; our indexing shift is %.2f of it "
            "(logit) and %.2f (rf)"
            % (kind, "/".join(mid), s["spread"],
               s["indexing_shift_over_spread_logit"], s["indexing_shift_over_spread_rf"]))

    results["total_runtime_s"] = time.time() - t_start
    say("")
    say("[done] total runtime %.1f s" % results["total_runtime_s"])

    with open(os.path.join(HERE, "results-pavia-composite.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(HERE, "run-pavia-composite.log"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
