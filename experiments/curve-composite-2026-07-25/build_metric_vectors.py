#!/usr/bin/env python3
"""
Step one of the curve-composite study, 2026-07-25.

The Pavia Compliance Score does not consume three scalars. It consumes the three
VECTORS from which the RGA, RGE and RGR curves are built (Giudici and Kolesnikov,
"Integrating Safe AI Metrics", eq. 13-17; Kolesnikov MSc thesis, eq. 2.13-2.17).
This script builds those three vectors on German credit, for a fitted logistic
model and a fitted random forest, and reports what each index means, how long each
vector naturally is, and what one evaluation costs.

Substrate: safeai (github.com/koleso500/safeai) at pinned commit 39768fc, the
vendored clone and import stubs of the 2026-07-24 redraw run, reused READ-ONLY.
This script never clones, never writes into the substrate, and never touches the
deposited experiment directories.

Grid convention. Every index in this study runs over the SAME normalised severity
grid s_t = t / GRID for t = 0 .. GRID, so index t means the same fraction of
severity in all three vectors and the diagonal i = j = k is meaningful. GRID is
set to the number of predictors d, which makes the RGE grid the natural one:
one feature removed per step, all d of them, plus the untouched anchor.
"""
import json
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

from safeai.rga import rga_curve
from safeai.rge import rge_curve
from safeai.rgr import rgr_curve, rgr_score

SEED = 20260723          # identical to the deposited experiment and the redraw run
TEST_SIZE = 0.3
SIGMA_MAX = 1.0          # noise arms sweep 0 .. 1.0 column standard deviations;
                         # the deposited fixed draw sits at 0.5, i.e. mid-grid


def say(msg, log):
    print(msg, flush=True)
    log.append(msg)


def load_data():
    ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "good").astype(int)
    names = list(ds.feature_names)
    return X, y, names, "statlog-german-credit (OpenML credit-g v1)"


# ---------------------------------------------------------------- the three vectors
def rga_vector(y_true, p_full, grid):
    """Article eq. (5)-(6): partial RGA contributions accumulated in reverse, from
    the full RGA down to zero. In the pinned package this is rga_curve with
    curve_method='partial'; note that curve_method='auto' resolves to 'removal',
    NOT to 'partial', despite the docstring, so 'partial' must be asked for.

    Index t = number of top-ranked segments already removed, t = 0 .. grid.
    Length grid + 1."""
    res = rga_curve(
        y_true, p_full,
        curve_method="partial",
        n_segments=grid,
        normalize_to_perfect=False,   # we want the vector, not the normalised area
    )
    return np.asarray(res["curve"], dtype=float)


def rge_vector(model, X, feature_names, grid, baseline):
    """Article eq. (12) and the thesis AURGE procedure: greedy monotone removal,
    at each step the feature whose removal keeps RGE* highest. In the pinned
    package this is rge_curve(method='tabular', masking_method='greedy').

    Index t = number of features already removed, t = 0 .. grid. Length grid + 1;
    grid cannot exceed d, so this is the one vector whose resolution the data fix.
    Returns the curve and the greedy removal order."""
    res = rge_curve(
        model, X,
        method="tabular",
        feature_names=feature_names,
        masking_method="greedy",
        baseline=baseline,
        n_steps=grid,
        verbose=False,
    )
    return np.asarray(res["rge_scores"], dtype=float), res["removed_features"]


def rgr_vector_package(model, X, grid, seed):
    """The only tabular RGR curve the pinned package ships: a Gaussian noise sweep
    with a SCALAR sigma applied in raw feature units. Index t = noise level
    sigma_t = SIGMA_MAX * t / grid. Length grid + 1."""
    strengths = SIGMA_MAX * np.arange(grid + 1) / grid
    res = rgr_curve(
        model, X, strengths,
        method="noise",
        random_seed=seed,
        verbose=False,
    )
    return np.asarray(res["rgr_scores"], dtype=float), strengths


def rgr_vector_scaled_noise(model, X, p_orig, grid, seed, col_std):
    """The deposited experiment's perturbation, swept: Gaussian noise of
    sigma_t = SIGMA_MAX * t / grid COLUMN STANDARD DEVIATIONS. Scored with the
    package's rgr_score, which is the published eq. (8). Length grid + 1."""
    rng = np.random.default_rng(seed)
    out = np.empty(grid + 1, dtype=float)
    for t in range(grid + 1):
        sigma = SIGMA_MAX * t / grid
        Xp = X + rng.normal(0.0, sigma, size=X.shape) * col_std if sigma > 0 else X
        p_pert = model.predict_proba(Xp)[:, 1]
        out[t] = rgr_score(p_orig, p_pert)
    return out


def tail_swap(X, p):
    """The article's own perturbation: within each column, values below the p-th
    percentile are swapped symmetrically with values above the (1-p)-th percentile.
    Rank i exchanges with rank n+1-i for the outermost floor(p*n) ranks on each
    side; at p = 0.5 the column ordering is fully reversed."""
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


def rgr_vector_tailswap(model, X, p_orig, grid):
    """The published RGR curve: RGR(p) for p = 0.5 * t / grid, t = 0 .. grid, so p
    runs over [0, 0.5] in fixed increments exactly as the article states. Scored
    with rgr_score (published eq. 8). Length grid + 1."""
    out = np.empty(grid + 1, dtype=float)
    ps = 0.5 * np.arange(grid + 1) / grid
    for t, p in enumerate(ps):
        p_pert = model.predict_proba(tail_swap(X, p))[:, 1]
        out[t] = rgr_score(p_orig, p_pert)
    return out, ps


# ---------------------------------------------------------------- the composite
def volume_tensor(a, e, r, kind):
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


def volume_closed_form(a, e, r, kind):
    """The factorisations, re-verified against the tensor below."""
    if kind == "arithmetic":
        return float((a.mean() + e.mean() + r.mean()) / 3.0)
    if kind == "geometric":
        return float(np.mean(np.cbrt(a)) * np.mean(np.cbrt(e)) * np.mean(np.cbrt(r)))
    return None


def volume_diagonal(a, e, r, kind):
    """The alternative reading: average only over i = j = k, matched severity."""
    if kind == "arithmetic":
        M = (a + e + r) / 3.0
    elif kind == "geometric":
        M = np.cbrt(a * e * r)
    elif kind == "rms":
        M = np.sqrt((a ** 2 + e ** 2 + r ** 2) / 3.0)
    else:
        raise ValueError(kind)
    return float(M.mean())


# ---------------------------------------------------------------- main
def main():
    log = []
    t_start = time.time()

    X, y, names, dataset = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    n, d = X_te.shape
    GRID = d                      # one RGE step per predictor; L = d + 1 knots
    L = GRID + 1
    say("[data] %s: train=%d, test=%d, d=%d" % (dataset, len(y_tr), n, d), log)
    say("[grid] severity grid s_t = t/%d, t = 0..%d -> vector length L = %d" % (GRID, GRID, L), log)

    models = {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for tag, m in models.items():
        m.fit(X_tr, y_tr)
    col_std = X_te.std(axis=0, keepdims=True)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "n_test": int(n), "d": int(d),
        "seed": SEED, "grid": int(GRID), "vector_length": int(L),
        "sigma_max_in_column_sd": SIGMA_MAX,
        "safeai_commit": subprocess.check_output(
            ["git", "-C", SAFEAI_REPO, "rev-parse", "HEAD"]).decode().strip(),
        "models": {},
    }

    timings = {}
    for tag, model in models.items():
        say("", log)
        say("=== %s ===" % tag, log)
        p_full = model.predict_proba(X_te)[:, 1]

        t0 = time.time(); a = rga_vector(y_te, p_full, GRID); t_rga = time.time() - t0
        t0 = time.time(); e_mean, order_mean = rge_vector(model, X_te, names, GRID, "mean"); t_rge_mean = time.time() - t0
        t0 = time.time(); e_zero, order_zero = rge_vector(model, X_te, names, GRID, "zero"); t_rge_zero = time.time() - t0
        t0 = time.time(); r_pkg, sig_pkg = rgr_vector_package(model, X_te, GRID, SEED); t_rgr_pkg = time.time() - t0
        t0 = time.time(); r_scaled = rgr_vector_scaled_noise(model, X_te, p_full, GRID, SEED, col_std); t_rgr_sc = time.time() - t0
        t0 = time.time(); r_swap, ps = rgr_vector_tailswap(model, X_te, p_full, GRID); t_rgr_sw = time.time() - t0

        def show(name, v, extra=""):
            say("  %-22s len=%2d  first4=%s  last=%.4f%s"
                % (name, len(v), np.array2string(v[:4], precision=4, floatmode="fixed"),
                   v[-1], extra), log)

        show("RGA (partial)", a)
        show("RGE (greedy, mean)", e_mean)
        show("RGE (greedy, zero)", e_zero)
        show("RGR (pkg noise, raw)", r_pkg)
        show("RGR (noise x col sd)", r_scaled)
        show("RGR (tail swap)", r_swap)
        say("  greedy removal order (mean baseline), first 5: %s" % order_mean[:5], log)

        # ---- re-verification of the factorisation, on these real vectors
        checks = {}
        for kind in ("arithmetic", "geometric", "rms"):
            V = volume_tensor(a, e_mean, r_swap, kind)
            Vc = volume_closed_form(a, e_mean, r_swap, kind)
            Vd = volume_diagonal(a, e_mean, r_swap, kind)
            checks[kind] = {
                "V_tensor": V,
                "V_closed_form": Vc,
                "abs_gap_closed_vs_tensor": (abs(V - Vc) if Vc is not None else None),
                "V_diagonal": Vd,
                "diagonal_minus_product_space": Vd - V,
            }
            say("  V[%-10s] tensor=%.6f  closed=%s  |gap|=%s  diag=%.6f  diag-V=%+.6f"
                % (kind, V,
                   ("%.6f" % Vc) if Vc is not None else "  none  ",
                   ("%.2e" % abs(V - Vc)) if Vc is not None else "   n/a  ",
                   Vd, Vd - V), log)

        timings[tag] = {
            "rga_s": t_rga, "rge_mean_s": t_rge_mean, "rge_zero_s": t_rge_zero,
            "rgr_pkg_s": t_rgr_pkg, "rgr_scaled_noise_s": t_rgr_sc,
            "rgr_tailswap_s": t_rgr_sw,
            "one_full_triple_mean_tailswap_s": t_rga + t_rge_mean + t_rgr_sw,
        }
        say("  timings (s): RGA %.3f | RGE mean %.3f, zero %.3f | RGR pkg %.3f, "
            "scaled %.3f, swap %.3f | one triple %.3f"
            % (t_rga, t_rge_mean, t_rge_zero, t_rgr_pkg, t_rgr_sc, t_rgr_sw,
               timings[tag]["one_full_triple_mean_tailswap_s"]), log)

        out["models"][tag] = {
            "RGA_partial": a.tolist(),
            "RGE_greedy_mean_baseline": e_mean.tolist(),
            "RGE_greedy_zero_baseline": e_zero.tolist(),
            "RGE_removal_order_mean": order_mean,
            "RGR_package_noise_raw_units": r_pkg.tolist(),
            "RGR_noise_times_column_sd": r_scaled.tolist(),
            "RGR_tail_swap": r_swap.tolist(),
            "rgr_package_sigma_grid": sig_pkg.tolist(),
            "rgr_tail_swap_p_grid": ps.tolist(),
            "composite_checks_RGA_RGEmean_RGRswap": checks,
            "timings_s": timings[tag],
        }

    # ---- cost of the bootstrap, measured rather than guessed
    say("", log)
    say("=== per-evaluation cost ===", log)
    for tag in models:
        tt = timings[tag]
        say("  %-6s one full triple (RGA + greedy RGE + tail-swap RGR) = %.3f s"
            % (tag, tt["one_full_triple_mean_tailswap_s"]), log)
        for B in (200, 2000):
            say("        B=%-5d naive re-evaluation: %.1f min"
                % (B, B * tt["one_full_triple_mean_tailswap_s"] / 60.0), log)
    out["total_runtime_s"] = time.time() - t_start
    say("[done] %.1f s" % out["total_runtime_s"], log)

    with open(os.path.join(HERE, "results-step1-vectors.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, "run-step1.log"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
