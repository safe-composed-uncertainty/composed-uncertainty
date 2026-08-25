#!/usr/bin/env python3
"""
Redraw rerun for the SAFE composed-uncertainty experiment (2026-07-24).

Question: in the v0.1 experiment the RGR perturbation matrix was drawn ONCE
(fixed seed) and conditioned on across all bootstrap replicates. Does the
reported cross-metric correlation structure (up to ~+0.60, chain rho ~+0.72)
survive if the perturbations are REDRAWN independently on every bootstrap
replicate (the unconditional bootstrap)?

Two schemes, same index streams, same data, same models, same pinned safeai:

  scheme A  "fixed-draw"  — exact reproduction of the v0.1 conditioning:
             one perturbation matrix per model, drawn from the global
             rng(SEED) in the same order as the original run_experiment.py
             (logit first, then rf). All bootstrap replicates reuse it.
  scheme B  "redraw"      — RGA and RGE columns are identical to scheme A
             (both are deterministic functions of the resampled rows; see
             the determinism audit in REDRAW-SUMMARY.md). Only RGR changes:
             replicate b uses a fresh perturbation matrix drawn with
             seed = REDRAW_BASE[model] + b, and fresh model predictions on
             X_te + pert_b. Per-replicate seeding keeps the run reproducible.

The paired-bootstrap row-index stream is IDENTICAL between schemes (and
identical to the original: np.random.default_rng(SEED + 1), re-instantiated
per model), so any difference between the two correlation matrices is
attributable to the redraw alone.

Chain: the two-link shared-index pass (rng SEED + 99, as in the original) is
run under both schemes. Under scheme B, link A (logit) and link B (rf) use
per-replicate perturbations with DISTINCT seed bases, so no artificial
cross-link coupling is injected by the redraw itself. The chain reuses the
same per-replicate perturbation matrices as the per-model paired pass (one
redraw sequence per model); the chain and per-model analyses are separate
estimations, so this reuse couples nothing within either analysis.

Substrate: safeai pinned commit 39768fc, cloned into ./safeai-src (this
directory), tabular paths only, same import-stub trick as the original.

Nothing in ../  (the original experiment directory) is read except nothing —
this script is self-contained; it re-derives everything from data + seeds.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SAFEAI_REPO = os.path.join(HERE, "safeai-src")
STUBS = os.path.join(HERE, "_stubs")

SEED = 20260723          # identical to the original experiment
B = int(os.environ.get("REDRAW_B", "2000"))
ALPHA = 0.05
NOISE_SD = 0.5
REDRAW_BASE = {"logit": SEED + 100_000, "rf": SEED + 200_000}


def _write_stubs():
    """Same inert torch/art import stubs as the original run_experiment.py."""
    files = {
        "torch/__init__.py":
            "class Tensor: pass\n"
            "class _Callable:\n"
            "    def __init__(self,*a,**k): raise RuntimeError('torch stub invoked')\n"
            "    def __call__(self,*a,**k): raise RuntimeError('torch stub invoked')\n"
            "float32 = 'float32'\n"
            "class _NoGrad:\n"
            "    def __call__(self, fn): return fn\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, *a): return False\n"
            "def no_grad(*a, **k): return _NoGrad()\n"
            "def tensor(*a,**k): raise RuntimeError('torch stub invoked')\n"
            "def softmax(*a,**k): raise RuntimeError('torch stub invoked')\n"
            "def __getattr__(name): return _Callable\n",
        "torch/nn/__init__.py": "class Module: pass\nfrom . import functional\n",
        "torch/nn/functional.py":
            "def __getattr__(name):\n"
            "    def _f(*a,**k): raise RuntimeError('torch stub functional invoked')\n"
            "    return _f\n",
        "torch/utils/__init__.py": "",
        "torch/utils/data/__init__.py":
            "class Dataset: pass\n"
            "class DataLoader:\n"
            "    def __init__(self,*a,**k): raise RuntimeError('torch stub DataLoader invoked')\n",
        "art/__init__.py": "",
        "art/attacks/__init__.py": "",
        "art/attacks/evasion.py": "".join(
            f"class {n}:\n    def __init__(self,*a,**k): raise RuntimeError('art stub: {n}')\n"
            for n in ["FastGradientMethod", "ProjectedGradientDescent", "SquareAttack",
                      "HopSkipJump", "SimBA", "Wasserstein", "SpatialTransformation"]),
        "art/estimators/__init__.py": "",
        "art/estimators/classification.py":
            "class PyTorchClassifier:\n    def __init__(self,*a,**k): raise RuntimeError('art stub')\n"
            "class SklearnClassifier:\n    def __init__(self,*a,**k): raise RuntimeError('art stub')\n",
    }
    for rel, body in files.items():
        path = os.path.join(STUBS, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(body)


_write_stubs()
sys.path.insert(0, STUBS)
sys.path.insert(0, SAFEAI_REPO)

import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from safeai.rga import rga_score
from safeai.rge import rge_score
from safeai.rgr import rgr_score

METRICS = ["RGA", "RGE", "RGR"]


def load_data():
    try:
        ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
        X = ds.data.astype(float)
        y = (ds.target == "good").astype(int)
        return X, y, "statlog-german-credit (OpenML credit-g v1)"
    except Exception as e:
        print(f"[data] openml unavailable ({e.__class__.__name__}); falling back to breast_cancer")
        ds = load_breast_cancer()
        return ds.data, ds.target, "sklearn breast_cancer"


def geo_mean(v):
    return float(np.exp(np.mean(np.log(np.clip(v, 1e-12, None)))))


def corr_dict(M):
    C = np.corrcoef(M, rowvar=False)
    return {
        "matrix": C.tolist(),
        "corr_RGA_RGE": float(C[0, 1]),
        "corr_RGA_RGR": float(C[0, 2]),
        "corr_RGE_RGR": float(C[1, 2]),
    }


def main():
    t0 = time.time()
    X, y, dataset_name = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    n = len(y_te)
    print(f"[data] {dataset_name}: train={len(y_tr)}, test={n}, d={X.shape[1]}, B={B}")

    models = {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for tag, model in models.items():
        model.fit(X_tr, y_tr)

    col_std = X_te.std(axis=0, keepdims=True)

    # ---- scheme A precompute: EXACT replication of the original global-rng flow.
    # Original: rng = default_rng(SEED); precompute(logit) draws one normal matrix,
    # then precompute(rf) draws the next from the SAME stream.
    rng_global = np.random.default_rng(SEED)
    pre = {}
    for tag in ["logit", "rf"]:          # same order as the original dict iteration
        model = models[tag]
        p_full = model.predict_proba(X_te)[:, 1]
        d = X_te.shape[1]
        col_mean = X_te.mean(axis=0)
        p_masked = np.empty((d, n))
        for j in range(d):
            Xm = X_te.copy()
            Xm[:, j] = col_mean[j]
            p_masked[j] = model.predict_proba(Xm)[:, 1]
        pert = rng_global.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
        p_pert_fixed = model.predict_proba(X_te + pert)[:, 1]
        pre[tag] = (p_full, p_masked, p_pert_fixed)
    print(f"[precompute] done in {time.time()-t0:.1f}s")

    # ---- scheme B: per-replicate redraw of the perturbations, fresh predictions.
    # seed = REDRAW_BASE[tag] + b  (deterministic per replicate index).
    t1 = time.time()
    p_pert_redraw = {}
    for tag in ["logit", "rf"]:
        model = models[tag]
        mat = np.empty((B, n))
        base = REDRAW_BASE[tag]
        for b in range(B):
            rb = np.random.default_rng(base + b)
            pert_b = rb.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
            mat[b] = model.predict_proba(X_te + pert_b)[:, 1]
        p_pert_redraw[tag] = mat
        print(f"[redraw] {tag}: {B} fresh perturbation draws + predictions "
              f"in {time.time()-t1:.1f}s cumulative")

    # ---- paired bootstrap, both schemes, SAME index stream per model
    # (original analyze(): boot_rng = default_rng(SEED + 1), fresh per model).
    def metric_vector_fixed(tag, idx):
        p_full, p_masked, p_pert_fixed = pre[tag]
        yb, pf = y_te[idx], p_full[idx]
        rga = rga_score(yb, pf)
        rge = float(np.mean([rge_score(pf, p_masked[j][idx]) for j in range(p_masked.shape[0])]))
        rgr = rgr_score(pf, p_pert_fixed[idx])
        return np.array([rga, rge, rgr])

    results = {
        "dataset": dataset_name, "n_test": int(n), "B": B, "alpha": ALPHA,
        "noise_sd": NOISE_SD, "seed": SEED,
        "redraw_seed_rule": "per-replicate seed = REDRAW_BASE[model] + b; "
                            f"REDRAW_BASE = {REDRAW_BASE}",
        "safeai_commit": subprocess.check_output(
            ["git", "-C", SAFEAI_REPO, "rev-parse", "HEAD"]).decode().strip(),
        "index_streams": {
            "paired_bootstrap": "default_rng(SEED+1), re-instantiated per model (as original)",
            "chain_shared": "default_rng(SEED+99) (as original)",
        },
        "models": {},
    }

    paired_store = {}
    for tag in ["logit", "rf"]:
        t2 = time.time()
        p_full = pre[tag][0]
        boot_rng = np.random.default_rng(SEED + 1)
        paired_A = np.empty((B, 3))
        paired_Bs = np.empty((B, 3))
        for b in range(B):
            idx = boot_rng.integers(0, n, n)
            vec = metric_vector_fixed(tag, idx)
            paired_A[b] = vec
            # scheme B: RGA and RGE are deterministic in idx -> identical columns;
            # RGR recomputed against the replicate-b fresh perturbation predictions.
            rgr_b = rgr_score(p_full[idx], p_pert_redraw[tag][b][idx])
            paired_Bs[b] = [vec[0], vec[1], rgr_b]
        paired_store[tag] = (paired_A, paired_Bs)

        point = metric_vector_fixed(tag, np.arange(n))
        CA_ = np.array([geo_mean(v) for v in paired_A])
        CB_ = np.array([geo_mean(v) for v in paired_Bs])
        qlo, qhi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
        results["models"][tag] = {
            "point_fixed_draw": dict(zip(METRICS + ["C_geomean"],
                                         [float(v) for v in point] + [geo_mean(point)])),
            "fixed_draw": {
                "Corr": corr_dict(paired_A),
                "Sigma": np.cov(paired_A, rowvar=False).tolist(),
                "sd": {m: float(paired_A[:, k].std(ddof=1)) for k, m in enumerate(METRICS)},
                "ci_C_paired_percentile": [float(np.percentile(CA_, qlo)), float(np.percentile(CA_, qhi))],
                "sd_C_paired": float(CA_.std(ddof=1)),
            },
            "redraw": {
                "Corr": corr_dict(paired_Bs),
                "Sigma": np.cov(paired_Bs, rowvar=False).tolist(),
                "sd": {m: float(paired_Bs[:, k].std(ddof=1)) for k, m in enumerate(METRICS)},
                "ci_C_paired_percentile": [float(np.percentile(CB_, qlo)), float(np.percentile(CB_, qhi))],
                "sd_C_paired": float(CB_.std(ddof=1)),
                "mean_RGR": float(paired_Bs[:, 2].mean()),
            },
        }
        cA = results["models"][tag]["fixed_draw"]["Corr"]
        cB = results["models"][tag]["redraw"]["Corr"]
        print(f"[{tag}] fixed-draw corr: RGA-RGE={cA['corr_RGA_RGE']:+.3f} "
              f"RGA-RGR={cA['corr_RGA_RGR']:+.3f} RGE-RGR={cA['corr_RGE_RGR']:+.3f}")
        print(f"[{tag}] redraw     corr: RGA-RGE={cB['corr_RGA_RGE']:+.3f} "
              f"RGA-RGR={cB['corr_RGA_RGR']:+.3f} RGE-RGR={cB['corr_RGE_RGR']:+.3f}")
        print(f"[{tag}] paired pass in {time.time()-t2:.1f}s")

    # ---- two-link chain under both schemes (shared idx stream SEED+99).
    t3 = time.time()
    shared_rng = np.random.default_rng(SEED + 99)
    CA_fix = np.empty(B); CB_fix = np.empty(B)
    CA_red = np.empty(B); CB_red = np.empty(B)
    # Persistence only (added 2026-07-27): record the shared-stream replicate
    # matrices so that Monte Carlo error bars and the DeLong cross-check can be
    # recomputed without a fresh run. Read-only side effect -- no random draw,
    # no arithmetic, and no ordering is changed by these stores.
    chain_vec_fix = {"logit": np.empty((B, 3)), "rf": np.empty((B, 3))}
    chain_vec_red = {"logit": np.empty((B, 3)), "rf": np.empty((B, 3))}
    chain_idx = np.empty((B, n), dtype=np.int32)
    for b in range(B):
        idx = shared_rng.integers(0, n, n)
        chain_idx[b] = idx
        for tag, arr_fix, arr_red in (("logit", CA_fix, CA_red), ("rf", CB_fix, CB_red)):
            vec = metric_vector_fixed(tag, idx)
            arr_fix[b] = geo_mean(vec)
            rgr_b = rgr_score(pre[tag][0][idx], p_pert_redraw[tag][b][idx])
            arr_red[b] = geo_mean([vec[0], vec[1], rgr_b])
            chain_vec_fix[tag][b] = vec
            chain_vec_red[tag][b] = [vec[0], vec[1], rgr_b]

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
        return {
            "estimator": "plug-in product of the two full-sample link scores",
            "link_point_values": {"A": float(point_A), "B": float(point_B)},
            "cross_link_correlation": rho,
            "link_covariance": S.tolist(),
            "h_point": h_point,
            "bootstrap_link_means": {"A": boot_A, "B": boot_B},
            "bootstrap_mean_product_sensitivity": boot_A * boot_B,
            "ci95_measured_covariance": [float(v) for v in ci_meas],
            "ci95_cross_term_declared_zero": [float(v) for v in ci_decl],
            "width_measured": float(w_meas),
            "width_declared_zero": float(w_decl),
            "understatement_of_width_pct": float(100 * (1 - w_decl / w_meas)),
        }

    point_A = results["models"]["logit"]["point_fixed_draw"]["C_geomean"]
    point_B = results["models"]["rf"]["point_fixed_draw"]["C_geomean"]
    results["chain"] = {
        "links": {"A": "logit", "B": "rf"},
        "chain_functional": "h = C_A * C_B (both-must-hold)",
        "fixed_draw": chain_block(CA_fix, CB_fix, point_A, point_B),
        "redraw": chain_block(CA_red, CB_red, point_A, point_B),
    }
    for scheme in ["fixed_draw", "redraw"]:
        ch = results["chain"][scheme]
        print(f"[chain/{scheme}] rho={ch['cross_link_correlation']:+.4f} "
              f"h={ch['h_point']:.4f} understatement={ch['understatement_of_width_pct']:.1f}%")
    print(f"[chain] pass in {time.time()-t3:.1f}s")

    # ---- replicate persistence (added 2026-07-27, no effect on any estimate).
    # Every array below was already computed above; writing it out lets the
    # Monte Carlo error bars and the DeLong cross-check reuse these exact
    # replicates instead of re-running the driver.
    rep_path = os.path.join(HERE, "replicates-redraw.npz")
    np.savez_compressed(
        rep_path,
        paired_fixed_logit=paired_store["logit"][0],
        paired_redraw_logit=paired_store["logit"][1],
        paired_fixed_rf=paired_store["rf"][0],
        paired_redraw_rf=paired_store["rf"][1],
        chain_C_fixed_A=CA_fix, chain_C_fixed_B=CB_fix,
        chain_C_redraw_A=CA_red, chain_C_redraw_B=CB_red,
        chain_vec_fixed_logit=chain_vec_fix["logit"],
        chain_vec_fixed_rf=chain_vec_fix["rf"],
        chain_vec_redraw_logit=chain_vec_red["logit"],
        chain_vec_redraw_rf=chain_vec_red["rf"],
        chain_row_index=chain_idx,
        point_logit=np.array([results["models"]["logit"]["point_fixed_draw"][m]
                              for m in METRICS]),
        point_rf=np.array([results["models"]["rf"]["point_fixed_draw"][m]
                           for m in METRICS]),
        y_test=y_te, p_full_logit=pre["logit"][0], p_full_rf=pre["rf"][0],
    )
    print(f"[replicates] {rep_path}")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    out_path = os.path.join(HERE, "results-redraw.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {out_path}  total {results['runtime_seconds']}s")


if __name__ == "__main__":
    main()
