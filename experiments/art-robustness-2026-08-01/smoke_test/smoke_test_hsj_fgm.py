"""Smoke test: the whole frozen-artifact ART arm on the paper's real models.

This is the first command Vasily runs in mid-August. It fits the exact
German-credit logistic-regression and random-forest models the deposited
experiments use (same SEED, split, scaler), builds data-fitted feature
constraints, runs HopSkipJump on a small held-out sample for both models,
projects to the legal manifold, and prints per-knot achieved distortion,
validity, and class-flip rate. It touches no bootstrap and writes no paper
result — it proves the plumbing on real data and reports runtime so the full
run can be budgeted.

Run:  python -m smoke_test.smoke_test_hsj_fgm  [N_ROWS]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The package dir name has hyphens, so it cannot be imported by dotted name.
# Put its own path on sys.path and import constraints/ and adapters/ as
# top-level packages (each has __init__.py); the modules' dual-import fallback
# resolves against these.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from constraints import constraints as constraints_m  # noqa: E402
from adapters import base as base_m                    # noqa: E402


SEED = 20260723
TEST_SIZE = 0.3


def fit_models():
    ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "good").astype(int)
    names = list(ds.feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    logit = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    logit.fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return X_tr, X_te, y_tr, y_te, names, {"logit": logit, "rf": rf}


def build_constraints(X_tr, names):
    # First-pass: all continuous, bounds fitted from training data. Vasily/Paolo
    # confirm mutability (Q2) later; this proves projection works end-to-end.
    FeatureSpec = constraints_m.FeatureSpec
    specs = [FeatureSpec(name=nm, kind="continuous") for nm in names]
    fc = constraints_m.FeatureConstraints(specs)
    fc.fit_bounds(X_tr)
    return fc


def run_hsj(model, X, y, fc, model_tag, grid, seed):
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import SklearnClassifier

    clf = SklearnClassifier(model=model)
    attack = HopSkipJump(classifier=clf, targeted=False, norm=2,
                         max_iter=20, max_eval=1000, init_eval=100,
                         batch_size=64, verbose=False)
    t0 = time.time()
    X_adv_top = attack.generate(x=X)
    gen_s = time.time() - t0

    class_order = list(model.classes_)
    p_clean = base_m.summarize_probabilities(model, X, class_order)
    rows = []
    for s in grid:
        X_interp = X + float(s) * (X_adv_top - X)
        X_proj, changed, dist = fc.project_batch(X, X_interp)
        p_adv = base_m.summarize_probabilities(model, X_proj, class_order)
        m, lo, hi, v, fl = base_m.curve_metadata(
            X, X_proj, changed, dist, p_clean, p_adv)
        rows.append({"s": s, "l2_median": round(m, 4),
                     "validity": round(v, 3), "flip_rate": round(fl, 3)})
    return gen_s, rows


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    grid = [0.0, 0.5, 1.0]
    X_tr, X_te, y_tr, y_te, names, models = fit_models()
    fc = build_constraints(X_tr, names)
    Xs = X_te[:n_rows]
    ys = y_te[:n_rows]

    report = {"n_rows": n_rows, "grid": grid, "seed": SEED, "models": {}}
    for tag, model in models.items():
        gen_s, rows = run_hsj(model, Xs, ys, fc, tag, grid, SEED)
        report["models"][tag] = {
            "attack_gen_seconds": round(gen_s, 2),
            "seconds_per_row": round(gen_s / max(n_rows, 1), 3),
            "curve": rows,
        }
        print(f"[{tag}] HSJ generate {gen_s:.2f}s "
              f"({gen_s/max(n_rows,1):.3f}s/row); curve:")
        for r in rows:
            print(f"    s={r['s']:.2f}  l2_med={r['l2_median']:.3f}  "
                  f"validity={r['validity']:.3f}  flip={r['flip_rate']:.3f}")

    out = ROOT / "smoke_test" / "SMOKE-RESULT.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
