#!/usr/bin/env python3
"""
Step-one companion: what one bootstrap replicate of the three vectors costs.

The naive evaluation re-runs the greedy RGE search inside every replicate, which
for the forest is 26 s a replicate. The fast path freezes the greedy removal ORDER
at the point estimate, precomputes the d+1 nested masked prediction vectors and the
d+1 tail-swapped prediction vectors once on the full test sample, and then a
replicate is nothing but row indexing plus rank statistics.

This probe (a) checks the fast path reproduces the package's own curve functions
exactly at the point estimate, and (b) times a replicate under both paths.
"""
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

from safeai.rga import rga_curve
from safeai.rge import rge_curve, rge_score
from safeai.rgr import rgr_score

sys.path.insert(0, HERE)
from build_metric_vectors import tail_swap, rga_vector, SEED, TEST_SIZE

CLASS_ORDER = np.array([0, 1])


def main():
    ds = fetch_openml("credit-g", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(float)
    y = (ds.target == "good").astype(int)
    names = list(ds.feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    n, d = X_te.shape
    GRID = d

    models = {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)

    for tag, model in models.items():
        print("=== %s ===" % tag)
        p_full = model.predict_proba(X_te)[:, 1]
        col_mean = X_te.mean(axis=0)

        # ---- point estimate with the package, greedy order frozen from it
        t0 = time.time()
        res = rge_curve(model, X_te, method="tabular", feature_names=names,
                        masking_method="greedy", baseline="mean", n_steps=GRID,
                        verbose=False)
        t_greedy = time.time() - t0
        e_pkg = np.asarray(res["rge_scores"], dtype=float)
        order = [names.index(f) for f in res["removed_features"]]

        # ---- precompute: nested masks and tail swaps, once
        t0 = time.time()
        p_mask = np.empty((GRID + 1, n))
        p_mask[0] = p_full
        for t in range(1, GRID + 1):
            Xm = X_te.copy()
            cols = order[:t]
            Xm[:, cols] = col_mean[cols]
            p_mask[t] = model.predict_proba(Xm)[:, 1]
        p_swap = np.empty((GRID + 1, n))
        for t in range(GRID + 1):
            p_swap[t] = model.predict_proba(tail_swap(X_te, 0.5 * t / GRID))[:, 1]
        t_pre = time.time() - t0

        # ---- fast path at the point estimate, checked against the package
        idx = np.arange(n)

        def fast_triple(idx):
            pf = p_full[idx]
            a = rga_vector(y_te[idx], pf, GRID)
            e = np.array([1.0] + [rge_score(pf, p_mask[t][idx], class_order=CLASS_ORDER)
                                  for t in range(1, GRID + 1)])
            r = np.array([rgr_score(pf, p_swap[t][idx], class_order=CLASS_ORDER)
                          for t in range(GRID + 1)])
            return a, e, r

        a0, e0, r0 = fast_triple(idx)
        print("  RGE fast vs package curve, max abs gap: %.3e" % np.max(np.abs(e0 - e_pkg)))

        t0 = time.time()
        for _ in range(20):
            fast_triple(np.random.default_rng(0).integers(0, n, n))
        t_fast = (time.time() - t0) / 20

        print("  greedy RGE at the point estimate: %.2f s" % t_greedy)
        print("  precompute (2*(d+1) predict_proba calls): %.2f s" % t_pre)
        print("  one replicate, fast path: %.4f s" % t_fast)
        for B in (200, 2000):
            print("        B=%-5d fast path total: %.1f s" % (B, t_pre + B * t_fast))


if __name__ == "__main__":
    main()
