#!/usr/bin/env python3
"""
Probe 5, run after the primary study: is the 300-tree forest's predict_proba
bitwise reproducible at n = 9000, and are its probabilities tie-dense enough for
a floating-point wobble to move a rank functional?

Why this exists. The first independent recomputation aborted on an assertion that
held the forest's point curves to exact agreement with the primary run: they
differed by 1.4e-07 in RGE and 1.2e-08 in RGR, while the logistic model agreed to
0.0e+00. Before relaxing any tolerance we settle by measurement whether that gap
is a defect of either script or a floor of the substrate.

Nothing here is a result of the study. It measures the toolchain.

Anton Sokolov, Tyche Institute, Tallinn. 28 July 2026.
"""
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REDRAW_DIR = os.path.join(os.path.dirname(HERE), "redraw-2026-07-24")
sys.path.insert(0, os.path.join(REDRAW_DIR, "_stubs"))
sys.path.insert(0, os.path.join(REDRAW_DIR, "safeai-src"))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from safeai.rge import rge_score

SEED = 20260723
LOG = []


def say(m=""):
    print(m, flush=True)
    LOG.append(m)


say("Probe 5: is the 300-tree forest's predict_proba bitwise reproducible at n = 9000?")
say("run of %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"))

ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                  parser="liac-arff")
X = ds.data.astype(float)
y = (ds.target == "1").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)

m1 = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
p_a = m1.predict_proba(X_te)[:, 1]
p_b = m1.predict_proba(X_te)[:, 1]
say("same fitted forest, two predict_proba calls, n_jobs=-1: max abs diff = %.3e"
    % np.max(np.abs(p_a - p_b)))
m1.n_jobs = 1
p_s = m1.predict_proba(X_te)[:, 1]
say("same forest, n_jobs=1 vs n_jobs=-1:                    max abs diff = %.3e"
    % np.max(np.abs(p_a - p_s)))
m2 = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
p_c = m2.predict_proba(X_te)[:, 1]
say("a SECOND fit with the same random_state:               max abs diff = %.3e"
    % np.max(np.abs(p_a - p_c)))

col_mean = X_te.mean(axis=0)
Xm = X_te.copy()
Xm[:, 0] = col_mean[0]
q_a = m1.predict_proba(Xm)[:, 1]
m2.n_jobs = 1
q_c = m2.predict_proba(Xm)[:, 1]
co = np.array([0, 1])
s_a = rge_score(p_a, q_a, class_order=co)
s_c = rge_score(p_c, q_c, class_order=co)
say("rge_score on those two probability vectors: %.12f vs %.12f -> gap %.3e"
    % (s_a, s_c, abs(s_a - s_c)))

u = np.unique(np.round(p_a, 12))
say("distinct rounded probabilities among %d rows: %d ; ties are therefore dense"
    % (len(p_a), len(u)))
srt = np.sort(p_a)
d = np.diff(srt)
dp = d[d > 0]
say("smallest positive gap between adjacent sorted probabilities: %.3e" % dp.min())
say("number of adjacent pairs closer than 1e-12: %d" % int((d < 1e-12).sum()))

with open(os.path.join(HERE, "probe-05-rf-determinism.log"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
