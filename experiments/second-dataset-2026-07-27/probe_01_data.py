#!/usr/bin/env python3
"""
Probe 1 of the second-dataset design (article item A11): availability and shape
of UCI Default of Credit Card Clients (Taiwan) through OpenML, profiled with the
preprocessing the pinned German-credit pipeline actually applies.

Design only. No composite is computed here; this probe answers "does it load,
what is in it, and what does the pinned pipeline have to do to it".

Anton Sokolov, Tyche Institute, Tallinn. 27 July 2026.
"""
import json
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
from sklearn.model_selection import train_test_split

SEED = 20260723
TEST_SIZE = 0.3

# The UCI column dictionary, in the order the UCI archive publishes it. The
# OpenML ARFF ships anonymous names x1..x23, so the mapping is asserted against
# measured column statistics below rather than taken on trust.
UCI_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
             "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
             "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
             "BILL_AMT6",
             "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5",
             "PAY_AMT6"]

LOG = []


def say(m=""):
    print(m, flush=True)
    LOG.append(m)


t0 = time.time()
ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                  parser="liac-arff")
t_fetch = time.time() - t0
X = ds.data.astype(float)
y = (ds.target == "1").astype(int)     # 1 = default payment next month
names_arff = list(ds.feature_names)

say("=" * 78)
say("Probe 1: UCI Default of Credit Card Clients (Taiwan) through OpenML")
say("run of %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"))
say("=" * 78)
say("[fetch]  fetch_openml('default-of-credit-card-clients', version=1, "
    "as_frame=False, parser='liac-arff') in %.2f s" % t_fetch)
say("[fetch]  OpenML data id %s, name %s, version %s, format %s, target attr %s"
    % (ds.details.get("id"), ds.details.get("name"), ds.details.get("version"),
       ds.details.get("format"), ds.details.get("default_target_attribute")))
say("[shape]  n = %d, d = %d, dtype %s" % (X.shape[0], X.shape[1], X.dtype))
say("[shape]  ARFF feature names: %s" % " ".join(names_arff))
say("[target] classes %s counts %s -> positive (default) share %.4f"
    % (np.unique(ds.target).tolist(),
       np.unique(ds.target, return_counts=True)[1].tolist(), y.mean()))
say("[nan]    non-finite cells in X: %d ; in y: %d"
    % (int((~np.isfinite(X)).sum()), int((~np.isfinite(y)).sum())))
say("[dup]    exactly duplicated rows: %d"
    % (X.shape[0] - np.unique(X, axis=0).shape[0]))

say("")
say("[columns] measured profile, with the UCI dictionary name asserted against it")
say("  %-3s %-11s %12s %12s %12s %12s %7s %8s"
    % ("col", "UCI name", "min", "median", "max", "sd", "n_uniq", "mode_frac"))
cols = []
for j in range(X.shape[1]):
    v = X[:, j]
    uq, cnt = np.unique(v, return_counts=True)
    rec = {"index": j, "arff_name": names_arff[j], "uci_name": UCI_NAMES[j],
           "min": float(v.min()), "median": float(np.median(v)),
           "max": float(v.max()), "sd": float(v.std(ddof=1)),
           "n_unique": int(uq.size),
           "mode_value": float(uq[int(np.argmax(cnt))]),
           "mode_fraction": float(cnt.max() / v.size),
           "integer_valued": bool(np.all(v == np.round(v)))}
    cols.append(rec)
    say("  %-3d %-11s %12.1f %12.1f %12.1f %12.1f %7d %8.3f"
        % (j, UCI_NAMES[j], rec["min"], rec["median"], rec["max"], rec["sd"],
           rec["n_unique"], rec["mode_fraction"]))

# --- assertions that pin the dictionary mapping to measured facts
checks = {
    "SEX is coded 1/2": sorted(np.unique(X[:, 1]).tolist()) == [1.0, 2.0],
    "AGE lies in [20, 100]": bool(X[:, 4].min() >= 20 and X[:, 4].max() <= 100),
    "EDUCATION is a small integer code": bool(len(np.unique(X[:, 2])) <= 8),
    "MARRIAGE is a small integer code": bool(len(np.unique(X[:, 3])) <= 5),
    "PAY_0..PAY_6 are small signed integer codes":
        bool(all(len(np.unique(X[:, j])) <= 12 and X[:, j].min() >= -2
                 for j in range(5, 11))),
    "LIMIT_BAL is a large positive amount":
        bool(X[:, 0].min() > 0 and X[:, 0].max() > 1e5),
    "BILL_AMT1..6 take negative values (credit balances)":
        bool(all(X[:, j].min() < 0 for j in range(11, 17))),
    "PAY_AMT1..6 are non-negative":
        bool(all(X[:, j].min() >= 0 for j in range(17, 23))),
}
say("")
say("[dictionary] mapping checks")
for k, v in checks.items():
    say("  %-55s %s" % (k, "PASS" if v else "FAIL"))

# --- the split the pinned protocol uses
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
say("")
say("[split]  stratified %d/%d at random_state = %d: train = %d, test = %d"
    % (int(100 * (1 - TEST_SIZE)), int(100 * TEST_SIZE), SEED, len(y_tr), len(y_te)))
say("[split]  positive share train %.4f, test %.4f, positives in test %d"
    % (y_tr.mean(), y_te.mean(), int(y_te.sum())))

# --- what the tail-swap perturbation can actually do here: ties matter.
# For a column with a dominant repeated value, exchanging the outermost m ranks
# can exchange equal values, which is an identity operation on those rows. The
# German-credit columns are ordinal codes too, so this diagnostic is comparable.
def tail_swap(Xin, p):
    n = Xin.shape[0]
    m = int(np.floor(p * n))
    if m == 0:
        return Xin.copy()
    Xp = Xin.copy()
    for j in range(Xin.shape[1]):
        order = np.argsort(Xin[:, j], kind="stable")
        col = Xin[:, j]
        new = col.copy()
        lo, hi = order[:m], order[n - m:][::-1]
        new[lo], new[hi] = col[hi], col[lo]
        Xp[:, j] = new
    return Xp


say("")
say("[tail-swap] effective bite: share of test rows whose value actually changes")
say("  %-11s %8s %8s %8s" % ("UCI name", "p=0.125", "p=0.25", "p=0.5"))
bite = {}
for p in (0.125, 0.25, 0.5):
    Xs = tail_swap(X_te, p)
    bite[p] = (Xs != X_te).mean(axis=0)
for j in range(X.shape[1]):
    say("  %-11s %8.3f %8.3f %8.3f"
        % (UCI_NAMES[j], bite[0.125][j], bite[0.25][j], bite[0.5][j]))
say("  mean over columns:  %8.3f %8.3f %8.3f"
    % (bite[0.125].mean(), bite[0.25].mean(), bite[0.5].mean()))

out = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "openml": {"id": ds.details.get("id"), "name": ds.details.get("name"),
               "version": ds.details.get("version"),
               "fetch_seconds": t_fetch},
    "n": int(X.shape[0]), "d": int(X.shape[1]),
    "positive_share": float(y.mean()),
    "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "positive_share_test": float(y_te.mean()),
    "columns": cols,
    "dictionary_checks": checks,
    "tail_swap_bite_by_p": {str(p): bite[p].tolist() for p in bite},
    "tail_swap_bite_mean": {str(p): float(bite[p].mean()) for p in bite},
}
with open(os.path.join(HERE, "probe-01-data.json"), "w") as fh:
    json.dump(out, fh, indent=1)
with open(os.path.join(HERE, "probe-01-data.log"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")
say("")
say("[done] %.1f s" % (time.time() - t0))
