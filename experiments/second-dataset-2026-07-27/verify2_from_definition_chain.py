"""A THIRD recomputation of the second-dataset study: the two-link chain.

Written during the adversarial verification of 28 July 2026, after the primary run
and after verify_second_dataset.py. It exists because verify_second_dataset.py,
although it shares no code with the driver, imports the same pinned safeai package
for RGA, RGE and RGR -- so it verifies the orchestration and not the estimators.
This script imports no safeai at all: the Lorenz-Gini coefficient, the weighted
Cramer-von Mises distance and all three scores are reimplemented from the published
definitions in the package source, and every index stream is redrawn from the seed
recipe.

Target: the whole chain block under both conditioning schemes -- the cross-link
correlation, h, both widths and the width understatement -- plus the scalar point
triples both links are built from.

Usage: verify2_from_definition_chain.py [B]      (B defaults to 2000)
"""
import json, math, os, sys, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

D = "/srv/tyche/repos/tyche-research-vault/papers/safe-composed-uncertainty/experiment/second-dataset-2026-07-27"
SEED = 20260723
Z = 1.959963984540054
NOISE_SD = 0.5
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
REDRAW_BASE = {"logit": SEED + 100_000, "rf": SEED + 200_000}

t_start = time.time()
def say(m):
    print("[%7.1f] %s" % (time.time() - t_start, m), flush=True)

# ------------------------------------------------------------------ my metrics
def gini(y):
    y = np.asarray(y, float)
    ys = np.sort(y)
    cum = np.cumsum(ys)
    s = cum[-1]
    if s == 0:
        return np.nan
    l = cum / s
    n = len(y)
    u = np.linspace(1.0 / n, 1.0, n)
    return 2.0 * np.mean(np.abs(u - l))

def _lw(y):
    """Lorenz curve and weights of the reference vector, shared across partners."""
    s = y.sum()
    o = np.argsort(y, kind="stable")
    ys = y[o]
    return np.cumsum(ys) / s, ys / s, s

def cvm_pre(y, l, w, s, yhat):
    o = np.argsort(yhat, kind="stable")
    c = np.cumsum(y[o]) / s
    return float(np.sum(np.abs(c - l) * w))

def cvm(y, yhat):
    l, w, s = _lw(np.asarray(y, float))
    return cvm_pre(np.asarray(y, float), l, w, s, np.asarray(yhat, float))

def rga(y_true, y_score):
    y = np.asarray(y_true, float)
    g = gini(y)
    if not np.isfinite(g) or g == 0:
        return np.nan
    return 1.0 - cvm(y, y_score) / g

class Ref:
    """Precomputed reference-vector state for one prediction vector, both the
    bare-vector convention and the two-column class_order convention."""
    __slots__ = ("p", "l", "w", "s", "g", "q", "lq", "wq", "sq", "gq")
    def __init__(self, p):
        self.p = p
        self.l, self.w, self.s = _lw(p)
        self.g = gini(p)
        q = 1.0 - p
        self.q = q
        self.lq, self.wq, self.sq = _lw(q)
        self.gq = gini(q)

def score_bare(ref, other):
    """1 - CvM(ref, other)/Gini(ref): this is both rge_score and rgr_score in the
    bare 1-D convention."""
    return 1.0 - cvm_pre(ref.p, ref.l, ref.w, ref.s, other) / ref.g

def score_co(ref, other):
    """The class_order=[0,1] convention: mean of the one-vs-rest scores on the two
    probability columns [1-p, p]."""
    k1 = 1.0 - cvm_pre(ref.p, ref.l, ref.w, ref.s, other) / ref.g
    k0 = 1.0 - cvm_pre(ref.q, ref.lq, ref.wq, ref.sq, 1.0 - other) / ref.gq
    return 0.5 * (k0 + k1)

def make_segments(n, k):
    size, rem = n // k, n % k
    out, start = [], 0
    for i in range(k):
        end = start + size + (1 if i < rem else 0)
        out.append((start, end))
        start = end
    return out

def rga_partial_curve(y_true, y_score, n_segments):
    """safeai._binary_rga_curve_partial with normalize_to_perfect=False."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_score, float)
    n = len(y)
    full_rga = rga(y, p)
    full_gini = gini(y)
    order = np.argsort(p)[::-1]          # package uses the default sort, then reverses
    ys, ss = y[order], p[order]
    partial = []
    for a, b in make_segments(n, n_segments):
        yseg, sseg = ys[a:b], ss[a:b]
        gseg = gini(yseg)
        rseg = rga(yseg, sseg)
        if np.isfinite(rseg) and np.isfinite(gseg) and gseg > 0:
            partial.append(rseg * gseg * (len(yseg) / n) / full_gini)
        else:
            partial.append(0.0)
    partial = np.asarray(partial, float)
    tot = partial.sum()
    if tot > 0:
        partial = partial * (full_rga / tot)
    curve = np.zeros(n_segments + 1)
    curve[0] = full_rga
    acc = 0.0
    for i in range(n_segments):
        acc += partial[i]
        curve[i + 1] = full_rga - acc
    bad = np.where(~np.isfinite(curve))[0]
    if len(bad):
        curve[bad[0]:] = 0.0
    return curve

def tail_swap_perm(X, p):
    """The eq.-(9) perturbation, built as an explicit permutation of the row order
    per column rather than by fancy-index assignment."""
    n = X.shape[0]
    m = int(np.floor(p * n))
    Xp = X.copy()
    if m == 0:
        return Xp
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable")
        perm = o.copy()
        perm[:m] = o[n - 1:n - m - 1:-1]
        perm[n - m:] = o[m - 1::-1]
        Xp[:, j] = X[perm, j]
    return Xp

# ------------------------------------------------------------------ data + models
say("loading OpenML 42477")
ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                  parser="liac-arff")
X = ds.data.astype(float)
y = (ds.target == "1").astype(int)
names = list(ds.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
n, d = X_te.shape
GRID, L = d, d + 1
say("n_test=%d d=%d positive share %.6f" % (n, d, y_te.mean()))

models = {"logit": make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, random_state=SEED)),
          "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)}
for m in models.values():
    m.fit(X_tr, y_tr)
say("models fitted")

col_mean = X_te.mean(axis=0)
col_std = X_te.std(axis=0, keepdims=True)
ps = 0.5 * np.arange(L) / GRID

# the shared deposit stream: logit draws first, then rf, exactly as the driver
dep_rng = np.random.default_rng(SEED)

PRE = {}
for tag in ("logit", "rf"):
    model = models[tag]
    p_full = model.predict_proba(X_te)[:, 1]
    p_mask_single = np.empty((d, n))
    for j in range(d):
        Xm = X_te.copy(); Xm[:, j] = col_mean[j]
        p_mask_single[j] = model.predict_proba(Xm)[:, 1]
    pert_fixed = dep_rng.normal(0.0, NOISE_SD, size=X_te.shape) * col_std
    p_pert_fixed = model.predict_proba(X_te + pert_fixed)[:, 1]
    PRE[tag] = dict(p_full=p_full, p_mask_single=p_mask_single,
                    p_pert_fixed=p_pert_fixed)
    say("%s scalar precompute done" % tag)

npz = np.load(D + "/replicates-second-dataset.npz")
res = json.load(open(D + "/results-second-dataset.json"))
OUT = {}

def rep(label, mine, theirs, tol):
    gap = abs(mine - theirs)
    ok = gap <= tol
    print("%s %-62s mine=%+.10f  run=%+.10f  gap=%.2e (tol %.0e)"
          % ("PASS" if ok else "FAIL", label, mine, theirs, gap, tol), flush=True)
    OUT[label] = dict(mine=mine, run=theirs, gap=gap, tol=tol, ok=bool(ok))
    return ok

# sanity: does my y_test match the deposited one?
rep("y_test identical to the run", float(np.abs(y_te - npz["y_test"]).max()), 0.0, 0.0)

# ------------------------------------------------------------------ geo mean
def geo_mean(v):
    return float(np.exp(np.mean(np.log(np.clip(np.asarray(v, float), 1e-12, None)))))

# ------------------------------------------------------------------ scalar point
point = {}
for tag in ("logit", "rf"):
    P = PRE[tag]
    ref = Ref(P["p_full"])
    v = np.array([rga(y_te, P["p_full"]),
                  float(np.mean([score_bare(ref, P["p_mask_single"][j]) for j in range(d)])),
                  score_bare(ref, P["p_pert_fixed"])])
    point[tag] = v
    js = res["scalar_construction"]["models"][tag]["point_fixed_draw"]
    tol = 0.0 if tag == "logit" else 1e-6
    rep("%s point scalar RGA" % tag, v[0], js["RGA"], tol)
    rep("%s point scalar RGE" % tag, v[1], js["RGE"], tol)
    rep("%s point scalar RGR" % tag, v[2], js["RGR"], tol)
    rep("%s point C geomean" % tag, geo_mean(v), js["C_geomean"], tol)

# ------------------------------------------------------------------ the chain
say("redraw perturbation predictions")
p_pert_redraw = {}
for tag in ("logit", "rf"):
    mat = np.empty((B, n))
    base = REDRAW_BASE[tag]
    for b in range(B):
        rb = np.random.default_rng(base + b)
        mat[b] = models[tag].predict_proba(
            X_te + rb.normal(0.0, NOISE_SD, size=X_te.shape) * col_std)[:, 1]
    p_pert_redraw[tag] = mat
    say("  %s redraw predictions done" % tag)

say("chain replicates")
shared = np.random.default_rng(SEED + 99)
CA_fix = np.empty(B); CB_fix = np.empty(B)
CA_red = np.empty(B); CB_red = np.empty(B)
for b in range(B):
    idx = shared.integers(0, n, n)
    yb = y_te[idx]
    for tag, af, ar in (("logit", CA_fix, CA_red), ("rf", CB_fix, CB_red)):
        P = PRE[tag]
        pb = P["p_full"][idx]
        ref = Ref(pb)
        vec = np.array([rga(yb, pb),
                        float(np.mean([score_bare(ref, P["p_mask_single"][j][idx])
                                       for j in range(d)])),
                        score_bare(ref, P["p_pert_fixed"][idx])])
        af[b] = geo_mean(vec)
        ar[b] = geo_mean([vec[0], vec[1], score_bare(ref, p_pert_redraw[tag][b][idx])])
    if (b + 1) % 500 == 0:
        say("  chain %d/%d" % (b + 1, B))

pA, pB = geo_mean(point["logit"]), geo_mean(point["rf"])
for scheme, CA, CB in (("fixed_draw", CA_fix, CB_fix), ("redraw", CA_red, CB_red)):
    ch = res["chain"][scheme]
    S = np.cov(np.vstack([CA, CB]))
    rho = float(np.corrcoef(CA, CB)[0, 1])
    h = pA * pB
    g = np.array([pB, pA])
    vm = float(g @ S @ g); vd = float(g[0] ** 2 * S[0, 0] + g[1] ** 2 * S[1, 1])
    wm, wd = 2 * Z * math.sqrt(vm), 2 * Z * math.sqrt(vd)
    rep("chain %s rho" % scheme, rho, ch["cross_link_correlation"], 1e-3)
    rep("chain %s h_point" % scheme, h, ch["h_point"], 1e-6)
    rep("chain %s width measured" % scheme, wm, ch["width_measured"], 1e-6)
    rep("chain %s width declared zero" % scheme, wd, ch["width_declared_zero"], 1e-6)
    rep("chain %s understatement pct" % scheme, 100 * (1 - wd / wm),
        ch["understatement_of_width_pct"], 1e-2)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "verify2-from-definition-chain.json"), "w"),
          indent=1)
say("chain block done")
