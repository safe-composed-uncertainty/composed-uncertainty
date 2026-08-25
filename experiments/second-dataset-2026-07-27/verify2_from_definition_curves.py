"""A THIRD recomputation of the second-dataset study: the logistic curve pass.

Written during the adversarial verification of 28 July 2026, after the primary run
and after verify_second_dataset.py, and for the same reason as its chain sibling:
the deposited verification imports the pinned safeai package for the estimators, so
it cannot catch an error inside them. This script imports no safeai. It rebuilds
the RGA partial curve, the RGE greedy removal order, the tail swap (as a
source-index gather rather than a fancy-index assignment), the Lorenz-Gini
coefficient and the weighted Cramer-von Mises distance from the published
definitions, redraws default_rng(SEED + 1) itself, and then checks:

  the three point curves and the terminal structural zero;
  the six replicate matrices Ab, Eb, Rb, Rb_noise, SC, SC_co, element by element;
  the full 3 x 3 curve-summary correlation matrix and its covariance;
  all six estimator-bridge rungs and both RGA endpoints;
  the four delta-method width understatements and their cross shares;
  the whole 24-knot per-severity-index correlation vector.

Usage: verify2_from_definition_curves.py [B]      (B defaults to 2000)
"""
import json, math, os, sys, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

D = "/srv/tyche/repos/tyche-research-vault/papers/safe-composed-uncertainty/experiment/second-dataset-2026-07-27"
SEED = 20260723
Z = 1.959963984540054
NOISE_SD = 0.5
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

t0 = time.time()
def say(m):
    print("[%7.1f] %s" % (time.time() - t0, m), flush=True)

# ---- my own metric kernel (identical to indep_recompute.py, restated here so the
#      script stands alone)
def gini(y):
    ys = np.sort(np.asarray(y, float)); cum = np.cumsum(ys); s = cum[-1]
    if s == 0: return np.nan
    n = len(ys); u = np.linspace(1.0 / n, 1.0, n)
    return 2.0 * np.mean(np.abs(u - cum / s))

def _lw(y):
    s = y.sum(); o = np.argsort(y, kind="stable"); ys = y[o]
    return np.cumsum(ys) / s, ys / s, s

def cvm_pre(y, l, w, s, yhat):
    o = np.argsort(yhat, kind="stable")
    return float(np.sum(np.abs(np.cumsum(y[o]) / s - l) * w))

def rga(y_true, y_score):
    y = np.asarray(y_true, float); g = gini(y)
    if not np.isfinite(g) or g == 0: return np.nan
    l, w, s = _lw(y)
    return 1.0 - cvm_pre(y, l, w, s, np.asarray(y_score, float)) / g

class Ref:
    __slots__ = ("p", "l", "w", "s", "g", "q", "lq", "wq", "sq", "gq")
    def __init__(self, p):
        self.p = p; self.l, self.w, self.s = _lw(p); self.g = gini(p)
        q = 1.0 - p; self.q = q; self.lq, self.wq, self.sq = _lw(q); self.gq = gini(q)

def score_bare(ref, other):
    return 1.0 - cvm_pre(ref.p, ref.l, ref.w, ref.s, other) / ref.g

def score_co(ref, other):
    k1 = 1.0 - cvm_pre(ref.p, ref.l, ref.w, ref.s, other) / ref.g
    k0 = 1.0 - cvm_pre(ref.q, ref.lq, ref.wq, ref.sq, 1.0 - other) / ref.gq
    return 0.5 * (k0 + k1)

def make_segments(n, k):
    size, rem = n // k, n % k
    out, start = [], 0
    for i in range(k):
        end = start + size + (1 if i < rem else 0); out.append((start, end)); start = end
    return out

def rga_partial_curve(y_true, y_score, n_segments, segs):
    y = np.asarray(y_true, float); p = np.asarray(y_score, float); n = len(y)
    full_rga = rga(y, p); full_gini = gini(y)
    order = np.argsort(p)[::-1]
    ys, ss = y[order], p[order]
    partial = np.empty(n_segments)
    for i, (a, b) in enumerate(segs):
        yseg, sseg = ys[a:b], ss[a:b]
        gseg = gini(yseg); rseg = rga(yseg, sseg)
        partial[i] = (rseg * gseg * (len(yseg) / n) / full_gini
                      if (np.isfinite(rseg) and np.isfinite(gseg) and gseg > 0) else 0.0)
    tot = partial.sum()
    if tot > 0: partial = partial * (full_rga / tot)
    curve = np.empty(n_segments + 1); curve[0] = full_rga
    acc = 0.0
    for i in range(n_segments):
        acc += partial[i]; curve[i + 1] = full_rga - acc
    bad = np.where(~np.isfinite(curve))[0]
    if len(bad): curve[bad[0]:] = 0.0
    return curve

def tail_swap_perm(X, p):
    """Rebuilt as an explicit source-index gather: row of rank i takes the value of
    the row of rank n-1-i, for the outermost m = floor(p*n) ranks on each side."""
    n = X.shape[0]; m = int(np.floor(p * n)); Xp = X.copy()
    if m == 0: return Xp
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable")
        lo = o[:m]                       # rows of the m lowest ranks
        hi = o[n - 1:n - m - 1:-1]       # hi[i] = row of rank n-1-i
        src = np.arange(n)
        src[lo] = hi
        src[hi] = lo
        Xp[:, j] = X[src, j]
    return Xp

# ---- data and the logistic model
ds = fetch_openml("default-of-credit-card-clients", version=1, as_frame=False,
                  parser="liac-arff")
X = ds.data.astype(float); y = (ds.target == "1").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
n, d = X_te.shape; GRID, L = d, d + 1
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                           random_state=SEED))
model.fit(X_tr, y_tr)
say("model fitted, n=%d d=%d" % (n, d))

col_mean = X_te.mean(axis=0); col_std = X_te.std(axis=0, keepdims=True)
ps = 0.5 * np.arange(L) / GRID
p_full = model.predict_proba(X_te)[:, 1]
ref_full = Ref(p_full)

# ---- my own greedy removal order (max RGE at each step, ascending scan, strict >)
removed, remaining, e0 = [], list(range(d)), [1.0]
for step in range(1, GRID + 1):
    best_j, best_v = None, -np.inf
    for j in remaining:
        Xm = X_te.copy(); Xm[:, removed + [j]] = col_mean[removed + [j]]
        v = score_co(ref_full, model.predict_proba(Xm)[:, 1])
        cand = -np.inf if not np.isfinite(v) else float(v)
        if cand > best_v:
            best_v, best_j = cand, j
    removed.append(best_j); remaining.remove(best_j); e0.append(best_v)
e0 = np.asarray(e0)
say("greedy order: %s" % removed[:6])

segs = make_segments(n, GRID)
a0 = rga_partial_curve(y_te, p_full, GRID, segs)
r0 = np.array([score_co(ref_full, model.predict_proba(tail_swap_perm(X_te, ps[t]))[:, 1])
               for t in range(L)])

npz = np.load(D + "/replicates-second-dataset.npz")
res = json.load(open(D + "/results-second-dataset.json"))
OUT = {}
def rep(label, mine, theirs, tol):
    gap = abs(mine - theirs); ok = gap <= tol
    print("%s %-62s mine=%+.10f  run=%+.10f  gap=%.2e (tol %.0e)"
          % ("PASS" if ok else "FAIL", label, mine, theirs, gap, tol), flush=True)
    OUT[label] = dict(mine=float(mine), run=float(theirs), gap=float(gap),
                      tol=tol, ok=bool(ok))

rep("logit point curve RGA max abs gap", float(np.max(np.abs(a0 - npz["logit_a0"]))), 0.0, 1e-12)
rep("logit point curve RGE max abs gap", float(np.max(np.abs(e0 - npz["logit_e0"]))), 0.0, 1e-12)
rep("logit point curve RGR max abs gap", float(np.max(np.abs(r0 - npz["logit_r0"]))), 0.0, 1e-12)
rep("logit terminal RGA knot (structural zero)", float(a0[-1]),
    float(res["models"]["logit"]["last_rga_knot"]["point_value"]), 1e-20)
say("point curves verified")

# ---- precompute the frozen fast-path matrices
p_mask = np.empty((L, n)); p_mask[0] = p_full
for t in range(1, L):
    Xm = X_te.copy(); cols = removed[:t]; Xm[:, cols] = col_mean[cols]
    p_mask[t] = model.predict_proba(Xm)[:, 1]
p_swap = np.array([model.predict_proba(tail_swap_perm(X_te, ps[t]))[:, 1] for t in range(L)])
rng_noise = np.random.default_rng(SEED)
p_noise = np.empty((L, n))
for t in range(L):
    sigma = t / float(GRID)
    Xp = (X_te + rng_noise.normal(0.0, sigma, size=X_te.shape) * col_std) if sigma > 0 else X_te
    p_noise[t] = model.predict_proba(Xp)[:, 1]
p_mask_single = np.empty((d, n))
for j in range(d):
    Xm = X_te.copy(); Xm[:, j] = col_mean[j]
    p_mask_single[j] = model.predict_proba(Xm)[:, 1]
dep_rng = np.random.default_rng(SEED)       # logit draws first
p_pert_fixed = model.predict_proba(
    X_te + dep_rng.normal(0.0, NOISE_SD, size=X_te.shape) * col_std)[:, 1]
say("fast-path matrices built")

# ---- the paired pass on my own regeneration of default_rng(SEED+1)
boot = np.random.default_rng(SEED + 1)
IDX = np.empty((B, n), dtype=np.int64)
for b in range(B):
    IDX[b] = boot.integers(0, n, n)
Ab = np.empty((B, L)); Eb = np.empty((B, L)); Rb = np.empty((B, L)); Nb = np.empty((B, L))
SC = np.empty((B, 3)); SCco = np.empty((B, 2))
for b in range(B):
    ib = IDX[b]
    pb = p_full[ib]; yb = y_te[ib]
    ref = Ref(pb)
    Ab[b] = rga_partial_curve(yb, pb, GRID, segs)
    Eb[b, 0] = 1.0
    for t in range(1, L):
        Eb[b, t] = score_co(ref, p_mask[t][ib])
    for t in range(L):
        Rb[b, t] = score_co(ref, p_swap[t][ib])
        Nb[b, t] = score_co(ref, p_noise[t][ib])
    SC[b, 0] = rga(yb, pb)
    SC[b, 1] = np.mean([score_bare(ref, p_mask_single[j][ib]) for j in range(d)])
    SC[b, 2] = score_bare(ref, p_pert_fixed[ib])
    SCco[b, 0] = np.mean([score_co(ref, p_mask_single[j][ib]) for j in range(d)])
    SCco[b, 1] = score_co(ref, p_pert_fixed[ib])
    if (b + 1) % 250 == 0:
        say("  paired %d/%d" % (b + 1, B))
say("paired pass done")

for nm, mine_, run_ in (("Ab", Ab, npz["logit_Ab"]), ("Eb", Eb, npz["logit_Eb"]),
                        ("Rb", Rb, npz["logit_Rb"]), ("Rb_noise", Nb, npz["logit_Rb_noise"]),
                        ("SC", SC, npz["logit_SC"]), ("SC_co", SCco, npz["logit_SC_co"])):
    rep("logit replicate matrix %s max abs gap" % nm,
        float(np.max(np.abs(mine_ - run_[:B]))), 0.0, 1e-12)

# ---- the full 3x3 curve-summary correlation matrix
S = np.column_stack([Ab.mean(1), Eb.mean(1), Rb.mean(1)])
C = np.corrcoef(S, rowvar=False)
Cj = np.array(res["models"]["logit"]["arms"]["arithmetic"]["delta_method"]["Corr_of_summaries"])
for i, j, lab in ((0, 1, "RGA-RGE"), (0, 2, "RGA-RGR"), (1, 2, "RGE-RGR = rung 6")):
    rep("logit curve-summary corr %s" % lab, float(C[i, j]), float(Cj[i, j]), 1e-12)
Sig = np.cov(S, rowvar=False)
Sigj = np.array(res["models"]["logit"]["arms"]["arithmetic"]["delta_method"]["Sigma_of_summaries"])
rep("logit curve-summary Sigma max abs gap", float(np.max(np.abs(Sig - Sigj))), 0.0, 1e-18)

# ---- the six bridge rungs
br = {r["rung"]: r["correlation"] for r in
      res["models"]["logit"]["estimator_bridge_RGE_RGR"]["rungs"]}
pairs = {1: (SC[:, 1], SC[:, 2]), 2: (SCco[:, 0], SCco[:, 1]),
         3: (Eb.mean(1), SCco[:, 1]), 4: (SCco[:, 0], Rb.mean(1)),
         5: (Eb.mean(1), Nb.mean(1)), 6: (Eb.mean(1), Rb.mean(1))}
for k in range(1, 7):
    u, v = pairs[k]
    rep("logit bridge rung %d" % k, float(np.corrcoef(u, v)[0, 1]), br[k], 1e-12)

# ---- RGA endpoints
ep = res["models"]["logit"]["estimator_bridge_RGA_endpoints"]
rep("logit RGA-RGE scalars", float(np.corrcoef(SC[:, 0], SC[:, 1])[0, 1]),
    ep["RGA_RGE_scalars"], 1e-12)
rep("logit RGA-RGR scalars", float(np.corrcoef(SC[:, 0], SC[:, 2])[0, 1]),
    ep["RGA_RGR_scalars"], 1e-12)
rep("logit RGA-RGE curve means", float(C[0, 1]), ep["RGA_RGE_curve_means"], 1e-12)
rep("logit RGA-RGR curve means", float(C[0, 2]), ep["RGA_RGR_curve_means"], 1e-12)

# ---- the four delta-method understatements
def grad_rms_entries(a, e, r):
    Lv = len(a)
    M = np.sqrt((a[:, None, None] ** 2 + e[None, :, None] ** 2 + r[None, None, :] ** 2) / 3.0)
    inv = 1.0 / M; c = 1.0 / (3.0 * Lv ** 3)
    return (a * inv.sum(axis=(1, 2)) * c, e * inv.sum(axis=(0, 2)) * c,
            r * inv.sum(axis=(0, 1)) * c)

def topsis_reference(Lv):
    return (np.concatenate([np.ones(Lv), np.concatenate([[1.0], np.zeros(Lv - 1)]), np.ones(Lv)]),
            np.concatenate([np.zeros(Lv), np.linspace(1.0, 0.5, Lv),
                            np.concatenate([[1.0], np.zeros(Lv - 1)])]))

def grad_topsis_entries(a, e, r, pis, nis, wvec):
    v = np.concatenate([a, e, r]); w2 = wvec ** 2
    sp = math.sqrt(float(((v - pis) ** 2 * w2).sum()))
    sm = math.sqrt(float(((v - nis) ** 2 * w2).sum()))
    g = (sp * (w2 * (v - nis) / sm) - sm * (w2 * (v - pis) / sp)) / (sm + sp) ** 2
    Lv = len(a); return g[:Lv], g[Lv:2 * Lv], g[2 * Lv:]

pis, nis = topsis_reference(L); wvec = np.concatenate([np.full(L, 1 / 3)] * 3)
ga, ge, gr = grad_rms_entries(a0, e0, r0)
gta, gte, gtr = grad_topsis_entries(a0, e0, r0, pis, nis, wvec)

def under(Sb, g):
    Sg = np.cov(Sb, rowvar=False)
    vm = float(g @ Sg @ g); vd = float(np.sum(g ** 2 * np.diag(Sg)))
    return 100.0 * (1 - math.sqrt(vd) / math.sqrt(vm)), 100.0 * (1 - vd / vm)

arms = res["models"]["logit"]["arms"]
S_ar = np.column_stack([Ab.mean(1), Eb.mean(1), Rb.mean(1)])
S_ge = np.column_stack([np.cbrt(Ab).mean(1), np.cbrt(Eb).mean(1), np.cbrt(Rb).mean(1)])
S0_ge = np.array([np.cbrt(a0).mean(), np.cbrt(e0).mean(), np.cbrt(r0).mean()])
S_rm = np.column_stack([Ab @ ga, Eb @ ge, Rb @ gr])
S_tp = np.column_stack([Ab @ gta, Eb @ gte, Rb @ gtr])
for nm, Sb, g in (("arithmetic", S_ar, np.array([1 / 3, 1 / 3, 1 / 3])),
                  ("geometric", S_ge, np.array([S0_ge[1] * S0_ge[2], S0_ge[0] * S0_ge[2],
                                                S0_ge[0] * S0_ge[1]])),
                  ("rms", S_rm, np.ones(3)), ("topsis", S_tp, np.ones(3))):
    u, cs = under(Sb, g)
    rep("logit delta understatement %% %s" % nm, u,
        arms[nm]["delta_method"]["understatement_of_width_pct"], 1e-9)
    rep("logit cross share of variance %% %s" % nm, cs,
        arms[nm]["delta_method"]["cross_covariance_share_of_variance_pct"], 1e-9)

# ---- per-severity-index RGE-RGR
per = res["models"]["logit"]["curve_summaries"]["per_severity_index_correlation"]["RGE_RGR"]
mine = []
for t in range(L):
    if Eb[:, t].std() <= 1e-12 or Rb[:, t].std() <= 1e-12:
        mine.append(None)
    else:
        mine.append(float(np.corrcoef(Eb[:, t], Rb[:, t])[0, 1]))
gaps = [abs(m - r) for m, r in zip(mine, per) if m is not None and r is not None]
rep("logit per-severity-index RGE-RGR max abs gap", max(gaps), 0.0, 1e-12)
rep("logit per-index None positions agree",
    float(sum(1 for m, r in zip(mine, per) if (m is None) != (r is None))), 0.0, 0.0)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "verify2-from-definition-curves.json"), "w"),
          indent=1)
nfail = sum(1 for v in OUT.values() if not v["ok"])
say("DONE: %d checks, %d failures" % (len(OUT), nfail))
