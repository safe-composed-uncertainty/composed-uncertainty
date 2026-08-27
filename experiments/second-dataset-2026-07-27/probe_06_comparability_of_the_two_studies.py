"""Comparability probes: is the Taiwan/German difference in the volume-composite
cross-term cost an artefact of the grid length L, or of the sd mix, or is it the
correlations themselves?"""
import json, math
import numpy as np

from pathlib import Path
_EXP = Path(__file__).resolve().parent.parent
D2 = str(_EXP / "second-dataset-2026-07-27")
DP = str(_EXP / "pavia-composite-2026-07-25")
z = np.load(D2 + "/replicates-second-dataset.npz")
tw = json.load(open(D2 + "/results-second-dataset.json"))
ge = json.load(open(DP + "/results-pavia-composite.json"))

def under(S, g):
    Sg = np.cov(S, rowvar=False)
    vm = float(g @ Sg @ g); vd = float(np.sum(g ** 2 * np.diag(Sg)))
    return 100.0 * (1 - math.sqrt(vd) / math.sqrt(vm)), 100.0 * (1 - vd / vm)

print("=" * 96)
print("C1. Does the grid length L drive it? Taiwan curve means restricted to the "
      "first k knots.")
print("=" * 96)
print("%-6s %-4s %10s %10s %10s %10s %10s" %
      ("model", "k", "rho(A,E)", "rho(A,R)", "rho(E,R)", "under%", "cross%"))
for tag in ("logit", "rf"):
    Ab, Eb, Rb = z[f"{tag}_Ab"], z[f"{tag}_Eb"], z[f"{tag}_Rb"]
    for k in (21, 24):
        S = np.column_stack([Ab[:, :k].mean(1), Eb[:, :k].mean(1), Rb[:, :k].mean(1)])
        C = np.corrcoef(S, rowvar=False)
        u, cs = under(S, np.array([1 / 3, 1 / 3, 1 / 3]))
        print("%-6s %-4d %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f"
              % (tag, k, C[0, 1], C[0, 2], C[1, 2], u, cs))
    # a 21-knot subgrid spanning the whole severity range (indices 0..23 -> 21 picks)
    sel = np.unique(np.round(np.linspace(0, 23, 21)).astype(int))
    S = np.column_stack([Ab[:, sel].mean(1), Eb[:, sel].mean(1), Rb[:, sel].mean(1)])
    C = np.corrcoef(S, rowvar=False)
    u, cs = under(S, np.array([1 / 3, 1 / 3, 1 / 3]))
    print("%-6s %-4s %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f"
          % (tag, "21sub", C[0, 1], C[0, 2], C[1, 2], u, cs))

print()
print("=" * 96)
print("C2. Counterfactual decomposition of the arithmetic understatement: swap the "
      "German correlation matrix onto the Taiwan sd mix and vice versa.")
print("=" * 96)
def u_from(C, s, g=np.array([1 / 3, 1 / 3, 1 / 3])):
    Sg = np.outer(s, s) * C
    vm = float(g @ Sg @ g); vd = float(np.sum(g ** 2 * np.diag(Sg)))
    return 100.0 * (1 - math.sqrt(vd) / math.sqrt(vm))

for tag in ("logit", "rf"):
    dg = ge["models"][tag]["arms"]["arithmetic"]["delta_method"]
    dt = tw["models"][tag]["arms"]["arithmetic"]["delta_method"]
    Cg = np.array(dg["Corr_of_summaries"]); Sg_ = np.array(dg["Sigma_of_summaries"])
    Ct = np.array(dt["Corr_of_summaries"]); St_ = np.array(dt["Sigma_of_summaries"])
    sg, st = np.sqrt(np.diag(Sg_)), np.sqrt(np.diag(St_))
    print(f"--- {tag}")
    print("   german corr + german sd  -> %+8.4f  (reported %+8.4f)"
          % (u_from(Cg, sg), dg["understatement_of_width_pct"]))
    print("   taiwan corr + taiwan sd  -> %+8.4f  (reported %+8.4f)"
          % (u_from(Ct, st), dt["understatement_of_width_pct"]))
    print("   TAIWAN corr + german sd  -> %+8.4f   <- correlations alone" % u_from(Ct, sg))
    print("   GERMAN corr + taiwan sd  -> %+8.4f   <- sd mix alone" % u_from(Cg, st))
    print("   sd ratios german %s   taiwan %s"
          % (np.round(sg / sg[0], 4), np.round(st / st[0], 4)))

print()
print("=" * 96)
print("C3. Which pair carries the cross term? 2 g_i g_j Sigma_ij as a share of the "
      "cross total.")
print("=" * 96)
for src, nm in ((ge, "german"), (tw, "taiwan")):
    for tag in ("logit", "rf"):
        cd = src["models"][tag]["arms"]["arithmetic"]["delta_method"]["cross_term_decomposition"]
        print("%-7s %-6s share of cross total: %s   cross share of var %+8.4f"
              % (nm, tag, {k: round(v, 1) for k, v in cd["share_of_cross_total_pct"].items()},
                 src["models"][tag]["arms"]["arithmetic"]["delta_method"]
                 ["cross_covariance_share_of_variance_pct"]))

print()
print("=" * 96)
print("C4. Sign pattern of the three cross contributions (the near-cancellation).")
print("=" * 96)
for src, nm in ((ge, "german"), (tw, "taiwan")):
    for tag in ("logit", "rf"):
        cd = src["models"][tag]["arms"]["arithmetic"]["delta_method"]["cross_term_decomposition"]
        cc = cd["cross_contributions_to_variance"]
        tot = cd["cross_total"]
        pos = sum(v for v in cc.values() if v > 0)
        neg = sum(v for v in cc.values() if v < 0)
        print("%-7s %-6s  positive %+.3e  negative %+.3e  net %+.3e  "
              "|net|/(|pos|+|neg|) = %.3f"
              % (nm, tag, pos, neg, tot, abs(tot) / (abs(pos) + abs(neg))))

print()
print("=" * 96)
print("C5. Is the breach of condition (d) an artefact of the class coding? An "
      "INDICATION only:\n    the flipped-coding correlation matrix applied to the "
      "primary run's sd mix. The flipped\n    covariance itself was not deposited, "
      "so this is not the paired recomputation section 4.3\n    of the "
      "pre-registration asked for.")
print("=" * 96)
for tag in ("logit", "rf"):
    dm = tw["models"][tag]["arms"]["arithmetic"]["delta_method"]
    Cp = np.array(dm["Corr_of_summaries"])
    s = np.sqrt(np.diag(np.array(dm["Sigma_of_summaries"])))
    cf = tw["coding_sensitivity"]["models"][tag]["corr_curve_means"]
    Cf = np.array([[1.0, cf["RGA_RGE"], cf["RGA_RGR"]],
                   [cf["RGA_RGE"], 1.0, cf["RGE_RGR"]],
                   [cf["RGA_RGR"], cf["RGE_RGR"], 1.0]])
    print("%-6s primary correlations -> %+7.3f    flipped correlations on the same "
          "sd mix -> %+7.3f" % (tag, u_from(Cp, s), u_from(Cf, s)))

print()
print("=" * 96)
print("C6. Why rungs 4 and 6 are invariant to relabelling: it is the two-column "
      "class_order\n    convention, not rank-reversal. Synthetic 500-point pair, "
      "p -> 1 - p.")
print("=" * 96)
def _gini(y):
    ys = np.sort(np.asarray(y, float)); c = np.cumsum(ys); n = len(ys)
    return 2 * np.mean(np.abs(np.linspace(1 / n, 1, n) - c / c[-1]))
def _cvm(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float); s = y.sum()
    o = np.argsort(y, kind="stable"); l = np.cumsum(y[o]) / s; w = y[o] / s
    oh = np.argsort(yh, kind="stable")
    return float(np.sum(np.abs(np.cumsum(y[oh]) / s - l) * w))
bare = lambda p, q: 1 - _cvm(p, q) / _gini(p)
co = lambda p, q: 0.5 * ((1 - _cvm(p, q) / _gini(p))
                         + (1 - _cvm(1 - p, 1 - q) / _gini(1 - p)))
rng = np.random.default_rng(7)
p = rng.uniform(0.02, 0.98, 500)
q = np.clip(p + rng.normal(0, 0.05, 500), 0.01, 0.99)
print("  bare 1-D convention   : %.12f -> %.12f   gap %.3e"
      % (bare(p, q), bare(1 - p, 1 - q), abs(bare(p, q) - bare(1 - p, 1 - q))))
print("  class_order=[0,1]     : %.12f -> %.12f   gap %.3e"
      % (co(p, q), co(1 - p, 1 - q), abs(co(p, q) - co(1 - p, 1 - q))))
print("  => the invariance belongs to the two-column convention. Rungs 1 and 3, "
      "which use the\n     bare convention, carry no such guarantee.")

print()
print("=" * 96)
print("C7. How exactly do rungs 4 and 6 survive the coding flip? Primary against "
      "flipped, verbatim.")
print("=" * 96)
for tag in ("logit", "rf"):
    cs = tw["coding_sensitivity"]["models"][tag]
    r4p = [r["correlation"] for r in tw["models"][tag]["estimator_bridge_RGE_RGR"]["rungs"]
           if r["rung"] == 4][0]
    r6p = [r["correlation"] for r in tw["models"][tag]["estimator_bridge_RGE_RGR"]["rungs"]
           if r["rung"] == 6][0]
    r4f = cs["bridge_rung4_scalarRGE_x_curvemeanRGR"]
    r6f = cs["bridge_rung6_curvemeans"]
    print("%-6s rung 4  primary %.16f  flipped %.16f  gap %.3e"
          % (tag, r4p, r4f, abs(r4p - r4f)))
    print("%-6s rung 6  primary %.16f  flipped %.16f  gap %.3e"
          % (tag, r6p, r6f, abs(r6p - r6f)))
