#!/usr/bin/env python3
"""Render the comparison tables of the second-dataset run as markdown, so that the
summary quotes the results file rather than a transcription of it. Read-only."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results-second-dataset.json")) as fh:
    R = json.load(fh)
G = R["german_credit_anchors"]
TAGS = ["logit", "rf"]
VOL = ["arithmetic", "geometric", "rms", "topsis"]
out = []


def w(s=""):
    out.append(s)


def fx(v, p=4):
    return ("%+.*f" % (p, v)) if v is not None else "n/a"


w("### Structure")
w("")
w("| | German credit | Taiwan |")
w("|:--|--:|--:|")
w("| n_test | %d | %d |" % (G["n_test"], R["n_test"]))
w("| d | %d | %d |" % (G["d"], R["d"]))
w("| curve length L | %d | %d |" % (G["curve_length_L"], R["curve_length_L"]))
w("| tensor cells | %d | %d |" % (G["curve_length_L"] ** 3, R["curve_length_L"] ** 3))
w("| B | %d | %d |" % (G["B"], R["B"]))
w("| B_full logit / rf | %d / %d | %d / %d |"
  % (G["B_conditioning_check"]["logit"], G["B_conditioning_check"]["rf"],
     R["B_conditioning_check"]["logit"], R["B_conditioning_check"]["rf"]))
w("")

w("### P6, the estimator bridge on the RGE-RGR pair")
w("")
w("| rung | construction | German logit | Taiwan logit | German rf | Taiwan rf |")
w("|--:|:--|--:|--:|--:|--:|")
for i in range(6):
    lab = R["models"]["logit"]["estimator_bridge_RGE_RGR"]["rungs"][i]["construction"]
    lab = " ".join(lab.split())
    row = [G["bridge_RGE_RGR"]["logit"][i],
           R["models"]["logit"]["estimator_bridge_RGE_RGR"]["rungs"][i]["correlation"],
           G["bridge_RGE_RGR"]["rf"][i],
           R["models"]["rf"]["estimator_bridge_RGE_RGR"]["rungs"][i]["correlation"]]
    se = [R["models"]["logit"]["estimator_bridge_RGE_RGR"]["rungs"][i]["mc_se"],
          R["models"]["rf"]["estimator_bridge_RGE_RGR"]["rungs"][i]["mc_se"]]
    w("| %d | %s | %s | %s (mc se %.4f) | %s | %s (mc se %.4f) |"
      % (i + 1, lab, fx(row[0]), fx(row[1]), se[0], fx(row[2]), fx(row[3]), se[1]))
w("")

w("### P1, P2, curve-summary correlations")
w("")
w("| quantity | German | Taiwan | Taiwan mc se |")
w("|:--|--:|--:|--:|")
for t in TAGS:
    cg = G["models"][t]["corr_curve_means"]
    ct = R["models"][t]["curve_summaries"]["mean_of_curve"]["Corr"]
    cse = R["models"][t]["curve_summaries"]["mean_of_curve"]["Corr_mc_se"]
    for nm, i, j in (("RGA-RGE", 0, 1), ("RGA-RGR", 0, 2), ("RGE-RGR", 1, 2)):
        key = nm.replace("-", "_")
        w("| corr(curve-mean %s), %s | %s | %s | %.4f |"
          % (nm, t, fx(cg[key]), fx(ct[i][j]), cse.get(key, float("nan"))))
w("")

w("### P3 and P4, the volume composite and the cross-terms")
w("")
w("| arm | model | German delta % | Taiwan delta % (mc se) | German empirical % | "
  "Taiwan empirical % (mc se) |")
w("|:--|:--|--:|--:|--:|--:|")
for t in TAGS:
    for k in VOL:
        a = R["models"][t]["arms"][k]
        w("| %s | %s | %+.2f | %+.2f (%.2f) | %+.2f | %+.2f (%.2f) |"
          % (k, t, G["models"][t]["delta_understatement_pct"][k],
             a["delta_method"]["understatement_of_width_pct"],
             a["delta_method"]["understatement_mc_se_pct"],
             G["models"][t]["empirical_understatement_pct"][k],
             a["empirical_understatement_of_width_pct"],
             a["empirical_understatement_mc_se_pct"]))
w("")

w("### Taiwan volume composite: point values and intervals")
w("")
w("| arm | model | V | paired boot 95% | width | delta 95% measured | "
  "delta 95% cross-terms zero |")
w("|:--|:--|--:|:--|--:|:--|:--|")
for t in TAGS:
    for k in VOL:
        a = R["models"][t]["arms"][k]
        dm = a["delta_method"]
        w("| %s | %s | %.6f | [%.6f, %.6f] | %.6f | [%.6f, %.6f] | [%.6f, %.6f] |"
          % (k, t, a["V_point"], a["paired_bootstrap"]["ci95_percentile"][0],
             a["paired_bootstrap"]["ci95_percentile"][1], a["paired_bootstrap"]["width"],
             dm["ci95_delta_measured_covariance"][0], dm["ci95_delta_measured_covariance"][1],
             dm["ci95_delta_cross_terms_declared_zero"][0],
             dm["ci95_delta_cross_terms_declared_zero"][1]))
w("")

w("### P5, the two-link chain")
w("")
w("| quantity | scheme | German | Taiwan | Taiwan mc se |")
w("|:--|:--|--:|--:|--:|")
for s in ("fixed_draw", "redraw"):
    c = R["chain"][s]
    w("| cross-link correlation | %s | %+.4f | %+.4f | %.4f |"
      % (s, G["chain"][s]["rho"], c["cross_link_correlation"],
         c["cross_link_correlation_mc_se"]))
    w("| width, measured covariance | %s | %.6f | %.6f | |"
      % (s, G["chain"][s]["width_measured"], c["width_measured"]))
    w("| width, cross-terms zero | %s | %.6f | %.6f | |"
      % (s, G["chain"][s]["width_declared_zero"], c["width_declared_zero"]))
    w("| understatement %% | %s | %.2f | %.2f | %.2f |"
      % (s, G["chain"][s]["understatement_pct"], c["understatement_of_width_pct"],
         c["understatement_mc_se_pct"]))
w("")

w("### Per-severity-index RGE-RGR correlation")
w("")
w("| index | German logit | Taiwan logit | German rf | Taiwan rf |")
w("|--:|--:|--:|--:|--:|")
gl = G["models"]["logit"]["per_severity_index_correlation"]["RGE_RGR"]
gr = G["models"]["rf"]["per_severity_index_correlation"]["RGE_RGR"]
tl = R["models"]["logit"]["curve_summaries"]["per_severity_index_correlation"]["RGE_RGR"]
tr = R["models"]["rf"]["curve_summaries"]["per_severity_index_correlation"]["RGE_RGR"]
for i in range(max(len(gl), len(tl))):
    w("| %d | %s | %s | %s | %s |"
      % (i, fx(gl[i], 4) if i < len(gl) and gl[i] is not None else "n/a",
         fx(tl[i], 4) if i < len(tl) and tl[i] is not None else "n/a",
         fx(gr[i], 4) if i < len(gr) and gr[i] is not None else "n/a",
         fx(tr[i], 4) if i < len(tr) and tr[i] is not None else "n/a"))
w("")
for t in TAGS:
    hg = R["models"][t]["curve_summaries"]["half_grid_means"]
    w("Taiwan %s half-grid means: RGE-RGR first %+.4f, second %+.4f; "
      "RGA-RGR first %+.4f, second %+.4f"
      % (t, hg["RGE_RGR"]["mean_first_half"], hg["RGE_RGR"]["mean_second_half"],
         hg["RGA_RGR"]["mean_first_half"], hg["RGA_RGR"]["mean_second_half"]))
    w("")
    w("German %s mean per-index: %s" % (t, G["models"][t]["mean_per_index_correlation"]))
    w("")
    w("Taiwan %s mean per-index: %s"
      % (t, R["models"][t]["curve_summaries"]["mean_per_index_correlation"]))
    w("")

w("### Coding sensitivity, y' = 1 - y")
w("")
w("| model | primary rung 6 | flipped rung 6 | primary rung 4 | flipped rung 4 |")
w("|:--|--:|--:|--:|--:|")
for t in TAGS:
    cs = R["coding_sensitivity"]["models"][t]
    w("| %s | %+.4f | %+.4f | %+.4f | %+.4f |"
      % (t, R["models"][t]["curve_summaries"]["mean_of_curve"]["Corr"][1][2],
         cs["bridge_rung6_curvemeans"],
         R["models"][t]["estimator_bridge_RGE_RGR"]["rungs"][3]["correlation"],
         cs["bridge_rung4_scalarRGE_x_curvemeanRGR"]))
w("")

w("### Conditioning checks")
w("")
w("| model | arm | sd frozen | sd full recompute | ratio | mc band |")
w("|:--|:--|--:|--:|--:|:--|")
for t in TAGS:
    for k in ("arithmetic", "geometric", "rms"):
        c = R["models"][t]["conditioning_check_curve_recompute"]["arms"][k]
        w("| %s | %s | %.6f | %.6f | %.3f | [%.3f, %.3f] |"
          % (t, k, c["sd_frozen"], c["sd_full"], c["sd_ratio_full_over_frozen"],
             c["sd_ratio_mc_band_95"][0], c["sd_ratio_mc_band_95"][1]))
w("")
w("### Scalar construction, both conditioning schemes")
w("")
w("| model | scheme | RGA-RGE | RGA-RGR | RGE-RGR |")
w("|:--|:--|--:|--:|--:|")
for t in TAGS:
    for s in ("fixed_draw", "redraw"):
        c = R["scalar_construction"]["models"][t][s]["Corr"]
        w("| %s | %s | %+.4f | %+.4f | %+.4f |"
          % (t, s, c["corr_RGA_RGE"], c["corr_RGA_RGR"], c["corr_RGE_RGR"]))
w("")
w("### Verdict")
w("")
w("verdict: %s" % R["verdict"]["verdict"])
w("")
for grp in ("generalises_conditions", "failure_conditions", "inconclusive_triggers"):
    w("%s:" % grp)
    for k, v in R["verdict"][grp].items():
        w("  - %s: %s" % (k, v))
    w("")
w("total runtime: %.1f s (%.1f min)" % (R["total_runtime_s"], R["total_runtime_s"] / 60.0))

sys.stdout.write("\n".join(out) + "\n")
