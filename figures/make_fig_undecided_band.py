"""Centerpiece figure: the UNDECIDED band and the more-evidence beat.

Panel (a): the recorded real-chain certificate interval [0.7683, 0.8469]
(point 0.8076, first_order_delta) evaluated against three policy bounds —
0.75 (ALLOW: lower bound clears), 0.79 (ESCALATE: interval straddles), and
0.86 (DENY: upper bound below policy). Same certificate, three verdicts:
the verdict is a property of the interval-policy pair, not of the point.

Panel (b): the more-evidence beat (MORE-EVIDENCE-BEAT.json) — same system,
same fixed 0.75 policy; n=30 evaluation rows give a wide straddling interval
(ESCALATE), n=300 narrow it and the lower bound clears (ALLOW).

Strict black-and-white house style: black on white, no colour, no rounded
corners, hatching for zones. Every number is read from the recorded
artifacts, never typed in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiment"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tyche-mpl-safe")
matplotlib.rcParams.update({
    "svg.hashsalt": "safe-undecided-band-2026-08-01",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

# ---- recorded inputs (read, not typed) ---------------------------------
cert = json.loads((EXP / "certificate-examples" / "real-chain.json").read_text())
score = cert["certificate"]["measurement"]["composed_score"]
POINT = score["value"]
LCB, UCB = score["interval95"]

beat = json.loads(
    (EXP / "certificate-enforcement-2026-07-31" / "MORE-EVIDENCE-BEAT.json")
    .read_text())
POL = beat["policy_minimum_lower_bound"]
runs = beat["runs"]

BLACK = "black"
WHITE = "white"


def whisker(ax, y, lo, hi, point, label_left):
    ax.plot([lo, hi], [y, y], color=BLACK, lw=1.4, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.14, y + 0.14], color=BLACK, lw=1.4)
    ax.plot([point], [y], marker="o", color=BLACK, ms=4.5, mfc=WHITE, mew=1.2)
    ax.text(-0.012, y, label_left, ha="right", va="center", fontsize=8.5,
            transform=ax.get_yaxis_transform())


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.0, 4.4), sharex=True,
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.42})

# ---- panel (a): one certificate, three policies ------------------------
policies = [(0.75, "ALLOW"), (0.79, "ESCALATE"), (0.86, "DENY")]
for row, (tau, verdict) in enumerate(policies):
    y = len(policies) - row
    whisker(ax1, y, LCB, UCB, POINT,
            f"policy LCB ≥ {tau:.2f}")
    ax1.plot([tau, tau], [y - 0.30, y + 0.30], color=BLACK, lw=1.0,
             linestyle=(0, (2, 2)))
    ax1.text(1.005, y, verdict, ha="left", va="center", fontsize=9,
             fontweight="bold", transform=ax1.get_yaxis_transform())
ax1.set_ylim(0.4, len(policies) + 0.6)
ax1.set_yticks([])
ax1.set_title(
    f"(a) One certificate [{LCB:.3f}, {UCB:.3f}] — "
    "the verdict is the interval–policy pair",
    fontsize=9, loc="left")

# ---- panel (b): the more-evidence beat, fixed policy -------------------
order = [("small", 2), ("full", 1)]
for tag, y in order:
    r = runs[tag]
    lo, hi = r["interval95"]
    whisker(ax2, y, lo, hi, r["h_point"],
            f"n = {r['n_eval_rows']} evaluation rows")
    ax2.text(1.005, y, r["verdict"], ha="left", va="center", fontsize=9,
             fontweight="bold", transform=ax2.get_yaxis_transform())
ax2.plot([POL, POL], [0.55, 2.45], color=BLACK, lw=1.0, linestyle=(0, (2, 2)))
ax2.text(POL, 2.52, f"fixed policy {POL:.2f}", ha="center", va="bottom",
         fontsize=8)
ax2.set_ylim(0.4, 2.95)
ax2.set_yticks([])
ax2.set_title(
    "(b) Same system, same policy — only the amount of evidence changes",
    fontsize=9, loc="left")
ax2.set_xlabel("composed score", fontsize=9)

ax2.set_xlim(0.55, 1.0)
for ax in (ax1, ax2):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

fig.subplots_adjust(left=0.22, right=0.86, top=0.92, bottom=0.13)
for ext in ("svg", "pdf", "png"):
    fig.savefig(HERE / f"fig-undecided-band.{ext}",
                dpi=300 if ext == "png" else None,
                metadata=None if ext == "png" else {})
print("wrote fig-undecided-band.{svg,pdf,png}")
print(f"(a) interval [{LCB:.4f}, {UCB:.4f}] point {POINT:.4f}; "
      f"(b) small n={runs['small']['n_eval_rows']} {runs['small']['verdict']}, "
      f"full n={runs['full']['n_eval_rows']} {runs['full']['verdict']}")
