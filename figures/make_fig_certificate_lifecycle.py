"""Certificate lifecycle mini-diagram (section-8 companion, strict B&W).

Renders the states the gate actually implements (certificate_policy_gate.py):
issue -> active (LCB judged against policy) -> past recalibrate_by_utc ->
deterministic DENY; recalibration (new evidence, retrain, profile change)
loops back to a NEW certificate. No state here is invented: "not yet valid",
"active", and "past recalibrate_by_utc" are the gate's literal checks, and
the recalibration triggers are the certificate's own validity block.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tyche-mpl-safe")
matplotlib.rcParams.update({
    "svg.hashsalt": "safe-cert-lifecycle-2026-08-01",
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
})

fig, ax = plt.subplots(figsize=(7.0, 1.9))
ax.set_xlim(0, 100)
ax.set_ylim(-1.6, 24)
ax.axis("off")

BOXES = [
    (2, "ISSUED", "signed; not valid\nbefore issued_utc"),
    (27, "ACTIVE", "interval vs policy:\nALLOW / ESCALATE / DENY"),
    (52, "PAST\nRECALIBRATE_BY", "validity window\nexceeded"),
    (77, "DETERMINISTIC\nDENY", "gate refuses;\nno interval is read"),
]
W, H, Y = 21, 13, 8

for x, title, sub in BOXES:
    ax.add_patch(Rectangle((x, Y), W, H, fill=False, edgecolor="black",
                           linewidth=1.1))
    ax.text(x + W / 2, Y + H - 3.2, title, ha="center", va="center",
            fontweight="bold", fontsize=8.5)
    ax.text(x + W / 2, Y + 3.6, sub, ha="center", va="center", fontsize=7.2)

for x0 in (2 + W, 27 + W, 52 + W):
    ax.add_patch(FancyArrowPatch((x0, Y + H / 2), (x0 + 4, Y + H / 2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="black", linewidth=1.1))

# recalibration loop: from DENY back along the bottom to a NEW certificate
ax.add_patch(FancyArrowPatch(
    (77 + W / 2, Y), (2 + W / 2, Y),
    connectionstyle="arc3,rad=-0.10", arrowstyle="-|>", mutation_scale=11,
    color="black", linewidth=1.0, linestyle=(0, (3, 2))))
ax.text(50, 0.2, "recalibration: new evidence, retrain, substrate or "
        "profile change  →  a NEW signed certificate",
        ha="center", va="center", fontsize=7.4)

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
for ext in ("svg", "pdf", "png"):
    fig.savefig(HERE / f"fig-certificate-lifecycle.{ext}",
                dpi=300 if ext == "png" else None)
print("wrote fig-certificate-lifecycle.{svg,pdf,png}")
