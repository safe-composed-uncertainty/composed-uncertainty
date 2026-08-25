#!/usr/bin/env python3
"""Headline figure: the composed compliance figure on the two-link German-credit
chain, with its 95 per cent interval computed two ways.

(a) carrying the measured cross-link covariance;
(b) with the cross-term declared zero (the naive independence assumption).

Every quantity that exists in the experiment output is read from
experiment/redraw-2026-07-24/results-redraw.json; nothing measured is hard-coded
here. The only chosen quantity is the decision threshold, which is a risk
appetite set by an operator and not an output of the experiment: it is derived by
a stated rule (midpoint of the two lower endpoints, rounded to three decimals)
purely so that the illustration is reproducible, and it is labelled illustrative
in the figure and in the caption.

Strict black and white: no colour, no grey fills, no rounded corners, no shadows.
The two intervals are distinguished by line style and cap style only.

Outputs: fig-chain-interval.pdf and fig-chain-interval.png next to this script.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
RESULTS = PAPER / "experiment" / "redraw-2026-07-24" / "results-redraw.json"

# 'redraw' = perturbations redrawn on every bootstrap replicate, i.e. the
# unconditional correlations. 'fixed_draw' would be conditional on one
# perturbation realisation.
RUN = "redraw"


def load():
    res = json.loads(RESULTS.read_text())
    chain = res["chain"][RUN]
    return res, chain


def main():
    res, chain = load()

    lo_m, hi_m = chain["ci95_measured_covariance"]
    lo_n, hi_n = chain["ci95_cross_term_declared_zero"]
    h = chain["h_point"]
    w_m = chain["width_measured"]
    w_n = chain["width_declared_zero"]
    under = chain["understatement_of_width_pct"]
    rho = chain["cross_link_correlation"]
    B = res["B"]
    n_test = res["n_test"]
    link_a = res["chain"]["links"]["A"]
    link_b = res["chain"]["links"]["B"]

    # ---- chosen, not measured -------------------------------------------------
    # A threshold that separates the two readings must lie strictly between the
    # lower endpoint of the wide (measured-covariance) interval and the lower
    # endpoint of the narrow (cross-term-zero) interval. Rule: midpoint of the
    # two, rounded to three decimals. Illustrative only.
    threshold = round(0.5 * (lo_m + lo_n), 3)
    assert lo_m < threshold < lo_n, (
        "chosen threshold does not separate the two readings: "
        f"{lo_m} < {threshold} < {lo_n} is false"
    )
    # ---------------------------------------------------------------------------

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.0,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    # Sized for a single-column text block, so the article can include
    # it at width=\textwidth with no scaling and no shrunken type.
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    FS = 8.0   # annotations
    FSE = 7.8  # endpoint values

    y_a, y_b = 0.72, 0.34
    cap_a, cap_b = 0.090, 0.048
    band_lo_y, band_hi_y = 0.10, 0.90
    K = dict(color="black", clip_on=False)

    # order matters: hatched band first, so the lines sit on top of it.
    # The band runs between the two lower endpoints: the thresholds that (b)
    # passes and (a) leaves undecided. It is NOT the whole range over which (a)
    # fails to decide -- that is the whole of interval (a) -- and it is not the
    # whole set of thresholds the two readings judge differently: a mirror band
    # of the same width, not drawn, sits between the two upper endpoints, where
    # (b) fails and (a) is again undecided.
    ax.add_patch(
        Rectangle(
            (lo_m, band_lo_y),
            lo_n - lo_m,
            band_hi_y - band_lo_y,
            facecolor="none",
            edgecolor="black",
            hatch="///",
            linewidth=0.0,
            zorder=0,
        )
    )
    # the two edges of the band, each a lower endpoint
    for x in (lo_m, lo_n):
        ax.plot([x, x], [band_lo_y, band_hi_y], lw=0.7, ls=(0, (1, 2)),
                zorder=1, **K)

    # operator threshold
    ax.plot([threshold, threshold], [band_lo_y, band_hi_y], lw=1.3, ls="-",
            zorder=4, **K)

    # (a) measured covariance carried: solid line, tall flat caps, filled point
    ax.plot([lo_m, hi_m], [y_a, y_a], lw=1.5, ls="-", zorder=5, **K)
    for x in (lo_m, hi_m):
        ax.plot([x, x], [y_a - cap_a, y_a + cap_a], lw=1.5, zorder=5, **K)
    ax.plot([h], [y_a], marker="o", ms=6.0, mfc="black", mec="black", zorder=6, **K)

    # (b) cross-term declared zero: dashed line, short caps, open point
    ax.plot(
        [lo_n, hi_n], [y_b, y_b], lw=1.5, ls=(0, (4.0, 2.2)), zorder=5, **K
    )
    for x in (lo_n, hi_n):
        ax.plot([x, x], [y_b - cap_b, y_b + cap_b], lw=1.5, zorder=5, **K)
    ax.plot([h], [y_b], marker="o", ms=6.0, mfc="white", mec="black", mew=1.3,
            zorder=6, **K)

    # endpoint values. Three sit clear of the hatched band and are set beside the
    # caps; the lower endpoint of (b) would land on the hatching, so it is set
    # above its own cap instead.
    pad = 0.0024
    ax.text(lo_m - pad, y_a, f"{lo_m:.3f}", ha="right", va="center", fontsize=FSE)
    ax.text(hi_m + pad, y_a, f"{hi_m:.3f}", ha="left", va="center", fontsize=FSE)
    ax.text(hi_n + pad, y_b, f"{hi_n:.3f}", ha="left", va="center", fontsize=FSE)
    ax.text(lo_n + 0.0012, y_b + cap_b + 0.012, f"{lo_n:.3f}", ha="left",
            va="bottom", fontsize=FSE)

    # widths above each interval, reading of the decision below it. Long labels
    # are centred on the axis to the right of the hatched band, with a margin,
    # so that no text lands on the hatching.
    x_right = 0.5 * (lo_n + hi_m + 0.28 * (hi_m - lo_m)) + 0.002
    ax.text(x_right, 0.835, f"point estimate {h:.3f}, width {w_m:.4f}",
            ha="center", va="bottom", fontsize=FS)
    ax.text(
        x_right, 0.60, "straddles the threshold: undecided",
        ha="center", va="top", fontsize=FS,
    )
    ax.text(
        x_right, 0.47,
        f"width {w_n:.4f}, understating it by {under:.1f} per cent",
        ha="center", va="bottom", fontsize=FS,
    )
    ax.text(
        x_right, 0.22, "clear of the threshold: reads as passing",
        ha="center", va="top", fontsize=FS,
    )

    # threshold label
    ax.text(
        threshold + 0.0030, 0.955,
        f"operator threshold {threshold:.3f} (illustrative)",
        ha="left", va="center", fontsize=FS,
    )

    # Band annotation. The band is the set of thresholds that (b) passes and (a)
    # leaves undecided, i.e. between the two lower endpoints -- not the whole
    # range over which (a) fails to decide, which is the whole of interval (a).
    band_mid = 0.5 * (lo_m + lo_n)
    ax.annotate(
        "any threshold here: (a) is undecided, (b) passes",
        xy=(band_mid, band_lo_y),
        xytext=(lo_n + 0.0035, 0.035),
        ha="left",
        va="center",
        fontsize=FS,
        arrowprops=dict(arrowstyle="-", lw=0.8, color="black",
                        shrinkA=2.0, shrinkB=0.0),
    )

    ax.set_yticks([y_a, y_b])
    ax.set_yticklabels(
        ["(a) measured\ncross-link\ncovariance", "(b) cross-term\ndeclared\nzero"],
        fontsize=8.4,
    )
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3.5, width=0.9, direction="out",
                   labelsize=8.2)

    span = hi_m - lo_m
    ax.set_xlim(lo_m - 0.28 * span, hi_m + 0.28 * span)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(
        f"composed compliance figure  h = C({link_a}) x C({link_b})",
        fontsize=8.8, labelpad=5,
    )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 4))

    note = (
        f"German credit (OpenML credit-g), n_test = {n_test}, B = {B} paired "
        "bootstrap replicates, perturbations\nredrawn per replicate; measured "
        f"cross-link correlation {rho:.3f}. The threshold is illustrative:\n"
        "a risk appetite is set by the operator, not by the data."
    )
    fig.text(0.005, -0.045, note, fontsize=7.0, ha="left", va="top",
             linespacing=1.35)

    out_pdf = HERE / "fig-chain-interval.pdf"
    out_png = HERE / "fig-chain-interval.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)

    print(f"run                     : {RUN}")
    print(f"point estimate h        : {h!r}")
    print(f"ci measured covariance  : [{lo_m!r}, {hi_m!r}]  width {w_m!r}")
    print(f"ci cross-term zero      : [{lo_n!r}, {hi_n!r}]  width {w_n!r}")
    print(f"understatement of width : {under!r} per cent")
    print(f"cross-link correlation  : {rho!r}")
    print(f"threshold (CHOSEN)      : {threshold!r}")
    print(f"wrote                   : {out_pdf}")
    print(f"wrote                   : {out_png}")


if __name__ == "__main__":
    main()
