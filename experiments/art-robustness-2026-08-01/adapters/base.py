"""Attack-adapter API for the frozen-artifact robustness arm.

A fitted sklearn model goes in; a length-L robustness curve comes out with
full severity/validity/achieved-distortion metadata and the per-severity
predicted-probability tensor in the SAME row order the existing SAFE pipeline
uses. This is the layer that sits between an ART evasion attack and safeai's
rgr_score, so every scored row is legality-projected first (ART shortlist
requirement).

The output keys are designed to drop into the `curves_point` dict produced by
pavia-composite-2026-07-25/pavia_composite_experiment.py (see OUTPUT-SCHEMA.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Protocol

import numpy as np

try:
    from ..constraints.constraints import FeatureConstraints
except ImportError:  # loaded as top-level packages from experiment dir
    from constraints.constraints import FeatureConstraints


@dataclass
class AttackResult:
    family: str                       # "hopskipjump" | "fgm" | ...
    model_tag: str                    # "logit" | "rf"
    class_order: list                 # label order behind p columns
    requested_severity: list[float]   # the predeclared grid knots
    achieved_l2_median: list[float]   # per knot: median projected L2 distance
    achieved_l2_p10: list[float]
    achieved_l2_p90: list[float]
    validity_rate: list[float]        # per knot: fraction of rows still legal-and-attacked
    flip_rate: list[float]            # per knot: fraction whose argmax class flipped
    p_perturbed: np.ndarray           # (L, n_rows, n_classes), row order == p_full
    rng_seed: int
    notes: list[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        """Everything except the heavy p_perturbed tensor, for JSON summaries."""
        d = asdict(self)
        d.pop("p_perturbed")
        return d


class AttackAdapter(Protocol):
    """One ART attack family wrapped so it emits a legality-projected robustness
    curve. Implementations must generate the attack ONCE per (model, severity)
    knot and never per bootstrap replicate (see the fixed-draw lock in the
    runtime-budget note)."""

    family: str

    def generate_curve(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        severity_grid: list[float],
        constraints: FeatureConstraints,
        *,
        model_tag: str,
        class_order: list,
        rng_seed: int,
    ) -> AttackResult:
        ...


def summarize_probabilities(
    model, X: np.ndarray, class_order: list
) -> np.ndarray:
    """predict_proba aligned to class_order; falls back to decision_function
    only if predict_proba is absent (not expected for the paper's models)."""
    proba = model.predict_proba(X)
    model_classes = list(model.classes_)
    index = [model_classes.index(c) for c in class_order]
    return np.asarray(proba, dtype=float)[:, index]


def curve_metadata(
    X_orig: np.ndarray,
    X_proj: np.ndarray,
    changed_mask: np.ndarray,
    distances: np.ndarray,
    p_clean: np.ndarray,
    p_adv: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Per-knot scalar metadata: (l2_median, l2_p10, l2_p90, validity_rate,
    flip_rate). validity_rate is the fraction of rows that were actually moved
    (a projected row identical to the original is a failed/no-op attack);
    flip_rate is the fraction whose predicted argmax class changed."""
    moved = np.any(changed_mask, axis=1)
    validity_rate = float(np.mean(moved))
    flips = np.argmax(p_clean, axis=1) != np.argmax(p_adv, axis=1)
    flip_rate = float(np.mean(flips))
    d = distances[moved] if moved.any() else distances
    return (
        float(np.median(d)),
        float(np.percentile(d, 10)),
        float(np.percentile(d, 90)),
        validity_rate,
        flip_rate,
    )
