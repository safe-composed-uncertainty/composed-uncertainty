"""Per-feature constraint projection for tabular adversarial robustness.

ART evasion attacks (HopSkipJump, FGM) move a row anywhere in R^d. On tabular
credit data that produces illegal rows: negative loan durations, fractional
category codes, mutated immutable attributes (age, sex). The robustness metric
must score attacks a realistic adversary could actually mount, so every
adversarial row is projected back onto the legal manifold before it reaches
the scorer. This module is the layer the ART shortlist requires between the
raw attack and safeai's rgr_score.

A FeatureConstraints object is built from a per-dataset JSON spec (see
constraints/feature_constraints_*.json). It clips continuous features to their
training-supported range, snaps ordinal/categorical features to the nearest
legal value, restores immutable fields verbatim, and reports the achieved
mixed-type distance and per-row validity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureSpec:
    name: str
    kind: str  # "continuous" | "ordinal" | "categorical" | "immutable"
    lo: float | None = None
    hi: float | None = None
    legal_values: list[float] | None = None


class FeatureConstraints:
    """Projects adversarial rows back onto the training-supported legal manifold.

    The spec order must match the model's feature column order. Continuous
    bounds default to the observed training min/max when not given explicitly;
    call `fit_bounds(X_train)` to fill them from data.
    """

    def __init__(self, specs: list[FeatureSpec]):
        self.specs = specs

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureConstraints":
        raw = json.loads(Path(path).read_text())
        specs = []
        for col in raw["features"]:
            specs.append(
                FeatureSpec(
                    name=col["name"],
                    kind=col["kind"],
                    lo=col.get("lo"),
                    hi=col.get("hi"),
                    legal_values=col.get("legal_values"),
                )
            )
        return cls(specs)

    def fit_bounds(self, X_train: np.ndarray) -> "FeatureConstraints":
        """Fill any missing continuous bounds and categorical legal sets from
        the training data, so the projection is training-supported by
        construction."""
        for j, spec in enumerate(self.specs):
            col = np.asarray(X_train[:, j], dtype=float)
            if spec.kind == "continuous":
                if spec.lo is None:
                    spec.lo = float(np.min(col))
                if spec.hi is None:
                    spec.hi = float(np.max(col))
            elif spec.kind in ("ordinal", "categorical"):
                if spec.legal_values is None:
                    spec.legal_values = sorted(float(v) for v in np.unique(col))
        return self

    # ---- projection -------------------------------------------------------
    def _project_value(self, spec: FeatureSpec, orig: float, adv: float) -> float:
        if spec.kind == "immutable":
            return float(orig)
        if spec.kind == "continuous":
            lo = spec.lo if spec.lo is not None else -np.inf
            hi = spec.hi if spec.hi is not None else np.inf
            return float(min(max(adv, lo), hi))
        if spec.kind in ("ordinal", "categorical"):
            legal = np.asarray(spec.legal_values, dtype=float)
            return float(legal[int(np.argmin(np.abs(legal - adv)))])
        raise ValueError(f"unknown kind {spec.kind!r} for {spec.name!r}")

    def project(
        self, x_orig_row: np.ndarray, x_adv_row: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (x_projected, changed_mask, l2_distance_to_original).

        changed_mask marks columns whose projected value differs from the
        original; the distance is the mixed-type L2 between projected and
        original (continuous in native units), i.e. the distortion a realistic
        adversary actually achieved after legality projection.
        """
        x_orig = np.asarray(x_orig_row, dtype=float)
        x_adv = np.asarray(x_adv_row, dtype=float)
        out = np.empty_like(x_orig)
        for j, spec in enumerate(self.specs):
            out[j] = self._project_value(spec, x_orig[j], x_adv[j])
        changed = ~np.isclose(out, x_orig)
        distance = float(np.linalg.norm(out - x_orig))
        return out, changed, distance

    def project_batch(
        self, X_orig: np.ndarray, X_adv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised project() over a batch; returns
        (X_projected, changed_mask[n,d], distances[n])."""
        X_orig = np.asarray(X_orig, dtype=float)
        X_adv = np.asarray(X_adv, dtype=float)
        n, d = X_orig.shape
        out = np.empty_like(X_orig)
        changed = np.zeros((n, d), dtype=bool)
        dist = np.empty(n, dtype=float)
        for i in range(n):
            out[i], changed[i], dist[i] = self.project(X_orig[i], X_adv[i])
        return out, changed, dist

    def immutable_columns(self) -> list[int]:
        return [j for j, s in enumerate(self.specs) if s.kind == "immutable"]
