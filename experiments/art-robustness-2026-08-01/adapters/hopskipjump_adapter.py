"""HopSkipJump decision-based black-box evasion adapter (primary ART arm).

Works for both the logistic-regression pipeline and the random forest, because
HopSkipJump needs only class predictions. The attack is generated ONCE at the
largest budget knot and evaluated along the predeclared severity grid by
interpolating each adversarial row toward its original and re-projecting to the
legal manifold. This keeps the attack out of the bootstrap loop (fixed-draw
lock): one attack generation per (model), curve read off by interpolation.
"""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier

try:
    from .base import AttackResult, summarize_probabilities, curve_metadata
    from ..constraints.constraints import FeatureConstraints
except ImportError:  # loaded as top-level packages from experiment dir
    from adapters.base import AttackResult, summarize_probabilities, curve_metadata
    from constraints.constraints import FeatureConstraints


class HopSkipJumpAdapter:
    family = "hopskipjump"

    def __init__(self, max_iter_top: int = 40, max_eval: int = 1000,
                 init_eval: int = 100, batch_size: int = 64):
        self.max_iter_top = max_iter_top
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.batch_size = batch_size

    def generate_curve(
        self, model, X, y, severity_grid, constraints, *,
        model_tag, class_order, rng_seed,
    ) -> AttackResult:
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(rng_seed)

        clf = SklearnClassifier(model=model)
        # One decision-based attack at the top budget; ART's estimator wraps
        # predict(), so this needs no gradients (works for the RF too).
        attack = HopSkipJump(
            classifier=clf, targeted=False, norm=2,
            max_iter=self.max_iter_top, max_eval=self.max_eval,
            init_eval=self.init_eval, batch_size=self.batch_size, verbose=False,
        )
        X_adv_top = attack.generate(x=X)

        p_clean = summarize_probabilities(model, X, class_order)

        # severity_grid entries in [0,1] are the fraction of the top attack
        # displacement applied, then legality-projected. 0 -> original row.
        L = len(severity_grid)
        n, n_classes = p_clean.shape
        p_perturbed = np.empty((L, n, n_classes), dtype=float)
        med, p10, p90, valid, flip = [], [], [], [], []
        notes = []
        for k, s in enumerate(severity_grid):
            X_interp = X + float(s) * (X_adv_top - X)
            X_proj, changed, dist = constraints.project_batch(X, X_interp)
            p_adv = summarize_probabilities(model, X_proj, class_order)
            p_perturbed[k] = p_adv
            m, lo, hi, v, fl = curve_metadata(
                X, X_proj, changed, dist, p_clean, p_adv)
            med.append(m); p10.append(lo); p90.append(hi)
            valid.append(v); flip.append(fl)
        notes.append(
            f"HSJ generated once at max_iter={self.max_iter_top}; "
            f"curve read by interpolation+projection over {L} knots.")
        return AttackResult(
            family=self.family, model_tag=model_tag, class_order=list(class_order),
            requested_severity=list(map(float, severity_grid)),
            achieved_l2_median=med, achieved_l2_p10=p10, achieved_l2_p90=p90,
            validity_rate=valid, flip_rate=flip, p_perturbed=p_perturbed,
            rng_seed=rng_seed, notes=notes,
        )
