#!/usr/bin/env python3
"""Finite-sample coverage study for the shared-sample interval layer.

The data-generating process has known component means.  For a latent Gaussian
row X and standard-normal CDF Phi,

    Y_1 = Phi(mu_1 + X_1)
    Y_2 = Phi(mu_2 + X_2)
    Y_3 = Phi(mu_3 + X_3 + epsilon), epsilon ~ N(0, tau^2).

Thus E[Y_k] is available analytically.  Components one and two depend only on
the shared test row; component three adds perturbation noise.  The primary
paired bootstrap resamples one row-index stream for all components and redraws
epsilon.  The negative control uses independent index streams per component.

Outputs are written next to this script:
  results-coverage.json
  results-coverage.csv
  COVERAGE-SUMMARY.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, ndtr, ndtri
from scipy.stats import chi2, f, norm


HERE = Path(__file__).resolve().parent
P = 3
ALPHA = 0.05
MASTER_SEED = 20260727


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    mu: tuple[float, float, float]
    latent_corr: tuple[tuple[float, float, float], ...]
    perturbation_tau: float


SCENARIOS = (
    Scenario("central-positive-n100", 100, (0.35, 0.15, -0.05),
             ((1.0, 0.60, 0.60), (0.60, 1.0, 0.60), (0.60, 0.60, 1.0)), 0.0),
    Scenario("central-positive-n300", 300, (0.35, 0.15, -0.05),
             ((1.0, 0.60, 0.60), (0.60, 1.0, 0.60), (0.60, 0.60, 1.0)), 0.0),
    Scenario("central-positive-n800", 800, (0.35, 0.15, -0.05),
             ((1.0, 0.60, 0.60), (0.60, 1.0, 0.60), (0.60, 0.60, 1.0)), 0.0),
    Scenario("upper-bound-positive-n300", 300, (2.20, 2.00, 1.80),
             ((1.0, 0.60, 0.60), (0.60, 1.0, 0.60), (0.60, 0.60, 1.0)), 0.0),
    Scenario("central-zero-n300", 300, (0.35, 0.15, -0.05),
             ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), 0.0),
    Scenario("central-negative-n300", 300, (0.35, 0.15, -0.05),
             ((1.0, -0.35, -0.35), (-0.35, 1.0, -0.35),
              (-0.35, -0.35, 1.0)), 0.0),
    Scenario("mixed-sign-n300", 300, (0.35, 0.15, -0.05),
             ((1.0, 0.55, -0.30), (0.55, 1.0, -0.20),
              (-0.30, -0.20, 1.0)), 0.0),
    Scenario("central-positive-noise-n300", 300, (0.35, 0.15, -0.05),
             ((1.0, 0.60, 0.60), (0.60, 1.0, 0.60), (0.60, 0.60, 1.0)), 0.8),
    Scenario("upper-bound-positive-noise-n300", 300, (2.20, 2.00, 1.80),
             ((1.0, 0.60, 0.60), (0.60, 1.0, 0.60), (0.60, 0.60, 1.0)), 0.8),
)


def geometric_mean(theta: np.ndarray) -> np.ndarray:
    return np.exp(np.log(np.clip(theta, 1e-12, None)).mean(axis=-1))


def true_theta(s: Scenario) -> np.ndarray:
    variances = np.array([1.0, 1.0, 1.0 + s.perturbation_tau**2])
    return ndtr(np.asarray(s.mu) / np.sqrt(1.0 + variances))


def safe_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(matrix, rcond=1e-12, hermitian=True)


def quadratic_rows(values: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", values, inverse, values)


def bca_interval(
    boot_values: np.ndarray,
    point_value: float,
    jackknife_values: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    less = (np.count_nonzero(boot_values < point_value) + 0.5) / (
        len(boot_values) + 1.0
    )
    z0 = ndtri(np.clip(less, 1e-8, 1 - 1e-8))
    jack_mean = float(jackknife_values.mean())
    diffs = jack_mean - jackknife_values
    denom = 6.0 * float(np.sum(diffs**2)) ** 1.5
    acceleration = float(np.sum(diffs**3)) / denom if denom > 0 else 0.0

    adjusted = []
    for probability in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = ndtri(probability)
        denominator = 1.0 - acceleration * (z0 + z)
        adjusted.append(ndtr(z0 + (z0 + z) / denominator))
    lo_q, hi_q = np.clip(adjusted, 0.0, 1.0)
    if lo_q > hi_q:
        lo_q, hi_q = hi_q, lo_q
    return tuple(float(x) for x in np.quantile(boot_values, [lo_q, hi_q]))


def interval_contains(interval: tuple[float, float], truth: float) -> int:
    return int(interval[0] <= truth <= interval[1])


def initialise_totals() -> dict[str, float]:
    names = (
        "joint_chi2",
        "joint_hotelling",
        "joint_bootstrap_radial",
        "joint_logit_chi2",
        "joint_logit_hotelling",
        "joint_logit_bootstrap_radial",
        "joint_independent_chi2",
        "composite_delta",
        "composite_logit_delta",
        "composite_percentile",
        "composite_bca",
        "composite_independent_percentile",
        "width_delta",
        "width_logit_delta",
        "width_percentile",
        "width_bca",
        "width_independent_percentile",
        "cross_variance_share",
    )
    return {name: 0.0 for name in names}


def simulate_scenario(
    scenario: Scenario,
    outer: int,
    bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    corr = np.asarray(scenario.latent_corr, dtype=float)
    eigenvalues = np.linalg.eigvalsh(corr)
    if eigenvalues.min() <= 0:
        raise ValueError(f"{scenario.name}: correlation matrix is not positive definite")

    theta_truth = true_theta(scenario)
    composite_truth = float(geometric_mean(theta_truth))
    totals = initialise_totals()
    z = norm.ppf(1.0 - ALPHA / 2.0)
    chi_cutoff = chi2.ppf(1.0 - ALPHA, P)
    hotelling_cutoff = (
        P * (scenario.n - 1) / (scenario.n - P)
        * f.ppf(1.0 - ALPHA, P, scenario.n - P)
    )

    for _ in range(outer):
        latent = rng.multivariate_normal(np.zeros(P), corr, size=scenario.n)
        epsilon = rng.normal(0.0, scenario.perturbation_tau, size=scenario.n)
        observed = np.column_stack(
            (
                ndtr(scenario.mu[0] + latent[:, 0]),
                ndtr(scenario.mu[1] + latent[:, 1]),
                ndtr(scenario.mu[2] + latent[:, 2] + epsilon),
            )
        )
        theta_hat = observed.mean(axis=0)
        composite_hat = float(geometric_mean(theta_hat))

        paired_idx = rng.integers(0, scenario.n, size=(bootstrap, scenario.n))
        paired = observed[paired_idx].mean(axis=1)
        if scenario.perturbation_tau > 0:
            redrawn_epsilon = rng.normal(
                0.0, scenario.perturbation_tau, size=(bootstrap, scenario.n)
            )
            paired[:, 2] = ndtr(
                scenario.mu[2]
                + latent[paired_idx, 2]
                + redrawn_epsilon
            ).mean(axis=1)

        independent = np.empty((bootstrap, P), dtype=float)
        for component in range(P):
            indices = rng.integers(0, scenario.n, size=(bootstrap, scenario.n))
            if component == 2 and scenario.perturbation_tau > 0:
                redrawn_epsilon = rng.normal(
                    0.0, scenario.perturbation_tau, size=(bootstrap, scenario.n)
                )
                independent[:, component] = ndtr(
                    scenario.mu[2] + latent[indices, 2] + redrawn_epsilon
                ).mean(axis=1)
            else:
                independent[:, component] = observed[indices, component].mean(axis=1)

        covariance = np.cov(paired, rowvar=False, ddof=1)
        independent_covariance = np.cov(independent, rowvar=False, ddof=1)
        inverse = safe_inverse(covariance)
        independent_inverse = safe_inverse(independent_covariance)
        error = theta_hat - theta_truth
        q_observed = float(error @ inverse @ error)
        q_independent = float(error @ independent_inverse @ error)
        centred_boot = paired - theta_hat
        radial_cutoff = float(
            np.quantile(quadratic_rows(centred_boot, inverse), 1.0 - ALPHA)
        )

        eta_hat = logit(np.clip(theta_hat, 1e-10, 1 - 1e-10))
        eta_truth = logit(np.clip(theta_truth, 1e-10, 1 - 1e-10))
        jacobian = np.diag(1.0 / (theta_hat * (1.0 - theta_hat)))
        eta_covariance = jacobian @ covariance @ jacobian
        eta_inverse = safe_inverse(eta_covariance)
        eta_error = eta_hat - eta_truth
        q_logit_observed = float(eta_error @ eta_inverse @ eta_error)
        eta_boot = logit(np.clip(paired, 1e-10, 1 - 1e-10))
        eta_radial_cutoff = float(
            np.quantile(
                quadratic_rows(eta_boot - eta_hat, eta_inverse),
                1.0 - ALPHA,
            )
        )

        totals["joint_chi2"] += q_observed <= chi_cutoff
        totals["joint_hotelling"] += q_observed <= hotelling_cutoff
        totals["joint_bootstrap_radial"] += q_observed <= radial_cutoff
        totals["joint_logit_chi2"] += q_logit_observed <= chi_cutoff
        totals["joint_logit_hotelling"] += q_logit_observed <= hotelling_cutoff
        totals["joint_logit_bootstrap_radial"] += (
            q_logit_observed <= eta_radial_cutoff
        )
        totals["joint_independent_chi2"] += q_independent <= chi_cutoff

        paired_composite = geometric_mean(paired)
        independent_composite = geometric_mean(independent)
        gradient = composite_hat / (P * theta_hat)
        variance = float(gradient @ covariance @ gradient)
        diagonal_variance = float(
            np.sum((gradient**2) * np.diag(covariance))
        )
        totals["cross_variance_share"] += (
            (variance - diagonal_variance) / variance if variance > 0 else 0.0
        )

        delta_interval = (
            max(0.0, composite_hat - z * math.sqrt(max(variance, 0.0))),
            min(1.0, composite_hat + z * math.sqrt(max(variance, 0.0))),
        )
        logit_se = (
            math.sqrt(max(variance, 0.0))
            / max(composite_hat * (1.0 - composite_hat), 1e-12)
        )
        logit_interval = (
            float(expit(logit(composite_hat) - z * logit_se)),
            float(expit(logit(composite_hat) + z * logit_se)),
        )
        percentile_interval = tuple(
            float(x)
            for x in np.quantile(paired_composite, [ALPHA / 2, 1 - ALPHA / 2])
        )
        independent_interval = tuple(
            float(x)
            for x in np.quantile(
                independent_composite, [ALPHA / 2, 1 - ALPHA / 2]
            )
        )

        jackknife_theta = (
            observed.sum(axis=0, keepdims=True) - observed
        ) / (scenario.n - 1)
        jackknife_composite = geometric_mean(jackknife_theta)
        bca = bca_interval(
            paired_composite, composite_hat, jackknife_composite, ALPHA
        )

        intervals = {
            "delta": delta_interval,
            "logit_delta": logit_interval,
            "percentile": percentile_interval,
            "bca": bca,
            "independent_percentile": independent_interval,
        }
        for name, interval in intervals.items():
            totals[f"composite_{name}"] += interval_contains(
                interval, composite_truth
            )
            totals[f"width_{name}"] += interval[1] - interval[0]

    coverages = {
        key: value / outer
        for key, value in totals.items()
        if key.startswith("joint_") or key.startswith("composite_")
    }
    mean_widths = {
        key.removeprefix("width_"): value / outer
        for key, value in totals.items()
        if key.startswith("width_")
    }
    paired_width = mean_widths["percentile"]
    independent_width = mean_widths["independent_percentile"]
    return {
        "scenario": asdict(scenario),
        "minimum_correlation_eigenvalue": float(eigenvalues.min()),
        "true_theta": theta_truth.tolist(),
        "true_composite": composite_truth,
        "outer_replications": outer,
        "bootstrap_replications": bootstrap,
        "coverage_monte_carlo_se_at_95pct": math.sqrt(0.95 * 0.05 / outer),
        "coverage": coverages,
        "mean_interval_width": mean_widths,
        "mean_cross_variance_share": totals["cross_variance_share"] / outer,
        "independent_width_change_pct": (
            100.0 * (independent_width / paired_width - 1.0)
        ),
    }


def write_csv(results: list[dict]) -> None:
    path = HERE / "results-coverage.csv"
    columns = (
        "scenario",
        "n",
        "tau",
        "true_composite",
        "joint_chi2",
        "joint_hotelling",
        "joint_bootstrap_radial",
        "joint_logit_chi2",
        "joint_logit_hotelling",
        "joint_logit_bootstrap_radial",
        "joint_independent_chi2",
        "composite_delta",
        "composite_logit_delta",
        "composite_percentile",
        "composite_bca",
        "composite_independent_percentile",
        "width_percentile",
        "width_independent_percentile",
        "independent_width_change_pct",
        "mean_cross_variance_share",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            coverage = result["coverage"]
            widths = result["mean_interval_width"]
            writer.writerow(
                {
                    "scenario": result["scenario"]["name"],
                    "n": result["scenario"]["n"],
                    "tau": result["scenario"]["perturbation_tau"],
                    "true_composite": result["true_composite"],
                    "joint_chi2": coverage["joint_chi2"],
                    "joint_hotelling": coverage["joint_hotelling"],
                    "joint_bootstrap_radial": coverage[
                        "joint_bootstrap_radial"
                    ],
                    "joint_logit_chi2": coverage["joint_logit_chi2"],
                    "joint_logit_hotelling": coverage[
                        "joint_logit_hotelling"
                    ],
                    "joint_logit_bootstrap_radial": coverage[
                        "joint_logit_bootstrap_radial"
                    ],
                    "joint_independent_chi2": coverage[
                        "joint_independent_chi2"
                    ],
                    "composite_delta": coverage["composite_delta"],
                    "composite_logit_delta": coverage["composite_logit_delta"],
                    "composite_percentile": coverage["composite_percentile"],
                    "composite_bca": coverage["composite_bca"],
                    "composite_independent_percentile": coverage[
                        "composite_independent_percentile"
                    ],
                    "width_percentile": widths["percentile"],
                    "width_independent_percentile": widths[
                        "independent_percentile"
                    ],
                    "independent_width_change_pct": result[
                        "independent_width_change_pct"
                    ],
                    "mean_cross_variance_share": result[
                        "mean_cross_variance_share"
                    ],
                }
            )


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def write_summary(results: list[dict], elapsed: float, args: argparse.Namespace) -> None:
    lines = [
        "# Finite-sample coverage study",
        "",
        "Run date: 2026-07-27.",
        f"Outer replications per scenario: {args.outer}. "
        f"Bootstrap replications per dataset: {args.bootstrap}.",
        f"Master seed: {MASTER_SEED}. Runtime: {elapsed:.1f} seconds.",
        "",
        "The data-generating process is a bounded Gaussian-copula model with "
        "analytically known component means. Components share each sampled row; "
        "the third can add independently redrawn perturbation noise. The paired "
        "bootstrap uses one index stream for all components. The independent "
        "negative control uses a separate stream per component.",
        "",
        "## Joint-region coverage (%)",
        "",
        "| Scenario | n | raw chi-squared | raw Hotelling | logit chi-squared | logit Hotelling | raw bootstrap radial | logit bootstrap radial | independent control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        c = result["coverage"]
        s = result["scenario"]
        lines.append(
            f"| {s['name']} | {s['n']} | {pct(c['joint_chi2'])} | "
            f"{pct(c['joint_hotelling'])} | {pct(c['joint_logit_chi2'])} | "
            f"{pct(c['joint_logit_hotelling'])} | "
            f"{pct(c['joint_bootstrap_radial'])} | "
            f"{pct(c['joint_logit_bootstrap_radial'])} | "
            f"{pct(c['joint_independent_chi2'])} |"
        )

    lines.extend(
        [
            "",
            "## Composite-interval coverage (%)",
            "",
            "| Scenario | delta | logit delta | percentile | BCa | independent percentile | paired width | independent width |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        c = result["coverage"]
        w = result["mean_interval_width"]
        lines.append(
            f"| {result['scenario']['name']} | {pct(c['composite_delta'])} | "
            f"{pct(c['composite_logit_delta'])} | "
            f"{pct(c['composite_percentile'])} | {pct(c['composite_bca'])} | "
            f"{pct(c['composite_independent_percentile'])} | "
            f"{w['percentile']:.4f} | {w['independent_percentile']:.4f} |"
        )

    paired_coverages = np.array(
        [r["coverage"]["composite_percentile"] for r in results]
    )
    independent_coverages = np.array(
        [r["coverage"]["composite_independent_percentile"] for r in results]
    )
    joint_chi = np.array([r["coverage"]["joint_chi2"] for r in results])
    joint_hot = np.array([r["coverage"]["joint_hotelling"] for r in results])
    joint_logit_chi = np.array(
        [r["coverage"]["joint_logit_chi2"] for r in results]
    )
    joint_logit_hot = np.array(
        [r["coverage"]["joint_logit_hotelling"] for r in results]
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"Across the nine scenarios, paired-percentile composite coverage "
            f"ranges from {100 * paired_coverages.min():.1f}% to "
            f"{100 * paired_coverages.max():.1f}%. The independent-stream "
            f"control ranges from {100 * independent_coverages.min():.1f}% to "
            f"{100 * independent_coverages.max():.1f}%. Joint chi-squared "
            f"coverage ranges from {100 * joint_chi.min():.1f}% to "
            f"{100 * joint_chi.max():.1f}%, and Hotelling coverage from "
            f"{100 * joint_hot.min():.1f}% to {100 * joint_hot.max():.1f}%. "
            f"On the logit scale, the corresponding ranges are "
            f"{100 * joint_logit_chi.min():.1f}% to "
            f"{100 * joint_logit_chi.max():.1f}% and "
            f"{100 * joint_logit_hot.min():.1f}% to "
            f"{100 * joint_logit_hot.max():.1f}%.",
            "",
            "The Monte Carlo standard error of a 95% coverage estimate is "
            f"{100 * results[0]['coverage_monte_carlo_se_at_95pct']:.2f} "
            "percentage points. Results should therefore be read in bands, "
            "not as exact rankings separated by one percentage point.",
            "",
            "Raw results: `results-coverage.json` and `results-coverage.csv`. "
            "Driver: `coverage_simulation.py`.",
            "",
        ]
    )
    (HERE / "COVERAGE-SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=400)
    parser.add_argument("--bootstrap", type=int, default=399)
    args = parser.parse_args()
    if args.outer < 50 or args.bootstrap < 99:
        raise SystemExit("Use at least 50 outer and 99 bootstrap replications.")

    start = time.time()
    results = []
    for index, scenario in enumerate(SCENARIOS):
        scenario_seed = MASTER_SEED + 10_000 * (index + 1)
        print(
            f"[{index + 1}/{len(SCENARIOS)}] {scenario.name}: "
            f"n={scenario.n}, outer={args.outer}, B={args.bootstrap}",
            flush=True,
        )
        result = simulate_scenario(
            scenario, args.outer, args.bootstrap, scenario_seed
        )
        results.append(result)
        print(
            "  paired percentile coverage "
            f"{100 * result['coverage']['composite_percentile']:.1f}%; "
            "independent "
            f"{100 * result['coverage']['composite_independent_percentile']:.1f}%",
            flush=True,
        )

    elapsed = time.time() - start
    payload = {
        "design": "bounded Gaussian-copula means with analytic truth",
        "master_seed": MASTER_SEED,
        "alpha": ALPHA,
        "outer_replications_per_scenario": args.outer,
        "bootstrap_replications": args.bootstrap,
        "runtime_seconds": elapsed,
        "numpy_version": np.__version__,
        "scenarios": results,
    }
    (HERE / "results-coverage.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(results)
    write_summary(results, elapsed, args)
    print(f"Completed in {elapsed:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
