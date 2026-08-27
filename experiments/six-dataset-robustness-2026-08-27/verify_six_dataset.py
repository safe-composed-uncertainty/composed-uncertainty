#!/usr/bin/env python3
"""Independent verifier for the six-dataset robustness extension.

This program deliberately does not import the primary experiment runner or the
safeai package.  It reads the persisted bootstrap curve matrices and recomputes
the claim-bearing quantities from NumPy formulas:

* the RGE--RGR curve-mean correlations for tail-swap and Gaussian robustness;
* their Fisher-z tail-minus-Gaussian contrast;
* paired-bootstrap percentile widths for the arithmetic, geometric, RMS and
  TOPSIS composites under the primary tail-swap arm; and
* measured-covariance and declared-zero-cross-covariance delta widths.

The assumed JSON/NPZ contract is recorded in VERIFIER-SCHEMA.md beside this
file.  Directory-wide production verification is strict: exactly six datasets,
B=2000, and conditioning checks of 100 logistic and 30 random-forest replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE
DEFAULT_REPORT = HERE / "verify-six-dataset.json"
MODEL_TAGS = ("logit", "rf")
AGGREGATORS = ("arithmetic", "geometric", "rms", "topsis")
PRODUCTION_B = 2000
PRODUCTION_B_FULL = {"logit": 100, "rf": 30}
DEFAULT_ALPHA = 0.05
FROZEN_PROTOCOL_SHA256 = "791769816ef3dbe99a7f67b4abb07deafefdbe45f5aa9415e05708f017bd488d"
FROZEN_MANIFEST_SHA256 = "b86e0e173e66f0b67f52046065b991e26097c4f98b1eabc7155ec81869042b7b"
FROZEN_REQUIREMENTS_SHA256 = "0f82ff87831ea010b09b9bdce3467ebdbc608d9166c69bcfa753e517061c489f"
FROZEN_SOURCE_AUDIT_SHA256 = "42c5ebafd911315a002d1bd259c8b7ee61772b66e32fb4bc482f07bd1d796199"
FROZEN_SCHEMA_AUDIT_SHA256 = "7dfb10655a7c9999d1535c40e5f8349067a858ebe93bc5d46a1e621be9aeac65"
FROZEN_SAFEAI_COMMIT = "39768fcd5264c881f7174268bbffda52b298ae89"
FROZEN_PYTHON = "3.12.3"
FROZEN_CONFIG = {
    "seed": 20260723,
    "bootstrap_seed": 20260724,
    "test_size": 0.30,
    "row_cap": 30_000,
    "z": 1.959963984540054,
}
FROZEN_LIBRARIES = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit_learn": "1.9.0",
    "pandas": "3.0.5",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
}
ATOL = 5e-10
RTOL = 5e-10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class VerificationError(RuntimeError):
    """A schema, invariant or numerical comparison failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    require(key in mapping, f"missing {label}.{key}")
    return mapping[key]


def pick_key(mapping: dict[str, Any], keys: tuple[str, ...], label: str) -> Any:
    """Return the first present spelling while keeping the accepted aliases explicit."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise VerificationError(f"missing {label}; accepted keys are {keys}")


def require_finite_scalar(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} must be numeric") from exc
    require(math.isfinite(result), f"{label} must be finite")
    return result


def require_int(value: Any, label: str) -> int:
    require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} must be an integer") from exc
    require(result == value, f"{label} must be an integer")
    return result


def require_array(value: Any, label: str, ndim: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} must be a numeric array") from exc
    if ndim is not None:
        require(result.ndim == ndim, f"{label} must have {ndim} dimensions")
    require(np.all(np.isfinite(result)), f"{label} contains non-finite values")
    return result


def compare(label: str, actual: Any, recorded: Any,
            atol: float = ATOL, rtol: float = RTOL) -> None:
    a = require_array(actual, f"computed {label}")
    r = require_array(recorded, f"recorded {label}")
    require(a.shape == r.shape,
            f"{label} shape mismatch: computed {a.shape}, recorded {r.shape}")
    if not np.allclose(a, r, atol=atol, rtol=rtol, equal_nan=False):
        gap = np.abs(a - r)
        flat = int(np.argmax(gap))
        idx = np.unravel_index(flat, gap.shape)
        raise VerificationError(
            f"{label} mismatch at {idx}: computed {a[idx]:.17g}, "
            f"recorded {r[idx]:.17g}, abs gap {gap[idx]:.3g}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def manual_corr(x: np.ndarray, y: np.ndarray, label: str) -> float:
    x = require_array(x, f"{label}.x", ndim=1)
    y = require_array(y, f"{label}.y", ndim=1)
    require(x.shape == y.shape, f"{label} vectors have different lengths")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    require(denom > 0.0, f"{label} is undefined for a constant vector")
    value = float((xc @ yc) / denom)
    require(-1.0 - 1e-12 <= value <= 1.0 + 1e-12,
            f"{label} escaped [-1, 1] beyond floating-point tolerance")
    return float(np.clip(value, -1.0, 1.0))


def percentile_block(values: np.ndarray, alpha: float) -> dict[str, Any]:
    values = require_array(values, "composite replicates", ndim=1)
    lo, hi = np.percentile(values, [100.0 * alpha / 2.0,
                                    100.0 * (1.0 - alpha / 2.0)])
    return {
        "ci95_percentile": [float(lo), float(hi)],
        "width": float(hi - lo),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
    }


def volume_arithmetic(A: np.ndarray, E: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (A.mean(axis=1) + E.mean(axis=1) + R.mean(axis=1)) / 3.0


def volume_geometric(A: np.ndarray, E: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (np.cbrt(A).mean(axis=1)
            * np.cbrt(E).mean(axis=1)
            * np.cbrt(R).mean(axis=1))


def volume_rms(A: np.ndarray, E: np.ndarray, R: np.ndarray) -> np.ndarray:
    require(A.shape == E.shape == R.shape,
            "RMS inputs must have identical shapes")
    B, L = A.shape
    # Bound each temporary tensor to roughly two million float cells.
    chunk = max(1, min(B, 2_000_000 // max(1, L ** 3)))
    out = np.empty(B, dtype=float)
    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        a = A[start:stop, :, None, None]
        e = E[start:stop, None, :, None]
        r = R[start:stop, None, None, :]
        out[start:stop] = np.sqrt((a * a + e * e + r * r) / 3.0).mean(
            axis=(1, 2, 3)
        )
    return out


def topsis_reference(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed positive/negative ideals and per-axis weights."""
    require(L > 0, "TOPSIS curve length must be positive")
    positive = np.concatenate([
        np.ones(L),
        np.r_[1.0, np.zeros(L - 1)],
        np.ones(L),
    ])
    negative = np.concatenate([
        np.zeros(L),
        np.linspace(1.0, 0.5, L),
        np.r_[1.0, np.zeros(L - 1)],
    ])
    weights = np.full(3 * L, 1.0 / 3.0)
    return positive, negative, weights


def volume_topsis(A: np.ndarray, E: np.ndarray, R: np.ndarray) -> np.ndarray:
    require(A.shape == E.shape == R.shape,
            "TOPSIS inputs must have identical shapes")
    _B, L = A.shape
    positive, negative, weights = topsis_reference(L)
    values = np.concatenate([A, E, R], axis=1)
    distance_positive = np.sqrt(
        np.sum(((values - positive) * weights) ** 2, axis=1)
    )
    distance_negative = np.sqrt(
        np.sum(((values - negative) * weights) ** 2, axis=1)
    )
    denominator = distance_negative + distance_positive
    require(np.all(denominator > 0.0),
            "TOPSIS closeness is undefined when both distances are zero")
    return distance_negative / denominator


VOLUME_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "arithmetic": volume_arithmetic,
    "geometric": volume_geometric,
    "rms": volume_rms,
    "topsis": volume_topsis,
}


def rms_entry_gradients(a: np.ndarray, e: np.ndarray,
                        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = len(a)
    M = np.sqrt((a[:, None, None] ** 2
                 + e[None, :, None] ** 2
                 + r[None, None, :] ** 2) / 3.0)
    require(np.all(M > 0.0),
            "RMS point tensor contains zero; its ordinary gradient is undefined")
    inv = 1.0 / M
    scale = 1.0 / (3.0 * L ** 3)
    ga = a * inv.sum(axis=(1, 2)) * scale
    ge = e * inv.sum(axis=(0, 2)) * scale
    gr = r * inv.sum(axis=(0, 1)) * scale
    return ga, ge, gr


def topsis_entry_gradients(a: np.ndarray, e: np.ndarray,
                           r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of the fixed-reference TOPSIS closeness coefficient."""
    L = len(a)
    require(len(e) == L and len(r) == L,
            "TOPSIS point curves must have identical lengths")
    positive, negative, weights = topsis_reference(L)
    values = np.concatenate([a, e, r])
    weights_squared = weights ** 2
    distance_positive = math.sqrt(float(
        np.sum((values - positive) ** 2 * weights_squared)
    ))
    distance_negative = math.sqrt(float(
        np.sum((values - negative) ** 2 * weights_squared)
    ))
    require(distance_positive > 0.0 and distance_negative > 0.0,
            "TOPSIS gradient is undefined at an ideal reference point")
    denominator = (distance_negative + distance_positive) ** 2
    gradient = (
        distance_positive
        * (weights_squared * (values - negative) / distance_negative)
        - distance_negative
        * (weights_squared * (values - positive) / distance_positive)
    ) / denominator
    return gradient[:L], gradient[L:2 * L], gradient[2 * L:]


def delta_inputs(kind: str, A: np.ndarray, E: np.ndarray, R: np.ndarray,
                 a0: np.ndarray, e0: np.ndarray,
                 r0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "arithmetic":
        summaries = np.column_stack([A.mean(axis=1), E.mean(axis=1), R.mean(axis=1)])
        point = np.array([a0.mean(), e0.mean(), r0.mean()], dtype=float)
        gradient = np.full(3, 1.0 / 3.0)
    elif kind == "geometric":
        summaries = np.column_stack([
            np.cbrt(A).mean(axis=1),
            np.cbrt(E).mean(axis=1),
            np.cbrt(R).mean(axis=1),
        ])
        point = np.array([
            np.cbrt(a0).mean(),
            np.cbrt(e0).mean(),
            np.cbrt(r0).mean(),
        ], dtype=float)
        gradient = np.array([
            point[1] * point[2],
            point[0] * point[2],
            point[0] * point[1],
        ], dtype=float)
    elif kind in {"rms", "topsis"}:
        if kind == "rms":
            ga, ge, gr = rms_entry_gradients(a0, e0, r0)
        else:
            ga, ge, gr = topsis_entry_gradients(a0, e0, r0)
        summaries = np.column_stack([A @ ga, E @ ge, R @ gr])
        point = np.array([a0 @ ga, e0 @ ge, r0 @ gr], dtype=float)
        gradient = np.ones(3, dtype=float)
    else:
        raise VerificationError(f"unsupported aggregator: {kind}")
    return summaries, point, gradient


def delta_block(kind: str, A: np.ndarray, E: np.ndarray, R: np.ndarray,
                a0: np.ndarray, e0: np.ndarray, r0: np.ndarray,
                alpha: float) -> dict[str, Any]:
    summaries, point, gradient = delta_inputs(kind, A, E, R, a0, e0, r0)
    Sigma = np.cov(summaries, rowvar=False, ddof=1)
    Corr = np.corrcoef(summaries, rowvar=False)
    var_measured = float(gradient @ Sigma @ gradient)
    var_zero = float(np.sum((gradient ** 2) * np.diag(Sigma)))
    require(var_measured > 0.0, f"{kind} measured delta variance is not positive")
    require(var_zero > 0.0, f"{kind} declared-zero delta variance is not positive")
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    width_measured = 2.0 * z * math.sqrt(var_measured)
    width_zero = 2.0 * z * math.sqrt(var_zero)
    return {
        "summary_point": point.tolist(),
        "gradient_at_point": gradient.tolist(),
        "Sigma_of_summaries": Sigma.tolist(),
        "Corr_of_summaries": Corr.tolist(),
        "var_measured_covariance": var_measured,
        "var_cross_terms_declared_zero": var_zero,
        "width_delta_measured": width_measured,
        "width_delta_declared_zero": width_zero,
        "understatement_of_width_pct": 100.0 * (1.0 - width_zero / width_measured),
    }


def point_curves(model: dict[str, Any], label: str,
                 L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = require_mapping(require_key(model, "curves_point", label),
                             f"{label}.curves_point")
    aliases = {
        "RGA": ("RGA_partial", "RGA"),
        "RGE": ("RGE_greedy_mean_baseline", "RGE"),
        "tail": ("RGR_tail_swap", "RGR_tail"),
        "gaussian": ("RGR_scaled_gaussian_noise", "RGR_gaussian"),
    }

    def one(name: str) -> np.ndarray:
        for key in aliases[name]:
            if key in curves:
                value = require_array(curves[key], f"{label}.curves_point.{key}", ndim=1)
                require(len(value) == L,
                        f"{label}.curves_point.{key} length is not curve_length_L")
                return value
        raise VerificationError(
            f"{label}.curves_point lacks one of {aliases[name]}"
        )

    return one("RGA"), one("RGE"), one("tail"), one("gaussian")


def compare_present_aliases(label: str, actual: Any,
                            mapping: dict[str, Any], keys: tuple[str, ...],
                            *, required: bool = True) -> None:
    """Compare every emitted alias, preventing contradictory duplicate fields."""
    present = [key for key in keys if key in mapping]
    if required:
        require(bool(present), f"missing {label}; accepted keys are {keys}")
    for key in present:
        compare(f"{label}[{key}]", actual, mapping[key])


def computed_primary(E: np.ndarray, Rt: np.ndarray, Rg: np.ndarray,
                     label: str) -> dict[str, Any]:
    corr_tail = manual_corr(E.mean(axis=1), Rt.mean(axis=1),
                            f"{label}.tail correlation")
    corr_gaussian = manual_corr(E.mean(axis=1), Rg.mean(axis=1),
                                f"{label}.Gaussian correlation")
    fisher_eps = 1e-12
    tail_for_z = float(np.clip(corr_tail, -1.0 + fisher_eps,
                               1.0 - fisher_eps))
    gaussian_for_z = float(np.clip(corr_gaussian, -1.0 + fisher_eps,
                                   1.0 - fisher_eps))
    contrast = math.atanh(tail_for_z) - math.atanh(gaussian_for_z)
    return {
        "corr_RGE_RGR_tail": corr_tail,
        "corr_RGE_RGR_gaussian": corr_gaussian,
        "fisher_z_tail_minus_gaussian": contrast,
        "prespecified_direction_delta_lt_zero": bool(contrast < 0.0),
    }


def verify_primary(label: str, model: dict[str, Any], E: np.ndarray,
                   Rt: np.ndarray, Rg: np.ndarray) -> dict[str, Any]:
    computed = computed_primary(E, Rt, Rg, label)
    blocks = [key for key in ("primary_endpoint", "primary_family_contrast")
              if key in model]
    require(bool(blocks),
            f"missing {label}.primary_endpoint/primary_family_contrast")
    for block_name in blocks:
        primary = require_mapping(model[block_name], f"{label}.{block_name}")
        verify_primary_mapping(f"{label}.{block_name}", primary, computed)
    return computed


def verify_primary_mapping(label: str, primary: dict[str, Any],
                           computed: dict[str, Any]) -> None:
    compare_present_aliases(
        f"{label}.corr_E_tail",
        computed["corr_RGE_RGR_tail"], primary,
        ("corr_E_tail", "corr_RGE_RGR_tail"),
    )
    compare_present_aliases(
        f"{label}.corr_E_gaussian",
        computed["corr_RGE_RGR_gaussian"], primary,
        ("corr_E_gaussian", "corr_RGE_RGR_gaussian"),
    )
    compare_present_aliases(
        f"{label}.delta_fisher_z",
        computed["fisher_z_tail_minus_gaussian"], primary,
        ("delta_fisher_z", "fisher_z_tail_minus_gaussian"),
    )
    direction = require_key(primary, "prespecified_direction_delta_lt_zero", label)
    require(isinstance(direction, bool),
            f"{label}.prespecified_direction_delta_lt_zero must be boolean")
    require(direction == computed["prespecified_direction_delta_lt_zero"],
            f"{label}: recorded primary direction disagrees with Delta")


def require_npz_matrix(npz: Any, key: str,
                       shape: tuple[int, int]) -> np.ndarray:
    require(key in npz.files, f"replicate archive missing {key}")
    value = require_array(npz[key], key, ndim=2)
    require(value.shape == shape,
            f"{key} shape must be {shape}, got {value.shape}")
    return value


def require_npz_vector(npz: Any, key: str, length: int) -> np.ndarray:
    require(key in npz.files, f"replicate archive missing {key}")
    value = require_array(npz[key], key, ndim=1)
    require(value.shape == (length,),
            f"{key} shape must be {(length,)}, got {value.shape}")
    return value


def validate_provenance(dataset: dict[str, Any], slug: str, d: int) -> None:
    prov = require_mapping(
        pick_key(dataset, ("provenance", "dataset_provenance"),
                 f"datasets.{slug}.provenance"),
        f"datasets.{slug}.provenance",
    )
    openml = require_mapping(require_key(prov, "openml", f"datasets.{slug}.provenance"),
                             f"datasets.{slug}.provenance.openml")
    for key in ("id", "name", "version", "md5_checksum", "url", "license"):
        require_key(openml, key, f"datasets.{slug}.dataset_provenance.openml")
    require(require_int(openml["id"], f"{slug}.openml.id") > 0,
            f"{slug}.openml.id must be positive")
    require(require_int(openml["version"], f"{slug}.openml.version") > 0,
            f"{slug}.openml.version must be positive")
    require(isinstance(openml["name"], str) and openml["name"].strip(),
            f"{slug}.openml.name must be non-empty")
    md5 = str(openml["md5_checksum"]).lower()
    require(bool(MD5_RE.fullmatch(md5)), f"{slug}.openml.md5_checksum is not MD5")
    require(isinstance(openml["url"], str)
            and openml["url"].startswith(("https://", "http://")),
            f"{slug}.openml.url must be HTTP(S)")
    require(isinstance(openml["license"], str) and openml["license"].strip(),
            f"{slug}.openml.license must be recorded; use 'unknown' if unavailable")
    for key in ("manifest_download_sha256", "raw_frame_sha256"):
        digest = str(require_key(prov, key, f"datasets.{slug}.provenance")).lower()
        require(bool(SHA256_RE.fullmatch(digest)),
                f"{slug}.provenance.{key} is not SHA-256")

    prep = require_mapping(require_key(dataset, "preprocessing", f"datasets.{slug}"),
                           f"datasets.{slug}.preprocessing")
    required = (
        "adapter_version", "raw_d", "encoded_d", "numeric_columns",
        "categorical_columns", "numeric_fill_values", "categorical_fill_values",
        "category_maps", "train_imputed_cells", "test_imputed_cells",
        "unseen_test_levels", "processed_train_sha256", "processed_test_sha256",
    )
    for key in required:
        require_key(prep, key, f"datasets.{slug}.preprocessing")
    raw_d = require_int(prep["raw_d"], f"{slug}.preprocessing.raw_d")
    encoded_d = require_int(prep["encoded_d"], f"{slug}.preprocessing.encoded_d")
    require(raw_d > 0, f"{slug}: raw_d must be positive")
    require(encoded_d > 0, f"{slug}: encoded_d must be positive")
    require(encoded_d == d,
            f"{slug}: encoded_d must equal the curve dimension d")
    require(isinstance(prep["adapter_version"], str)
            and prep["adapter_version"].strip(),
            f"{slug}: adapter_version must be a non-empty string")
    numeric = require_list(prep["numeric_columns"], f"{slug}.numeric_columns")
    categorical = require_list(prep["categorical_columns"], f"{slug}.categorical_columns")
    require(all(isinstance(x, str) and x for x in numeric + categorical),
            f"{slug}: preprocessing column names must be non-empty strings")
    require(len(numeric) == len(set(numeric)),
            f"{slug}: numeric_columns contains duplicates")
    require(len(categorical) == len(set(categorical)),
            f"{slug}: categorical_columns contains duplicates")
    require(set(numeric).isdisjoint(categorical),
            f"{slug}: numeric/categorical lists overlap")
    require(len(numeric) + len(categorical) == raw_d,
            f"{slug}: numeric/categorical lists do not cover raw_d")
    numeric_fill = require_mapping(prep["numeric_fill_values"],
                                   f"{slug}.numeric_fill_values")
    categorical_fill = require_mapping(prep["categorical_fill_values"],
                                       f"{slug}.categorical_fill_values")
    category_maps = require_mapping(prep["category_maps"], f"{slug}.category_maps")
    require(set(numeric_fill) == set(numeric),
            f"{slug}: numeric fill values must cover every numeric column exactly")
    require(set(categorical_fill) == set(categorical),
            f"{slug}: categorical fill values must cover every categorical column exactly")
    require(set(category_maps) == set(categorical),
            f"{slug}: category maps must cover every categorical column exactly")
    for name, value in numeric_fill.items():
        require_finite_scalar(value, f"{slug}.numeric_fill_values.{name}")
    for name, value in categorical_fill.items():
        require(isinstance(value, str) and value,
                f"{slug}.categorical_fill_values.{name} must be a non-empty string")
    for name, raw_levels in category_maps.items():
        levels = require_list(raw_levels, f"{slug}.category_maps.{name}")
        require(bool(levels) and all(isinstance(value, str) and value for value in levels),
                f"{slug}.category_maps.{name} must contain non-empty strings")
        require(len(levels) == len(set(levels)),
                f"{slug}.category_maps.{name} contains duplicate levels")
    if "encoded_columns" in prep:
        encoded = require_list(prep["encoded_columns"], f"{slug}.encoded_columns")
        require(len(encoded) == encoded_d,
                f"{slug}: encoded_columns must have encoded_d entries")
        require(len(set(encoded)) == encoded_d,
                f"{slug}: encoded_columns contains duplicates")
    for key in ("train_imputed_cells", "test_imputed_cells", "unseen_test_levels"):
        require(require_int(prep[key], f"{slug}.preprocessing.{key}") >= 0,
                f"{slug}.preprocessing.{key} must be non-negative")
    for key in ("processed_train_sha256", "processed_test_sha256"):
        digest = str(prep[key]).lower()
        require(bool(SHA256_RE.fullmatch(digest)),
                f"{slug}.preprocessing.{key} is not SHA-256")


def validate_audit_arrays(npz: Any, dataset: dict[str, Any],
                          slug: str, d: int, L: int) -> None:
    keys = set(npz.files)
    n_train = require_int(require_key(dataset, "n_train", f"datasets.{slug}"),
                          f"{slug}.n_train")
    n_test = require_int(require_key(dataset, "n_test", f"datasets.{slug}"),
                         f"{slug}.n_test")
    n_retained = require_int(require_key(dataset, "n_retained", f"datasets.{slug}"),
                             f"{slug}.n_retained")
    require(n_train > 0 and n_test > 0 and n_retained > 0,
            f"{slug}: retained/split counts must be positive")
    require(n_train + n_test == n_retained,
            f"{slug}: train/test counts do not sum to n_retained")
    sampling = require_mapping(require_key(dataset, "sampling", f"datasets.{slug}"),
                               f"datasets.{slug}.sampling")
    recorded_selected_digest = str(require_key(
        sampling, "retained_source_indices_sha256", f"datasets.{slug}.sampling"
    )).lower()
    require(bool(SHA256_RE.fullmatch(recorded_selected_digest)),
            f"{slug}: retained source-index digest is not SHA-256")
    adverse_test = require_int(require_key(
        dataset, "adverse_count_test", f"datasets.{slug}"
    ), f"{slug}.adverse_count_test")
    adverse_train = require_int(require_key(
        dataset, "adverse_count_train", f"datasets.{slug}"
    ), f"{slug}.adverse_count_train")
    adverse_retained = require_int(require_key(
        dataset, "adverse_count_retained", f"datasets.{slug}"
    ), f"{slug}.adverse_count_retained")
    require(adverse_train + adverse_test == adverse_retained,
            f"{slug}: train/test adverse counts do not sum to retained count")
    require(0 < adverse_train < n_train and 0 < adverse_test < n_test,
            f"{slug}: a split is missing one target class")

    require("severity_t_over_d" in keys and "tail_p" in keys,
            f"{slug}: replicate archive missing frozen severity grids")
    severity = require_array(npz["severity_t_over_d"],
                             f"{slug}.severity_t_over_d", ndim=1)
    tail_p = require_array(npz["tail_p"], f"{slug}.tail_p", ndim=1)
    require(severity.shape == tail_p.shape == (L,),
            f"{slug}: severity grids must be length L")
    expected_severity = np.arange(L, dtype=float) / float(d)
    compare(f"{slug}.severity_t_over_d", expected_severity, severity)
    compare(f"{slug}.tail_p", 0.5 * expected_severity, tail_p)

    if "train_indices" in keys or "test_indices" in keys:
        require({"train_indices", "test_indices"}.issubset(keys),
                f"{slug}: train_indices and test_indices must appear together")
        train = np.asarray(npz["train_indices"])
        test = np.asarray(npz["test_indices"])
        require(train.ndim == test.ndim == 1, f"{slug}: split indices must be 1-D")
        require(np.issubdtype(train.dtype, np.integer)
                and np.issubdtype(test.dtype, np.integer),
                f"{slug}: split indices must be integers")
        require(np.all(train >= 0) and np.all(test >= 0),
                f"{slug}: split indices must be non-negative")
        require(len(np.unique(train)) == len(train)
                and len(np.unique(test)) == len(test),
                f"{slug}: split indices contain duplicates")
        require(len(np.intersect1d(train, test)) == 0,
                f"{slug}: train/test indices overlap")
        require(len(train) == n_train and len(test) == n_test,
                f"{slug}: split-index lengths disagree with JSON counts")
        selected = np.sort(np.concatenate([train, test])).astype(np.int64, copy=False)
        require(sha256_array(selected) == recorded_selected_digest,
                f"{slug}: retained source-index digest disagrees with split arrays")

    if "y_test" in keys:
        y_test = np.asarray(npz["y_test"])
        require(y_test.ndim == 1 and len(y_test) == n_test,
                f"{slug}: y_test must be a length-n_test vector")
        require(np.issubdtype(y_test.dtype, np.integer),
                f"{slug}: y_test must be integer")
        require(set(np.unique(y_test)) == {0, 1},
                f"{slug}: y_test must contain both binary classes")
        require(int(y_test.sum()) == adverse_test,
                f"{slug}: y_test adverse count disagrees with JSON")


def verify_arm(label: str, arm: dict[str, Any], kind: str,
               A: np.ndarray, E: np.ndarray, R: np.ndarray,
               a0: np.ndarray, e0: np.ndarray, r0: np.ndarray,
               alpha: float) -> dict[str, Any]:
    values = VOLUME_FUNCTIONS[kind](A, E, R)
    pct = percentile_block(values, alpha)
    recorded_pct = require_mapping(require_key(arm, "paired_bootstrap", label),
                                   f"{label}.paired_bootstrap")
    compare(f"{label}.paired_bootstrap.ci95_percentile",
            pct["ci95_percentile"],
            require_key(recorded_pct, "ci95_percentile", f"{label}.paired_bootstrap"))
    compare(f"{label}.paired_bootstrap.width", pct["width"],
            require_key(recorded_pct, "width", f"{label}.paired_bootstrap"))
    for optional in ("mean", "sd"):
        if optional in recorded_pct:
            compare(f"{label}.paired_bootstrap.{optional}", pct[optional],
                    recorded_pct[optional])

    point_value = float(VOLUME_FUNCTIONS[kind](
        a0[None, :], e0[None, :], r0[None, :]
    )[0])
    compare(f"{label}.V_point", point_value,
            require_key(arm, "V_point", label))

    delta = delta_block(kind, A, E, R, a0, e0, r0, alpha)
    recorded_delta = require_mapping(require_key(arm, "delta_method", label),
                                     f"{label}.delta_method")
    required_aliases = {
        "summary_point": ("summary_point",),
        "gradient_at_point": ("gradient_at_point",),
        "Sigma_of_summaries": ("Sigma_of_summaries",),
        "width_delta_measured": ("width_delta_measured",),
        "width_delta_declared_zero": (
            "width_delta_declared_zero", "width_delta_cross_covariances_zero",
        ),
        "understatement_of_width_pct": (
            "understatement_of_width_pct", "signed_width_change_pct_zero_vs_measured",
        ),
    }
    for computed_key, recorded_keys in required_aliases.items():
        compare_present_aliases(
            f"{label}.delta_method.{computed_key}",
            delta[computed_key], recorded_delta, recorded_keys,
        )
    optional_aliases = {
        "Corr_of_summaries": ("Corr_of_summaries",),
        "var_measured_covariance": ("var_measured_covariance",),
        "var_cross_terms_declared_zero": (
            "var_cross_terms_declared_zero", "var_cross_covariances_zero",
        ),
    }
    for computed_key, recorded_keys in optional_aliases.items():
        compare_present_aliases(
            f"{label}.delta_method.{computed_key}",
            delta[computed_key], recorded_delta, recorded_keys, required=False,
        )
    return {
        "point": point_value,
        "percentile": pct,
        "delta": delta,
    }


def verify_arms(label: str, model: dict[str, Any],
                A: np.ndarray, E: np.ndarray, R: np.ndarray,
                a0: np.ndarray, e0: np.ndarray, r0: np.ndarray,
                alpha: float) -> dict[str, Any]:
    block_names = [key for key in ("composites_tail", "arms") if key in model]
    require(bool(block_names), f"missing {label}.composites_tail/arms")
    blocks = [require_mapping(model[key], f"{label}.{key}") for key in block_names]
    if len(blocks) == 2:
        require(blocks[0] == blocks[1],
                f"{label}.composites_tail and {label}.arms disagree")
    arms = blocks[0]
    require(set(AGGREGATORS).issubset(arms),
            f"{label}: missing one or more composite arms {AGGREGATORS}")
    return {
        kind: verify_arm(
            f"{label}.composites_tail.{kind}",
            require_mapping(arms[kind], f"{label}.composites_tail.{kind}"),
            kind, A, E, R, a0, e0, r0, alpha,
        )
        for kind in AGGREGATORS
    }


def verify_conditioning(label: str, tag: str, model: dict[str, Any], npz: Any,
                        B: int, L: int, A: np.ndarray, E: np.ndarray,
                        Rt: np.ndarray, Rg: np.ndarray,
                        encoded_columns: list[str]) -> dict[str, Any]:
    block_names = [key for key in (
        "conditioning_check", "conditioning_check_curve_recompute"
    ) if key in model]
    require(bool(block_names), f"missing {label}.conditioning_check")
    blocks = [require_mapping(model[key], f"{label}.{key}") for key in block_names]
    if len(blocks) == 2:
        require(blocks[0] == blocks[1],
                f"{label}: conditioning aliases disagree")
    conditioning = blocks[0]
    Bf = require_int(
        pick_key(conditioning, ("B_full", "replicates", "n_replicates"),
                 f"{label}.conditioning_check.B_full"),
        f"{label}.conditioning_check.B_full",
    )
    require(Bf == PRODUCTION_B_FULL[tag],
            f"{label}: conditioning result count must be {PRODUCTION_B_FULL[tag]}")
    require(require_int(require_key(conditioning, "B_requested",
                                    f"{label}.conditioning_check"),
                        f"{label}.conditioning_check.B_requested") == Bf,
            f"{label}: B_requested disagrees with B_full")
    require(require_int(require_key(conditioning, "B_completed",
                                    f"{label}.conditioning_check"),
                        f"{label}.conditioning_check.B_completed") == Bf,
            f"{label}: B_completed disagrees with B_full")
    require(Bf <= B, f"{label}: conditioning count exceeds primary B")

    Af = require_npz_matrix(npz, f"{tag}_conditioning_Ab", (Bf, L))
    Ef = require_npz_matrix(npz, f"{tag}_conditioning_Eb", (Bf, L))
    Rf = require_npz_matrix(npz, f"{tag}_conditioning_Rb_tail", (Bf, L))
    # RGA has no fitted removal order or perturbation draw to recondition, so
    # its full-recompute values must equal the first paired primary replicates.
    compare(f"{label}.conditioning.RGA_vs_primary", Af, A[:Bf])

    orders = require_list(require_key(conditioning, "greedy_orders",
                                      f"{label}.conditioning_check"),
                          f"{label}.conditioning_check.greedy_orders")
    require(len(orders) == Bf,
            f"{label}: conditioning greedy_orders must have B_full entries")
    expected_names = set(encoded_columns)
    for index, raw_order in enumerate(orders):
        order = require_list(raw_order, f"{label}.conditioning.greedy_orders[{index}]")
        require(all(isinstance(name, str) and name for name in order),
                f"{label}: conditioning order names must be non-empty strings")
        require(len(order) == L - 1 and len(set(order)) == L - 1,
                f"{label}: conditioning order {index} is not a permutation")
        if expected_names:
            require(set(order) == expected_names,
                    f"{label}: conditioning order {index} disagrees with encoded columns")

    recorded_arms = require_mapping(
        pick_key(conditioning, ("composites_tail", "arms"),
                 f"{label}.conditioning_check.composites_tail"),
        f"{label}.conditioning_check.composites_tail",
    )
    require(set(AGGREGATORS).issubset(recorded_arms),
            f"{label}: conditioning check lacks one or more composite arms")
    computed_arms: dict[str, Any] = {}
    for kind in AGGREGATORS:
        frozen_values = VOLUME_FUNCTIONS[kind](A[:Bf], E[:Bf], Rt[:Bf])
        full_values = VOLUME_FUNCTIONS[kind](Af, Ef, Rf)
        sd_frozen = float(frozen_values.std(ddof=1))
        sd_full = float(full_values.std(ddof=1))
        ratio = sd_full / sd_frozen if sd_frozen > 0.0 else None
        computed = {
            "sd_frozen": sd_frozen,
            "sd_full": sd_full,
            "mean_abs_paired_difference": float(
                np.mean(np.abs(full_values - frozen_values))
            ),
            "max_abs_paired_difference": float(
                np.max(np.abs(full_values - frozen_values))
            ),
        }
        recorded = require_mapping(recorded_arms[kind],
                                   f"{label}.conditioning.{kind}")
        for key, value in computed.items():
            compare(f"{label}.conditioning.{kind}.{key}", value,
                    require_key(recorded, key, f"{label}.conditioning.{kind}"))
        recorded_ratio = require_key(recorded, "sd_ratio_full_over_frozen",
                                     f"{label}.conditioning.{kind}")
        if ratio is None:
            require(recorded_ratio is None,
                    f"{label}.conditioning.{kind}.sd_ratio must be null when "
                    "the frozen SD is zero")
        else:
            compare(f"{label}.conditioning.{kind}.sd_ratio_full_over_frozen",
                    ratio, recorded_ratio)
        computed["sd_ratio_full_over_frozen"] = ratio
        computed_arms[kind] = computed

    try:
        primary_frozen = computed_primary(E[:Bf], Rt[:Bf], Rg[:Bf],
                                          f"{label}.conditioning.frozen")
        primary_full = computed_primary(Ef, Rf, Rg[:Bf],
                                        f"{label}.conditioning.full")
    except VerificationError as exc:
        require("primary_endpoint_note" in conditioning,
                f"{label}: undefined conditioning primary endpoint lacks a note")
        primary_report: dict[str, Any] = {"note": str(exc)}
    else:
        frozen_recorded = require_mapping(
            require_key(conditioning, "primary_endpoint_frozen",
                        f"{label}.conditioning_check"),
            f"{label}.conditioning_check.primary_endpoint_frozen",
        )
        full_recorded = require_mapping(
            require_key(conditioning, "primary_endpoint_full_order_tail",
                        f"{label}.conditioning_check"),
            f"{label}.conditioning_check.primary_endpoint_full_order_tail",
        )
        verify_primary_mapping(f"{label}.conditioning.primary_endpoint_frozen",
                               frozen_recorded, primary_frozen)
        verify_primary_mapping(f"{label}.conditioning.primary_endpoint_full_order_tail",
                               full_recorded, primary_full)
        primary_report = {"frozen": primary_frozen, "full_order_tail": primary_full}

    return {
        "B_full": Bf,
        "arms_tail": computed_arms,
        "primary": primary_report,
    }


def verify_dataset(slug: str, dataset: dict[str, Any], base: Path,
                   alpha: float, B: int) -> dict[str, Any]:
    label = f"datasets.{slug}"
    d = require_int(require_key(dataset, "d", label), f"{label}.d")
    L = require_int(require_key(dataset, "curve_length_L", label),
                    f"{label}.curve_length_L")
    require(L == d + 1, f"{slug}: curve_length_L must equal d + 1")
    if "B" in dataset:
        require(require_int(dataset["B"], f"{label}.B") == B,
                f"{slug}: dataset B differs from root B")
    validate_provenance(dataset, slug, d)

    expected_npz_name = f"replicates-{slug}.npz"
    recorded_npz_name = require_key(dataset, "replicates_file", label)
    require(isinstance(recorded_npz_name, str)
            and recorded_npz_name == expected_npz_name,
            f"{slug}: recorded replicates_file must be {expected_npz_name!r}")
    npz_path = base / expected_npz_name
    require(npz_path.is_file(), f"missing replicate archive: {npz_path}")
    actual_npz_digest = sha256_file(npz_path)
    recorded_npz_digest = str(require_key(dataset, "replicates_sha256", label)).lower()
    require(bool(SHA256_RE.fullmatch(recorded_npz_digest)),
            f"{slug}: recorded replicates_sha256 is not SHA-256")
    require(recorded_npz_digest == actual_npz_digest,
            f"{slug}: recorded replicates_sha256 disagrees with the archive")
    models = require_mapping(require_key(dataset, "models", label), f"{label}.models")
    require(set(models) == set(MODEL_TAGS),
            f"{slug}: models must be exactly {MODEL_TAGS}")
    preprocessing = require_mapping(require_key(dataset, "preprocessing", label),
                                    f"{label}.preprocessing")
    encoded_columns = require_list(
        require_key(preprocessing, "encoded_columns", f"{label}.preprocessing"),
        f"{label}.preprocessing.encoded_columns",
    )

    computed_models: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=False) as npz:
        validate_audit_arrays(npz, dataset, slug, d, L)
        for tag in MODEL_TAGS:
            model_label = f"{label}.models.{tag}"
            model = require_mapping(models[tag], model_label)
            A = require_npz_matrix(npz, f"{tag}_Ab", (B, L))
            E = require_npz_matrix(npz, f"{tag}_Eb", (B, L))
            Rt = require_npz_matrix(npz, f"{tag}_Rb_tail", (B, L))
            Rg = require_npz_matrix(npz, f"{tag}_Rb_gaussian", (B, L))
            a0 = require_npz_vector(npz, f"{tag}_a0", L)
            e0 = require_npz_vector(npz, f"{tag}_e0", L)
            rt0 = require_npz_vector(npz, f"{tag}_r0_tail", L)
            rg0 = require_npz_vector(npz, f"{tag}_r0_gaussian", L)
            json_points = point_curves(model, model_label, L)
            for curve_name, deposited, recorded in zip(
                ("RGA", "RGE", "RGR_tail", "RGR_gaussian"),
                (a0, e0, rt0, rg0), json_points,
            ):
                compare(f"{model_label}.point_curve.{curve_name}", deposited, recorded)

            primary = verify_primary(model_label, model, E, Rt, Rg)
            arm_results = verify_arms(model_label, model, A, E, Rt,
                                      a0, e0, rt0, alpha)

            flip_model = require_mapping(
                require_key(model, "label_complement", model_label),
                f"{model_label}.label_complement",
            )
            Af = require_npz_matrix(npz, f"{tag}_flip_Ab", (B, L))
            Ef = require_npz_matrix(npz, f"{tag}_flip_Eb", (B, L))
            Rtf = require_npz_matrix(npz, f"{tag}_flip_Rb_tail", (B, L))
            Rgf = require_npz_matrix(npz, f"{tag}_flip_Rb_gaussian", (B, L))
            a0f = require_npz_vector(npz, f"{tag}_flip_a0", L)
            e0f = require_npz_vector(npz, f"{tag}_flip_e0", L)
            rt0f = require_npz_vector(npz, f"{tag}_flip_r0_tail", L)
            rg0f = require_npz_vector(npz, f"{tag}_flip_r0_gaussian", L)
            flip_json_points = point_curves(
                flip_model, f"{model_label}.label_complement", L
            )
            for curve_name, deposited, recorded in zip(
                ("RGA", "RGE", "RGR_tail", "RGR_gaussian"),
                (a0f, e0f, rt0f, rg0f), flip_json_points,
            ):
                compare(f"{model_label}.label_complement.point_curve.{curve_name}",
                        deposited, recorded)
            flip_primary = verify_primary(
                f"{model_label}.label_complement", flip_model, Ef, Rtf, Rgf
            )
            flip_arms = verify_arms(
                f"{model_label}.label_complement", flip_model,
                Af, Ef, Rtf, a0f, e0f, rt0f, alpha,
            )

            conditioning = verify_conditioning(
                model_label, tag, model, npz, B, L, A, E, Rt, Rg,
                encoded_columns,
            )

            computed_models[tag] = {
                **primary,
                "arms_tail": arm_results,
                "conditioning_check": conditioning,
                "label_complement": {
                    **flip_primary,
                    "arms_tail": flip_arms,
                },
            }

    return {
        "replicates_file": npz_path.name,
        "replicates_sha256": actual_npz_digest,
        "n_retained": require_int(dataset["n_retained"], f"{slug}.n_retained"),
        "n_train": require_int(dataset["n_train"], f"{slug}.n_train"),
        "n_test": require_int(dataset["n_test"], f"{slug}.n_test"),
        "raw_d": require_int(preprocessing["raw_d"], f"{slug}.raw_d"),
        "d": d,
        "encoded_d": d,
        "curve_length_L": L,
        "cap_applied": bool(dataset["sampling"]["applied"]),
        "models": computed_models,
    }


def digest_from_result(results: dict[str, Any], name: str) -> str:
    containers = [results]
    for key in ("config", "provenance"):
        if isinstance(results.get(key), dict):
            containers.append(results[key])
    found: list[str] = []
    for container in containers:
        if name in container:
            digest = str(container[name]).lower()
            require(bool(SHA256_RE.fullmatch(digest)), f"{name} is not SHA-256")
            found.append(digest)
    require(bool(found), f"result does not record {name}")
    require(len(set(found)) == 1,
            f"contradictory {name} values appear in result containers")
    return found[0]


def dataset_slug_from_result(results: dict[str, Any]) -> str:
    dataset = require_key(results, "dataset", "root")
    if isinstance(dataset, str):
        slug = dataset
    else:
        dataset_obj = require_mapping(dataset, "root.dataset")
        slug = str(pick_key(dataset_obj, ("slug", "dataset_slug"),
                            "root.dataset.slug"))
    require(bool(SLUG_RE.fullmatch(slug)), f"invalid result dataset slug: {slug!r}")
    return slug


def validate_run_artifact_bindings(results: dict[str, Any], base: Path) -> None:
    expected = {
        "requirements_sha256": (
            base / "requirements-six-dataset.txt", FROZEN_REQUIREMENTS_SHA256
        ),
        "source_audit_sha256": (
            base / "data-source-audit.json", FROZEN_SOURCE_AUDIT_SHA256
        ),
        "schema_audit_sha256": (
            base / "schema-preflight-audit.json", FROZEN_SCHEMA_AUDIT_SHA256
        ),
    }
    for result_key, (path, frozen_digest) in expected.items():
        require(path.is_file(), f"missing frozen run artifact: {path}")
        actual = sha256_file(path)
        require(actual == frozen_digest,
                f"{path.name} does not match its hard-anchored frozen digest")
        recorded = str(require_key(results, result_key, "root")).lower()
        require(recorded == actual, f"root.{result_key} disagrees with {path.name}")
    runner_path = base / "run_six_dataset.py"
    require(runner_path.is_file(), f"missing primary runner: {runner_path}")
    require(str(require_key(results, "runner_sha256", "root")).lower()
            == sha256_file(runner_path),
            "root.runner_sha256 disagrees with run_six_dataset.py")
    repository_commit = str(require_key(results, "repository_commit", "root")).lower()
    require(bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", repository_commit)),
            "root.repository_commit is not a full Git object id")


def normalize_per_dataset_result(results: dict[str, Any], expected_slug: str | None,
                                 protocol_digest: str | None,
                                 manifest_digest: str | None) -> tuple[str, dict[str, Any], float, int, str]:
    schema_version = str(require_key(results, "schema_version", "root"))
    require(bool(schema_version.strip()), "root.schema_version must be non-empty")
    status = str(require_key(results, "status", "root")).lower()
    require(status in {"complete", "completed", "ok", "pass", "success"},
            f"dataset result is not complete: status={status!r}")
    slug = dataset_slug_from_result(results)
    if expected_slug is not None:
        require(slug == expected_slug,
                f"result slug {slug!r} does not match expected {expected_slug!r}")
    if "slug" in results:
        require(str(results["slug"]) == slug,
                "root.slug disagrees with root.dataset.slug")

    config = require_mapping(require_key(results, "config", "root"), "root.config")
    B = require_int(require_key(config, "B", "root.config"), "root.config.B")
    require(B == PRODUCTION_B,
            f"production verification requires B={PRODUCTION_B}, got {B}")
    alpha = require_finite_scalar(require_key(config, "alpha", "root.config"),
                                  "root.config.alpha")
    require(abs(alpha - DEFAULT_ALPHA) <= 1e-15,
            f"production verification requires alpha={DEFAULT_ALPHA}")
    b_full = require_mapping(
        require_key(config, "B_conditioning_check", "root.config"),
        "root.config.B_conditioning_check",
    )
    require(set(b_full) == set(MODEL_TAGS),
            f"B_conditioning_check must contain exactly {MODEL_TAGS}")
    for tag in MODEL_TAGS:
        require(require_int(b_full[tag], f"root.config.B_conditioning_check.{tag}")
                == PRODUCTION_B_FULL[tag],
                f"production B_full for {tag} must be {PRODUCTION_B_FULL[tag]}")
    for key, expected in FROZEN_CONFIG.items():
        recorded = require_key(config, key, "root.config")
        if isinstance(expected, int):
            require(require_int(recorded, f"root.config.{key}") == expected,
                    f"root.config.{key} disagrees with the frozen protocol")
        else:
            compare(f"root.config.{key}", expected, recorded,
                    atol=1e-15, rtol=1e-15)

    environment = require_mapping(require_key(results, "environment", "root"),
                                  "root.environment")
    require(str(require_key(environment, "python", "root.environment"))
            == FROZEN_PYTHON,
            "root.environment.python disagrees with the frozen runtime")
    for key, expected in FROZEN_LIBRARIES.items():
        require(str(require_key(environment, key, "root.environment")) == expected,
                f"root.environment.{key} disagrees with the frozen requirements")
    require(str(require_key(environment, "safeai_commit", "root.environment"))
            == FROZEN_SAFEAI_COMMIT,
            "root.environment.safeai_commit disagrees with the frozen substrate")
    threadpools = require_list(require_key(environment, "threadpools", "root.environment"),
                               "root.environment.threadpools")
    for index, raw_pool in enumerate(threadpools):
        pool = require_mapping(raw_pool, f"root.environment.threadpools[{index}]")
        require(require_int(require_key(pool, "num_threads",
                                        f"root.environment.threadpools[{index}]"),
                            f"root.environment.threadpools[{index}].num_threads") == 1,
                "recorded BLAS/OpenMP thread pool was not constrained to one thread")

    if "B" in results:
        require(require_int(results["B"], "root.B") == B,
                "root.B disagrees with root.config.B")
    if "alpha" in results:
        compare("root.alpha", alpha, results["alpha"], atol=1e-15, rtol=1e-15)
    if "B_conditioning_check" in results:
        require(results["B_conditioning_check"] == b_full,
                "root.B_conditioning_check disagrees with root.config")

    recorded_protocol = digest_from_result(results, "protocol_sha256")
    recorded_manifest = digest_from_result(results, "dataset_spec_sha256")
    if protocol_digest is not None:
        require(recorded_protocol == protocol_digest,
                f"{slug}: recorded protocol SHA-256 does not match PROTOCOL.md")
    if manifest_digest is not None:
        require(recorded_manifest == manifest_digest,
                f"{slug}: recorded dataset-spec SHA-256 does not match dataset-manifest.json")
    if "manifest_sha256" in results:
        require(str(results["manifest_sha256"]).lower() == recorded_manifest,
                f"{slug}: manifest_sha256 disagrees with dataset_spec_sha256")

    preprocessing = require_mapping(require_key(results, "preprocessing", "root"),
                                    "root.preprocessing")
    provenance = require_mapping(require_key(results, "provenance", "root"),
                                 "root.provenance")
    if "dataset_provenance" in results:
        require(results["dataset_provenance"] == provenance,
                f"{slug}: provenance and dataset_provenance disagree")
    if "d" in config or "encoded_d" in config:
        d_value = pick_key(config, ("d", "encoded_d"), "root.config.d")
    else:
        d_value = require_key(preprocessing, "encoded_d", "root.preprocessing")
    d = require_int(d_value, "root.config.d")
    if "d" in config and "encoded_d" in config:
        require(require_int(config["encoded_d"], "root.config.encoded_d") == d,
                "root.config.d and encoded_d disagree")
    L = require_int(require_key(config, "curve_length_L", "root.config"),
                    "root.config.curve_length_L")
    if "d" in results:
        require(require_int(results["d"], "root.d") == d,
                "root.d disagrees with root.config.d")
    if "curve_length_L" in results:
        require(require_int(results["curve_length_L"], "root.curve_length_L") == L,
                "root.curve_length_L disagrees with root.config.curve_length_L")
    normalized = {
        "d": d,
        "curve_length_L": L,
        "B": B,
        "replicates_file": require_key(results, "replicates_file", "root"),
        "replicates_sha256": require_key(results, "replicates_sha256", "root"),
        "n_retained": require_key(results, "n_retained", "root"),
        "n_train": require_key(results, "n_train", "root"),
        "n_test": require_key(results, "n_test", "root"),
        "adverse_count_retained": require_key(results, "adverse_count_retained", "root"),
        "adverse_count_train": require_key(results, "adverse_count_train", "root"),
        "adverse_count_test": require_key(results, "adverse_count_test", "root"),
        "sampling": require_mapping(require_key(results, "sampling", "root"),
                                    "root.sampling"),
        "provenance": provenance,
        "preprocessing": preprocessing,
        "models": require_mapping(require_key(results, "models", "root"), "root.models"),
    }
    return slug, normalized, alpha, B, schema_version


def validate_schema_preflight_binding(results: dict[str, Any],
                                      normalized: dict[str, Any],
                                      base: Path, slug: str) -> dict[str, Any]:
    audit_path = base / "schema-preflight-audit.json"
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = require_mapping(json.load(handle), "schema preflight audit")
    require(audit.get("status") == "PASS"
            and audit.get("safe_metrics_or_models_computed") is False,
            "schema preflight audit is not an outcome-blind PASS")
    entries = require_list(require_key(audit, "datasets", "schema preflight audit"),
                           "schema preflight audit.datasets")
    matches = [require_mapping(item, "schema preflight dataset") for item in entries
               if isinstance(item, dict) and str(item.get("slug")) == slug]
    require(len(matches) == 1,
            f"{slug}: schema preflight audit does not contain exactly one entry")
    entry = matches[0]
    require(entry.get("status") == "PASS", f"{slug}: schema preflight did not pass")

    provenance = require_mapping(normalized["provenance"], f"{slug}.provenance")
    preprocessing = require_mapping(normalized["preprocessing"],
                                    f"{slug}.preprocessing")
    compare_pairs = {
        "raw_frame_sha256": provenance["raw_frame_sha256"],
        "processed_train_sha256": preprocessing["processed_train_sha256"],
        "processed_test_sha256": preprocessing["processed_test_sha256"],
        "retained_source_indices_sha256":
            normalized["sampling"]["retained_source_indices_sha256"],
    }
    for key, actual in compare_pairs.items():
        require(str(actual) == str(require_key(entry, key, f"schema audit.{slug}")),
                f"{slug}: {key} disagrees with the frozen schema preflight")
    exact_pairs = {
        "retained_rows": normalized["n_retained"],
        "train_rows": normalized["n_train"],
        "test_rows": normalized["n_test"],
        "retained_adverse_count": normalized["adverse_count_retained"],
        "train_adverse_count": normalized["adverse_count_train"],
        "test_adverse_count": normalized["adverse_count_test"],
        "encoded_d": normalized["d"],
        "curve_length_L": normalized["curve_length_L"],
    }
    for key, actual in exact_pairs.items():
        require(require_int(actual, f"{slug}.{key}")
                == require_int(require_key(entry, key, f"schema audit.{slug}"),
                               f"schema audit.{slug}.{key}"),
                f"{slug}: {key} disagrees with the frozen schema preflight")
    require(preprocessing["numeric_columns"] == entry["numeric_columns"],
            f"{slug}: numeric columns disagree with the frozen schema preflight")
    require(preprocessing["categorical_columns"] == entry["categorical_columns"],
            f"{slug}: categorical columns disagree with the frozen schema preflight")
    require(preprocessing["encoded_columns"] == entry["encoded_columns"],
            f"{slug}: encoded columns disagree with the frozen schema preflight")
    for key in ("train_imputed_cells", "test_imputed_cells", "unseen_test_levels"):
        require(require_int(preprocessing[key], f"{slug}.{key}")
                == require_int(entry[key], f"schema audit.{slug}.{key}"),
                f"{slug}: {key} disagrees with the frozen schema preflight")
    require(preprocessing.get("zero_variance_columns_removed")
            == entry["zero_variance_columns_removed"],
            f"{slug}: removed constants disagree with the frozen schema preflight")
    require(results.get("raw_duplicate_full_rows")
            == entry["duplicate_full_rows_including_target"],
            f"{slug}: duplicate-row count disagrees with the frozen schema preflight")
    require(normalized["sampling"].get("applied") == entry["cap_applied"],
            f"{slug}: cap flag disagrees with the frozen schema preflight")

    expected_contract = {
        "raw_frame_sha256": entry["raw_frame_sha256"],
        "source_rows": entry["source_rows"],
        "raw_d": entry["source_predictors"],
        "source_adverse_count": entry["source_adverse_count"],
        "predictor_names": entry["predictor_names"],
        "predictor_dtypes": entry["predictor_dtypes"],
        "missing_cells_by_predictor": entry["missing_cells_by_predictor"],
        "duplicate_full_rows_including_target":
            entry["duplicate_full_rows_including_target"],
        "retained_source_indices_sha256": entry["retained_source_indices_sha256"],
        "processed_train_sha256": entry["processed_train_sha256"],
        "processed_test_sha256": entry["processed_test_sha256"],
        "train_indices_sha256": entry["train_source_indices_sha256"],
        "test_indices_sha256": entry["test_source_indices_sha256"],
        "n_retained": entry["retained_rows"],
        "n_train": entry["train_rows"],
        "n_test": entry["test_rows"],
        "retained_adverse_count": entry["retained_adverse_count"],
        "train_adverse_count": entry["train_adverse_count"],
        "test_adverse_count": entry["test_adverse_count"],
        "encoded_d": entry["encoded_d"],
        "curve_length_L": entry["curve_length_L"],
    }
    require(require_key(results, "all_six_preflight_match", "root")
            == expected_contract,
            f"{slug}: run-time preflight binding disagrees with the frozen audit")
    return entry


def verify_result_file(results_path: Path, expected_slug: str | None = None,
                       expected_spec: dict[str, Any] | None = None,
                       protocol_digest: str | None = None,
                       manifest_digest: str | None = None) -> dict[str, Any]:
    require(results_path.is_file(), f"results file does not exist: {results_path}")
    with results_path.open("r", encoding="utf-8") as handle:
        results = require_mapping(json.load(handle), "root")
    validate_run_artifact_bindings(results, results_path.parent)
    slug, normalized, alpha, B, schema_version = normalize_per_dataset_result(
        results, expected_slug, protocol_digest, manifest_digest,
    )
    validate_schema_preflight_binding(results, normalized, results_path.parent, slug)
    if expected_spec is not None:
        prov = require_mapping(normalized["provenance"], f"{slug}.provenance")
        openml = require_mapping(require_key(prov, "openml", f"{slug}.provenance"),
                                 f"{slug}.provenance.openml")
        expected_pairs = {
            "id": expected_spec["openml_data_id"],
            "version": expected_spec["openml_version"],
            "name": expected_spec["openml_name"],
            "md5_checksum": expected_spec["openml_md5"],
        }
        for key, expected in expected_pairs.items():
            require(str(openml.get(key)) == str(expected),
                    f"{slug}: provenance.openml.{key} disagrees with frozen manifest")
        require(str(prov.get("manifest_download_sha256"))
                == str(expected_spec["download_sha256"]),
                f"{slug}: provenance download SHA-256 disagrees with frozen manifest")
        require(require_key(results, "dataset_manifest_entry", "root") == expected_spec,
                f"{slug}: embedded dataset manifest entry disagrees with frozen manifest")
        require(str(require_key(results, "manifest_entry_sha256", "root")).lower()
                == sha256_json(expected_spec),
                f"{slug}: manifest-entry SHA-256 disagrees with frozen manifest")
        prep = require_mapping(normalized["preprocessing"], f"{slug}.preprocessing")
        require(require_int(prep["raw_d"], f"{slug}.preprocessing.raw_d")
                == require_int(expected_spec["raw_predictors"],
                               f"manifest.{slug}.raw_predictors"),
                f"{slug}: raw_d disagrees with frozen manifest")
        source_rows = require_int(expected_spec["rows"], f"manifest.{slug}.rows")
        expected_retained = min(30_000, source_rows)
        expected_test = math.ceil(0.30 * expected_retained)
        expected_train = expected_retained - expected_test
        require(require_int(normalized["n_retained"], f"{slug}.n_retained")
                == expected_retained,
                f"{slug}: n_retained disagrees with the frozen 30000-row cap")
        require(require_int(normalized["n_train"], f"{slug}.n_train")
                == expected_train,
                f"{slug}: n_train disagrees with the frozen 70/30 split")
        require(require_int(normalized["n_test"], f"{slug}.n_test")
                == expected_test,
                f"{slug}: n_test disagrees with the frozen 70/30 split")
        sampling = require_mapping(normalized["sampling"], f"{slug}.sampling")
        applied = require_key(sampling, "applied", f"{slug}.sampling")
        require(isinstance(applied, bool) and applied == (source_rows > 30_000),
                f"{slug}: sampling.applied disagrees with the frozen cap rule")
        require(require_int(require_key(sampling, "rng_seed", f"{slug}.sampling"),
                            f"{slug}.sampling.rng_seed")
                == 20260723 + 5000 + require_int(expected_spec["order"],
                                                  f"manifest.{slug}.order"),
                f"{slug}: sampling seed disagrees with the frozen dataset order")
        if source_rows <= 30_000:
            require(require_int(normalized["adverse_count_retained"],
                                f"{slug}.adverse_count_retained")
                    == require_int(expected_spec["adverse_count"],
                                   f"manifest.{slug}.adverse_count"),
                    f"{slug}: uncapped retained target count disagrees with manifest")
    computed = verify_dataset(slug, normalized, results_path.parent, alpha, B)
    return {
        "slug": slug,
        "results_file": results_path.name,
        "results_sha256": sha256_file(results_path),
        "schema_version_verified": schema_version,
        "computed": computed,
    }


def frozen_specs(manifest_path: Path) -> list[dict[str, Any]]:
    require(manifest_path.is_file(), f"missing dataset manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = require_mapping(json.load(handle), "dataset manifest")
    entries = require_list(require_key(manifest, "datasets", "dataset manifest"),
                           "dataset manifest.datasets")
    require(len(entries) == 6, "frozen manifest must contain exactly six datasets")
    ordered: list[tuple[int, str, dict[str, Any]]] = []
    for index, raw in enumerate(entries):
        entry = require_mapping(raw, f"manifest.datasets[{index}]")
        order = require_int(require_key(entry, "order", f"manifest.datasets[{index}]"),
                            f"manifest.datasets[{index}].order")
        slug = str(require_key(entry, "slug", f"manifest.datasets[{index}]"))
        require(bool(SLUG_RE.fullmatch(slug)), f"invalid manifest slug: {slug!r}")
        ordered.append((order, slug, entry))
    ordered.sort(key=lambda item: item[0])
    require([item[0] for item in ordered] == list(range(1, 7)),
            "manifest dataset order must be exactly 1 through 6")
    slugs = [item[1] for item in ordered]
    require(len(set(slugs)) == 6, "manifest contains duplicate slugs")
    return [item[2] for item in ordered]


def verify_results_dir(results_dir: Path) -> dict[str, Any]:
    require(results_dir.is_dir(), f"results directory does not exist: {results_dir}")
    protocol_path = results_dir / "PROTOCOL.md"
    manifest_path = results_dir / "dataset-manifest.json"
    require(protocol_path.is_file(), f"missing frozen protocol: {protocol_path}")
    protocol_digest = sha256_file(protocol_path)
    manifest_digest = sha256_file(manifest_path)
    require(protocol_digest == FROZEN_PROTOCOL_SHA256,
            "PROTOCOL.md does not match the hard-anchored frozen digest")
    require(manifest_digest == FROZEN_MANIFEST_SHA256,
            "dataset-manifest.json does not match the hard-anchored frozen digest")
    specs = frozen_specs(manifest_path)
    order = [str(spec["slug"]) for spec in specs]
    datasets: dict[str, Any] = {}
    for spec in specs:
        slug = str(spec["slug"])
        datasets[slug] = verify_result_file(
            results_dir / f"results-{slug}.json",
            expected_slug=slug,
            expected_spec=spec,
            protocol_digest=protocol_digest,
            manifest_digest=manifest_digest,
        )
    model_arms_negative = 0
    dataset_classes = {"directionally_concordant": [], "mixed": [], "contrary": []}
    covariance_effects: list[dict[str, Any]] = []
    for slug in order:
        models = datasets[slug]["computed"]["models"]
        negative_here = 0
        for tag in MODEL_TAGS:
            model = models[tag]
            if model["fisher_z_tail_minus_gaussian"] < 0.0:
                model_arms_negative += 1
                negative_here += 1
            for kind in AGGREGATORS:
                covariance_effects.append({
                    "slug": slug,
                    "model": tag,
                    "composite": kind,
                    "signed_width_change_pct_zero_vs_measured":
                        model["arms_tail"][kind]["delta"]
                        ["understatement_of_width_pct"],
                })
        if negative_here == 2:
            dataset_classes["directionally_concordant"].append(slug)
        elif negative_here == 1:
            dataset_classes["mixed"].append(slug)
        else:
            dataset_classes["contrary"].append(slug)
    minimum_effect = min(
        covariance_effects,
        key=lambda item: item["signed_width_change_pct_zero_vs_measured"],
    )
    maximum_effect = max(
        covariance_effects,
        key=lambda item: item["signed_width_change_pct_zero_vs_measured"],
    )
    aggregate = {
        "datasets_directionally_concordant_out_of_6":
            len(dataset_classes["directionally_concordant"]),
        "model_arms_delta_lt_zero_out_of_12": model_arms_negative,
        "dataset_classification": dataset_classes,
        "cross_covariance_effect_count": len(covariance_effects),
        "cross_covariance_effect_signed_pct_range": {
            "minimum": minimum_effect,
            "maximum": maximum_effect,
        },
    }
    return {
        "verifier": "verify_six_dataset.py",
        "verifier_schema": "1.3",
        "status": "PASS",
        "scope": "all-six-intended-datasets",
        "protocol_sha256": protocol_digest,
        "dataset_spec_sha256": manifest_digest,
        "B_verified": PRODUCTION_B,
        "B_conditioning_check_verified": PRODUCTION_B_FULL,
        "dataset_order": order,
        "datasets": datasets,
        "aggregate": aggregate,
    }


def brute_volume(curve_a: np.ndarray, curve_e: np.ndarray,
                 curve_r: np.ndarray, kind: str) -> float:
    total = 0.0
    count = 0
    for a in curve_a:
        for e in curve_e:
            for r in curve_r:
                if kind == "arithmetic":
                    value = (a + e + r) / 3.0
                elif kind == "geometric":
                    value = float(np.cbrt(a * e * r))
                elif kind == "rms":
                    value = math.sqrt((a * a + e * e + r * r) / 3.0)
                else:
                    raise ValueError(kind)
                total += value
                count += 1
    return total / count


def self_test() -> None:
    rng = np.random.default_rng(20260827)
    A = rng.uniform(0.15, 0.95, size=(5, 4))
    E = rng.uniform(0.20, 0.90, size=(5, 4))
    R = rng.uniform(0.10, 0.98, size=(5, 4))
    for kind in ("arithmetic", "geometric", "rms"):
        fn = VOLUME_FUNCTIONS[kind]
        vector = fn(A, E, R)
        brute = np.array([brute_volume(A[i], E[i], R[i], kind)
                          for i in range(len(A))])
        compare(f"self-test {kind} volume", vector, brute, atol=2e-14, rtol=2e-14)
    topsis = volume_topsis(A, E, R)
    topsis_direct = []
    positive, negative, weights = topsis_reference(A.shape[1])
    for i in range(len(A)):
        values = np.concatenate([A[i], E[i], R[i]])
        dp = math.sqrt(float(np.sum(((values - positive) * weights) ** 2)))
        dn = math.sqrt(float(np.sum(((values - negative) * weights) ** 2)))
        topsis_direct.append(dn / (dn + dp))
    compare("self-test TOPSIS volume", topsis, topsis_direct,
            atol=2e-14, rtol=2e-14)
    a0, e0, r0 = A[0], E[0], R[0]
    ga, ge, gr = rms_entry_gradients(a0, e0, r0)
    rms0 = brute_volume(a0, e0, r0, "rms")
    compare("self-test RMS Euler identity", ga @ a0 + ge @ e0 + gr @ r0,
            rms0, atol=2e-14, rtol=2e-14)
    for kind, gradient_fn in (
        ("rms", rms_entry_gradients),
        ("topsis", topsis_entry_gradients),
    ):
        analytic = np.concatenate(gradient_fn(a0, e0, r0))
        base = np.concatenate([a0, e0, r0])
        numeric = np.empty_like(base)
        epsilon = 1e-7
        L = len(a0)
        for j in range(len(base)):
            plus, minus = base.copy(), base.copy()
            plus[j] += epsilon
            minus[j] -= epsilon
            fp = VOLUME_FUNCTIONS[kind](
                plus[:L][None], plus[L:2 * L][None], plus[2 * L:][None]
            )[0]
            fm = VOLUME_FUNCTIONS[kind](
                minus[:L][None], minus[L:2 * L][None], minus[2 * L:][None]
            )[0]
            numeric[j] = (fp - fm) / (2.0 * epsilon)
        compare(f"self-test {kind} finite-difference gradient", analytic, numeric,
                atol=2e-9, rtol=2e-8)
    for kind in VOLUME_FUNCTIONS:
        delta = delta_block(kind, A, E, R, a0, e0, r0, DEFAULT_ALPHA)
        require(np.asarray(delta["Sigma_of_summaries"]).shape == (3, 3),
                f"self-test {kind} delta covariance is not 3 by 3")
        require(np.asarray(delta["gradient_at_point"]).shape == (3,),
                f"self-test {kind} delta gradient is not length 3")
    corr = manual_corr(np.array([1.0, 2.0, 3.0]),
                       np.array([2.0, 4.0, 6.0]), "self-test correlation")
    compare("self-test correlation", corr, 1.0, atol=2e-14, rtol=2e-14)
    try:
        compare_present_aliases("self-test contradictory aliases", 1.0,
                                {"old": 1.0, "new": 2.0}, ("old", "new"))
    except VerificationError:
        pass
    else:
        raise VerificationError("self-test failed to reject contradictory aliases")
    print("SELF-TEST PASS")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path,
                        help="verify one per-dataset results-<slug>.json")
    source.add_argument("--results-dir", type=Path,
                        help="verify all six results in the frozen study directory")
    parser.add_argument("--output", type=Path,
                        help="verification report JSON (default follows verification scope)")
    parser.add_argument("--self-test", action="store_true",
                        help="run formula checks without reading experiment outcomes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
        return 0
    single_path = args.results.resolve() if args.results is not None else None
    results_dir = (args.results_dir.resolve() if args.results_dir is not None
                   else DEFAULT_RESULTS_DIR)
    if args.output is not None:
        output_path = args.output.resolve()
    elif single_path is not None:
        suffix = single_path.name.removeprefix("results-")
        output_path = single_path.with_name(f"verify-{suffix}")
    else:
        output_path = results_dir / DEFAULT_REPORT.name
    try:
        if single_path is not None:
            base = single_path.parent
            protocol_path = base / "PROTOCOL.md"
            manifest_path = base / "dataset-manifest.json"
            require(protocol_path.is_file(), f"missing frozen protocol: {protocol_path}")
            require(manifest_path.is_file(), f"missing dataset manifest: {manifest_path}")
            protocol_digest = sha256_file(protocol_path)
            manifest_digest = sha256_file(manifest_path)
            require(protocol_digest == FROZEN_PROTOCOL_SHA256,
                    "PROTOCOL.md does not match the hard-anchored frozen digest")
            require(manifest_digest == FROZEN_MANIFEST_SHA256,
                    "dataset-manifest.json does not match the hard-anchored frozen digest")
            with single_path.open("r", encoding="utf-8") as handle:
                single_root = require_mapping(json.load(handle), "root")
            single_slug = dataset_slug_from_result(single_root)
            matching_specs = [spec for spec in frozen_specs(manifest_path)
                              if str(spec["slug"]) == single_slug]
            require(len(matching_specs) == 1,
                    f"single-result slug {single_slug!r} is not in the frozen manifest")
            one = verify_result_file(
                single_path,
                expected_slug=single_slug,
                expected_spec=matching_specs[0],
                protocol_digest=protocol_digest,
                manifest_digest=manifest_digest,
            )
            report = {
                "verifier": "verify_six_dataset.py",
                "verifier_schema": "1.3",
                "status": "PASS",
                "scope": "single-dataset",
                "dataset": one,
            }
        else:
            report = verify_results_dir(results_dir)
    except (VerificationError, OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as exc:
        failure = {
            "verifier": "verify_six_dataset.py",
            "verifier_schema": "1.3",
            "status": "FAIL",
            "input": str(single_path if single_path is not None else results_dir),
            "error": str(exc),
        }
        write_report(output_path, failure)
        print(f"VERIFY FAIL: {exc}", file=sys.stderr)
        return 1
    write_report(output_path, report)
    count = (len(report["dataset_order"])
             if report["scope"] == "all-six-intended-datasets" else 1)
    print(f"VERIFY PASS: {count} dataset(s); report {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
