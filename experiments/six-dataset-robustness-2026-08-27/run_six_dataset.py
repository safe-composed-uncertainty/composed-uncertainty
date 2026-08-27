#!/usr/bin/env python3
"""Run the frozen six-dataset robustness extension.

This is the primary runner for PROTOCOL.md in this directory.  It deliberately
has no cross-dataset stopping rule: every requested dataset writes its own log
and terminal JSON status.  Raw OpenML data are fetched at run time and are not
redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "dataset-manifest.json"
PROTOCOL_PATH = HERE / "PROTOCOL.md"
SOURCE_AUDIT_PATH = HERE / "data-source-audit.json"
SCHEMA_AUDIT_PATH = HERE / "schema-preflight-audit.json"
REQUIREMENTS_PATH = HERE / "requirements-six-dataset.txt"
RUNNER_PATH = Path(__file__).resolve()
VERIFIER_PATH = HERE / "verify_six_dataset.py"
VERIFIER_SCHEMA_PATH = HERE / "VERIFIER-SCHEMA.md"
SUMMARY_SCRIPT_PATH = HERE / "make_six_dataset_summary.py"
SCHEMA_AUDIT_SCRIPT_PATH = HERE / "audit_schema_preflight.py"
IMPLEMENTATION_NOTES_PATH = HERE / "IMPLEMENTATION-NOTES.md"
RUNBOOK_PATH = HERE / "RUNBOOK.md"
EXPECTED_PROTOCOL_SHA256 = "791769816ef3dbe99a7f67b4abb07deafefdbe45f5aa9415e05708f017bd488d"
EXPECTED_MANIFEST_SHA256 = "b86e0e173e66f0b67f52046065b991e26097c4f98b1eabc7155ec81869042b7b"
EXPECTED_REQUIREMENTS_SHA256 = "0f82ff87831ea010b09b9bdce3467ebdbc608d9166c69bcfa753e517061c489f"
EXPECTED_SOURCE_AUDIT_SHA256 = "42c5ebafd911315a002d1bd259c8b7ee61772b66e32fb4bc482f07bd1d796199"
EXPECTED_SCHEMA_AUDIT_SHA256 = "7dfb10655a7c9999d1535c40e5f8349067a858ebe93bc5d46a1e621be9aeac65"
EXPECTED_VERSIONS = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "pandas": "3.0.5",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
}
EXPECTED_PYTHON = "3.12.3"
SAFEAI_COMMIT = "39768fcd5264c881f7174268bbffda52b298ae89"
SAFEAI_URL = "https://github.com/koleso500/safeai"
RUNTIME = HERE / "_runtime"
SAFEAI_REPO = Path(os.environ.get("SAFEAI_REPO", RUNTIME / "safeai-src"))
STUBS = RUNTIME / "stubs"

SEED = 20260723
BOOT_SEED = 20260724
TEST_SIZE = 0.30
ROW_CAP = 30_000
ALPHA = 0.05
Z = 1.959963984540054
QLO, QHI = 2.5, 97.5
CLASS_ORDER = np.array([0, 1])
MODEL_TAGS = ("logit", "rf")
DEFAULT_CONDITIONING = {"logit": 100, "rf": 30}
CONST_TOL = 1e-12

RGA_CURVE = None
RGE_CURVE = None
RGE_SCORE = None
RGR_SCORE = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def as_json(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return as_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric result: {number!r}")
        return number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    converted = as_json(value)
    tmp.write_text(json.dumps(converted, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.fh = path.open("x", encoding="utf-8")

    def say(self, message: str = "") -> None:
        print(message, flush=True)
        self.fh.write(message + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class RunInterrupted(RuntimeError):
    """A catchable SIGINT/SIGTERM used to persist terminal run status."""


def ensure_substrate() -> str:
    """Provide the exact SafeAI checkout and inert image/adversarial stubs."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not (SAFEAI_REPO / ".git").is_dir():
        if SAFEAI_REPO.exists():
            raise RuntimeError(f"SAFEAI_REPO exists but is not a git checkout: {SAFEAI_REPO}")
        subprocess.check_call(["git", "clone", "--quiet", SAFEAI_URL, str(SAFEAI_REPO)])
        subprocess.check_call(["git", "-C", str(SAFEAI_REPO), "checkout", "--quiet",
                               SAFEAI_COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(SAFEAI_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SAFEAI_COMMIT:
        raise RuntimeError(f"SafeAI commit mismatch: found {head}, require {SAFEAI_COMMIT}")
    dirty = subprocess.check_output(
        ["git", "-C", str(SAFEAI_REPO), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("SafeAI checkout has modified tracked files")

    files = {
        "torch/__init__.py": (
            "class Tensor: pass\n"
            "class _Callable:\n"
            "    def __init__(self,*a,**k): raise RuntimeError('torch stub invoked')\n"
            "    def __call__(self,*a,**k): raise RuntimeError('torch stub invoked')\n"
            "float32 = 'float32'\n"
            "class _NoGrad:\n"
            "    def __call__(self, fn): return fn\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self,*a): return False\n"
            "def no_grad(*a,**k): return _NoGrad()\n"
            "def tensor(*a,**k): raise RuntimeError('torch stub invoked')\n"
            "def softmax(*a,**k): raise RuntimeError('torch stub invoked')\n"
            "def __getattr__(name): return _Callable\n"),
        "torch/nn/__init__.py": "class Module: pass\nfrom . import functional\n",
        "torch/nn/functional.py": (
            "def __getattr__(name):\n"
            "    def f(*a,**k): raise RuntimeError('torch functional stub invoked')\n"
            "    return f\n"),
        "torch/utils/__init__.py": "",
        "torch/utils/data/__init__.py": (
            "class Dataset: pass\n"
            "class DataLoader:\n"
            "    def __init__(self,*a,**k): raise RuntimeError('torch stub invoked')\n"),
        "art/__init__.py": "",
        "art/attacks/__init__.py": "",
        "art/attacks/evasion.py": "".join(
            f"class {name}:\n    def __init__(self,*a,**k): raise RuntimeError('art stub')\n"
            for name in ("FastGradientMethod", "ProjectedGradientDescent", "SquareAttack",
                         "HopSkipJump", "SimBA", "Wasserstein",
                         "SpatialTransformation")),
        "art/estimators/__init__.py": "",
        "art/estimators/classification.py": (
            "class PyTorchClassifier:\n"
            "    def __init__(self,*a,**k): raise RuntimeError('art stub')\n"
            "class SklearnClassifier:\n"
            "    def __init__(self,*a,**k): raise RuntimeError('art stub')\n"),
    }
    for relative, body in files.items():
        path = STUBS / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    sys.path.insert(0, str(STUBS))
    sys.path.insert(0, str(SAFEAI_REPO))
    global RGA_CURVE, RGE_CURVE, RGE_SCORE, RGR_SCORE
    RGA_CURVE = importlib.import_module("safeai.rga").rga_curve
    rge = importlib.import_module("safeai.rge")
    RGE_CURVE, RGE_SCORE = rge.rge_curve, rge.rge_score
    RGR_SCORE = importlib.import_module("safeai.rgr").rgr_score
    return head


def canonical_target(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("target contains missing values")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_raw(spec: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    ds = fetch_openml(data_id=int(spec["openml_data_id"]), as_frame=True,
                      parser="auto")
    details = dict(ds.details)
    if int(details.get("id", -1)) != int(spec["openml_data_id"]):
        raise RuntimeError("OpenML id does not match the frozen manifest")
    if int(details.get("version", -1)) != int(spec["openml_version"]):
        raise RuntimeError("OpenML version does not match the frozen manifest")
    if str(details.get("name", "")) != str(spec["openml_name"]):
        raise RuntimeError("OpenML name does not match the frozen manifest")
    if str(details.get("md5_checksum", "")).lower() != spec["openml_md5"].lower():
        raise RuntimeError("OpenML MD5 does not match the frozen manifest")
    if ds.frame is None:
        raise RuntimeError("OpenML did not return a frame")
    frame = ds.frame.copy()
    target = str(spec["target_column"])
    if target not in frame.columns:
        raise RuntimeError(f"asserted target column {target!r} is absent")
    y_raw = frame.pop(target)  # explicit even where OpenML declares no default target
    if target in frame.columns:
        raise AssertionError("target pop failed")
    if not frame.columns.is_unique:
        raise RuntimeError("predictor names are not unique")
    if frame.shape != (int(spec["rows"]), int(spec["raw_predictors"])):
        raise RuntimeError(f"frozen schema mismatch: got {frame.shape}")

    adverse = {canonical_target(v) for v in spec["adverse_values"]}
    favorable = {canonical_target(v) for v in spec["favorable_values"]}
    overlap = adverse & favorable
    if overlap:
        raise RuntimeError(f"target maps overlap: {sorted(overlap)}")
    encoded = []
    unknown = set()
    for value in y_raw:
        key = canonical_target(value)
        if key in adverse:
            encoded.append(1)
        elif key in favorable:
            encoded.append(0)
        else:
            unknown.add(key)
    if unknown:
        raise RuntimeError(f"unmapped target values: {sorted(unknown)}")
    y = np.asarray(encoded, dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("target does not contain both classes")
    if int(y.sum()) != int(spec["adverse_count"]):
        raise RuntimeError("adverse count does not match the frozen manifest")

    url = str(details.get("url", f"https://www.openml.org/d/{spec['openml_data_id']}"))
    provenance = {
        "openml": {
            "id": int(spec["openml_data_id"]),
            "version": int(spec["openml_version"]),
            "name": str(details.get("name", spec["openml_name"])),
            "md5_checksum": str(details["md5_checksum"]).lower(),
            "url": url,
            "license": str(details.get("licence") or details.get("license")
                           or spec.get("license") or "unknown"),
        },
        "manifest_download_sha256": spec.get("download_sha256"),
        "raw_frame_sha256": sha256_json({
            "columns": [str(c) for c in frame.columns],
            "dtypes": [str(v) for v in frame.dtypes],
            "row_hashes": pd.util.hash_pandas_object(frame, index=True).astype(str).tolist(),
            "target_hashes": pd.util.hash_pandas_object(y_raw, index=True).astype(str).tolist(),
        }),
    }
    return frame.reset_index(drop=True), y, provenance


def capped_indices(y: np.ndarray, order: int) -> np.ndarray:
    n = len(y)
    if n <= ROW_CAP:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(SEED + 5000 + int(order))
    classes, counts = np.unique(y, return_counts=True)
    exact = counts * (ROW_CAP / float(n))
    take = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - take), kind="stable")[:ROW_CAP - int(take.sum())]:
        take[j] += 1
    selected = []
    for cls, number in zip(classes, take):
        pool = np.flatnonzero(y == cls)
        selected.append(rng.choice(pool, size=int(number), replace=False))
    result = np.sort(np.concatenate(selected)).astype(np.int64)
    if len(result) != ROW_CAP or len(np.unique(result)) != ROW_CAP:
        raise AssertionError("invalid stratified cap sample")
    return result


def stringify_categories(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        out[col] = frame[col].map(lambda v: np.nan if pd.isna(v) else str(v))
    return out


def preprocess(X: pd.DataFrame, y: np.ndarray, source_rows: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          list[str], dict[str, Any], np.ndarray, np.ndarray]:
    local = np.arange(len(y), dtype=np.int64)
    train_local, test_local = train_test_split(
        local, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    Xtr_raw, Xte_raw = X.iloc[train_local].copy(), X.iloc[test_local].copy()
    ytr, yte = y[train_local].copy(), y[test_local].copy()
    columns = [str(c) for c in X.columns]
    categorical = [str(c) for c in X.columns
                   if (isinstance(X[c].dtype, pd.CategoricalDtype)
                       or pd.api.types.is_object_dtype(X[c].dtype)
                       or pd.api.types.is_string_dtype(X[c].dtype)
                       or pd.api.types.is_bool_dtype(X[c].dtype))]
    numeric = [c for c in columns if c not in categorical]
    if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(columns):
        raise AssertionError("preprocessing column partition is invalid")

    train_missing_by_col = {c: int(Xtr_raw[c].isna().sum()) for c in columns}
    test_missing_by_col = {c: int(Xte_raw[c].isna().sum()) for c in columns}
    numeric_fill: dict[str, float] = {}
    categorical_fill: dict[str, str] = {}
    category_maps: dict[str, list[str]] = {}
    encoded_names: list[str] = []
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    if numeric:
        ntr = Xtr_raw[numeric].apply(pd.to_numeric, errors="raise")
        nte = Xte_raw[numeric].apply(pd.to_numeric, errors="raise")
        if any(ntr[c].notna().sum() == 0 for c in numeric):
            raise RuntimeError("numeric predictor is entirely missing in training")
        ni = SimpleImputer(strategy="median")
        train_parts.append(np.asarray(ni.fit_transform(ntr), dtype=np.float64))
        test_parts.append(np.asarray(ni.transform(nte), dtype=np.float64))
        numeric_fill = {c: float(v) for c, v in zip(numeric, ni.statistics_)}
        encoded_names.extend([f"num__{c}" for c in numeric])

    unseen_by_col: dict[str, int] = {}
    if categorical:
        ctr = stringify_categories(Xtr_raw, categorical)
        cte = stringify_categories(Xte_raw, categorical)
        if any(ctr[c].notna().sum() == 0 for c in categorical):
            raise RuntimeError("categorical predictor is entirely missing in training")
        ci = SimpleImputer(strategy="most_frequent")
        ctr_i = ci.fit_transform(ctr)
        cte_i = ci.transform(cte)
        categorical_fill = {c: str(v) for c, v in zip(categorical, ci.statistics_)}
        ohe = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False,
                            dtype=np.float64)
        ctr_o = np.asarray(ohe.fit_transform(ctr_i), dtype=np.float64)
        cte_o = np.asarray(ohe.transform(cte_i), dtype=np.float64)
        train_parts.append(ctr_o)
        test_parts.append(cte_o)
        for j, col in enumerate(categorical):
            levels = [str(v) for v in ohe.categories_[j]]
            category_maps[col] = levels
            encoded_names.extend([f"cat__{col}={v}" for v in levels])
            unseen_by_col[col] = int(sum(str(v) not in set(levels) for v in cte_i[:, j]))

    Xtr = np.column_stack(train_parts).astype(np.float64, copy=False)
    Xte = np.column_stack(test_parts).astype(np.float64, copy=False)
    before = Xtr.shape[1]
    keep = np.ptp(Xtr, axis=0) != 0.0
    removed = [name for name, retain in zip(encoded_names, keep) if not retain]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    names = [name for name, retain in zip(encoded_names, keep) if retain]
    if Xtr.shape[1] == 0 or not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(Xte)):
        raise RuntimeError("processed matrices are empty or non-finite")

    metadata = {
        "adapter_version": "training-only-median-mode-onehot-v1",
        "raw_d": len(columns),
        "encoded_d_before_zero_variance": int(before),
        "encoded_d": int(Xtr.shape[1]),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "numeric_fill_values": numeric_fill,
        "categorical_fill_values": categorical_fill,
        "category_maps": category_maps,
        "zero_variance_columns_removed": removed,
        "train_imputed_cells": int(sum(train_missing_by_col.values())),
        "test_imputed_cells": int(sum(test_missing_by_col.values())),
        "train_imputed_cells_by_column": train_missing_by_col,
        "test_imputed_cells_by_column": test_missing_by_col,
        "unseen_test_levels": int(sum(unseen_by_col.values())),
        "unseen_test_levels_by_column": unseen_by_col,
        "processed_train_sha256": sha256_array(Xtr),
        "processed_test_sha256": sha256_array(Xte),
        "encoded_columns": names,
        "encoded_feature_names": names,
    }
    return (Xtr, Xte, ytr, yte, names, metadata,
            source_rows[train_local], source_rows[test_local])


def validate_source_audit(manifest: dict[str, Any]) -> str:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("PROTOCOL.md differs from the publicly frozen bytes")
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("dataset-manifest.json differs from the publicly frozen bytes")
    if sha256_file(REQUIREMENTS_PATH) != EXPECTED_REQUIREMENTS_SHA256:
        raise RuntimeError("requirements file differs from the pre-outcome environment pin")
    if not SOURCE_AUDIT_PATH.is_file():
        raise RuntimeError(f"missing frozen source audit: {SOURCE_AUDIT_PATH}")
    if sha256_file(SOURCE_AUDIT_PATH) != EXPECTED_SOURCE_AUDIT_SHA256:
        raise RuntimeError("data-source-audit.json differs from the audited public artifact")
    expected = [str(spec["slug"]) for spec in manifest["datasets"]]
    if not SCHEMA_AUDIT_PATH.is_file() \
            or sha256_file(SCHEMA_AUDIT_PATH) != EXPECTED_SCHEMA_AUDIT_SHA256:
        raise RuntimeError("schema-preflight-audit.json differs from its outcome-blind artifact")
    schema_audit = json.loads(SCHEMA_AUDIT_PATH.read_text(encoding="utf-8"))
    if (schema_audit.get("status") != "PASS"
            or schema_audit.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or [item.get("slug") for item in schema_audit.get("datasets", [])] != expected):
        raise RuntimeError("schema preflight does not pass the exact frozen six")
    audit = json.loads(SOURCE_AUDIT_PATH.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(MANIFEST_PATH)
    if audit.get("status") != "PASS" or audit.get("failures") != []:
        raise RuntimeError("frozen source audit does not have terminal PASS status")
    if audit.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("frozen source audit was not run against this manifest")
    entries = audit.get("datasets")
    if not isinstance(entries, list):
        raise TypeError("frozen source audit has no dataset list")
    observed = [str(item.get("slug")) for item in entries]
    if observed != expected or not all(item.get("all_checks_pass") is True for item in entries):
        raise RuntimeError("frozen source audit does not pass the exact six-dataset order")
    return sha256_file(SOURCE_AUDIT_PATH)


def validate_environment() -> None:
    if platform.python_version() != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Python {platform.python_version()} differs from frozen {EXPECTED_PYTHON}"
        )
    observed = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    if observed != EXPECTED_VERSIONS:
        raise RuntimeError(
            f"loaded package versions differ from the frozen environment: {observed}"
        )


def validate_repository_state() -> str:
    root = Path(subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True
    ).strip())
    required = (
        RUNNER_PATH, VERIFIER_PATH, VERIFIER_SCHEMA_PATH, SUMMARY_SCRIPT_PATH,
        SCHEMA_AUDIT_SCRIPT_PATH, IMPLEMENTATION_NOTES_PATH, RUNBOOK_PATH,
        PROTOCOL_PATH, MANIFEST_PATH, REQUIREMENTS_PATH, SOURCE_AUDIT_PATH,
        SCHEMA_AUDIT_PATH,
    )
    for path in required:
        subprocess.check_call(
            ["git", "-C", str(root), "ls-files", "--error-unmatch",
             str(path.relative_to(root))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    tracked_status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain",
         "--untracked-files=no"], text=True
    ).strip()
    if tracked_status:
        raise RuntimeError("public repository has modified or staged tracked files")
    expected_python = {RUNNER_PATH, VERIFIER_PATH, SUMMARY_SCRIPT_PATH,
                       SCHEMA_AUDIT_SCRIPT_PATH, HERE / "audit_data_sources.py"}
    observed_python = set(HERE.glob("*.py"))
    if observed_python != expected_python:
        raise RuntimeError(
            "study directory has an unexpected or missing top-level Python module: "
            f"{sorted(str(path.name) for path in observed_python ^ expected_python)}"
        )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def build_preflight_contract(Xraw: pd.DataFrame, yraw: np.ndarray,
                             selected: np.ndarray, y: np.ndarray,
                             Xtr: np.ndarray, Xte: np.ndarray,
                             ytr: np.ndarray, yte: np.ndarray,
                             prep: dict[str, Any], train_idx: np.ndarray,
                             test_idx: np.ndarray,
                             provenance: dict[str, Any]) -> dict[str, Any]:
    duplicate_frame = Xraw.copy()
    duplicate_frame["__target__"] = yraw
    return {
        "raw_frame_sha256": provenance["raw_frame_sha256"],
        "source_rows": len(yraw),
        "raw_d": Xraw.shape[1],
        "source_adverse_count": int(yraw.sum()),
        "predictor_names": [str(value) for value in Xraw.columns],
        "predictor_dtypes": {str(name): str(dtype)
                             for name, dtype in Xraw.dtypes.items()},
        "missing_cells_by_predictor": {
            str(name): int(value) for name, value in Xraw.isna().sum().items()
        },
        "duplicate_full_rows_including_target": int(duplicate_frame.duplicated().sum()),
        "retained_source_indices_sha256": sha256_array(selected),
        "processed_train_sha256": prep["processed_train_sha256"],
        "processed_test_sha256": prep["processed_test_sha256"],
        "train_indices_sha256": sha256_array(train_idx),
        "test_indices_sha256": sha256_array(test_idx),
        "n_retained": len(y),
        "n_train": len(ytr),
        "n_test": len(yte),
        "retained_adverse_count": int(y.sum()),
        "train_adverse_count": int(ytr.sum()),
        "test_adverse_count": int(yte.sum()),
        "encoded_d": int(Xtr.shape[1]),
        "curve_length_L": int(Xte.shape[1] + 1),
    }


def frozen_schema_contract(entry: dict[str, Any]) -> dict[str, Any]:
    return {
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


def preflight_all_sources(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load and preprocess all six snapshots before any SAFE outcome is computed."""
    specs = manifest.get("datasets")
    if not isinstance(specs, list) or len(specs) != 6:
        raise RuntimeError("manifest must contain exactly six intended datasets")
    schema_audit = json.loads(SCHEMA_AUDIT_PATH.read_text(encoding="utf-8"))
    schema_by_slug = {str(entry["slug"]): entry for entry in schema_audit["datasets"]}
    summaries: dict[str, dict[str, Any]] = {}
    statuses: list[dict[str, Any]] = []
    for spec in specs:
        slug = str(spec["slug"])
        print(f"[preflight] {slug}", flush=True)
        try:
            Xraw, yraw, provenance = load_raw(spec)
            selected = capped_indices(yraw, int(spec["order"]))
            X = Xraw.iloc[selected].reset_index(drop=True)
            y = yraw[selected]
            Xtr, Xte, ytr, yte, _names, prep, train_idx, test_idx = preprocess(
                X, y, selected
            )
            summaries[slug] = build_preflight_contract(
                Xraw, yraw, selected, y, Xtr, Xte, ytr, yte, prep,
                train_idx, test_idx, provenance
            )
            expected = frozen_schema_contract(schema_by_slug[slug])
            if summaries[slug] != expected:
                differing = sorted(
                    key for key in expected if summaries[slug].get(key) != expected[key]
                )
                raise RuntimeError(
                    f"{slug}: runtime preflight differs from frozen schema fields {differing}"
                )
            statuses.append({"slug": slug, "status": "passed",
                             "contract": summaries[slug]})
        except Exception as exc:  # noqa: BLE001 -- audit every intended source
            statuses.append({
                "slug": slug,
                "status": "failed",
                "failure": {"type": type(exc).__name__, "message": str(exc),
                            "traceback": traceback.format_exc()},
            })
    failures = [item for item in statuses if item["status"] == "failed"]
    if failures:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failure_path = HERE / f"preflight-failure-{timestamp}.json"
        atomic_json(failure_path, {
            "schema_version": "six-dataset-preflight-v1",
            "status": "failed",
            "generated_utc": utc_now(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "dataset_spec_sha256": sha256_file(MANIFEST_PATH),
            "datasets": statuses,
            "requires_protocol_deviation_before_retry": True,
        })
        raise RuntimeError(
            f"all-six preflight failed for {len(failures)} dataset(s); retained {failure_path}"
        )
    return summaries


def make_model(tag: str):
    if tag == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED))
    if tag == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)
    raise ValueError(tag)


def rga_vector(y: np.ndarray, p: np.ndarray, d: int) -> np.ndarray:
    result = RGA_CURVE(y, p, curve_method="partial", n_segments=d,
                       normalize_to_perfect=False)
    curve = np.asarray(result["curve"], dtype=np.float64)
    if curve.shape != (d + 1,):
        raise RuntimeError(f"RGA returned shape {curve.shape}, expected {(d + 1,)}")
    return curve


def rge_vector(model, X: np.ndarray, names: list[str], d: int
               ) -> tuple[np.ndarray, list[int]]:
    result = RGE_CURVE(model, X, method="tabular", feature_names=names,
                       masking_method="greedy", baseline="mean", n_steps=d,
                       verbose=False)
    curve = np.asarray(result["rge_scores"], dtype=np.float64)
    order_names = list(result["removed_features"])
    if curve.shape != (d + 1,) or len(order_names) != d:
        raise RuntimeError("RGE curve/order length mismatch")
    if len(set(order_names)) != d or set(order_names) != set(names):
        raise RuntimeError("RGE removal order is not a permutation")
    return curve, [names.index(name) for name in order_names]


def tail_swap(X: np.ndarray, proportion: float) -> np.ndarray:
    n = X.shape[0]
    m = int(np.floor(proportion * n))
    if m == 0:
        return X.copy()
    out = X.copy()
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        source = X[:, j]
        column = source.copy()
        lo, hi = order[:m], order[n - m:][::-1]
        column[lo], column[hi] = source[hi], source[lo]
        out[:, j] = column
    return out


def point_and_frozen(model, X: np.ndarray, y: np.ndarray, names: list[str]
                     ) -> dict[str, Any]:
    n, d = X.shape
    L = d + 1
    p_full = model.predict_proba(X)[:, 1]
    a0 = rga_vector(y, p_full, d)
    e0, order = rge_vector(model, X, names, d)
    means = X.mean(axis=0)
    p_mask = np.empty((L, n), dtype=np.float64)
    p_mask[0] = p_full
    for t in range(1, L):
        Xm = X.copy()
        cols = order[:t]
        Xm[:, cols] = means[cols]
        p_mask[t] = model.predict_proba(Xm)[:, 1]
    severity = np.arange(L, dtype=np.float64) / float(d)
    tail_p = 0.5 * severity
    p_tail = np.empty((L, n), dtype=np.float64)
    for t, p in enumerate(tail_p):
        p_tail[t] = model.predict_proba(tail_swap(X, float(p)))[:, 1]
    std = X.std(axis=0, keepdims=True)
    rng = np.random.default_rng(SEED)
    p_gauss = np.empty((L, n), dtype=np.float64)
    for t, sigma in enumerate(severity):
        Xp = X if sigma == 0.0 else X + rng.normal(0.0, float(sigma), X.shape) * std
        p_gauss[t] = model.predict_proba(Xp)[:, 1]
    r0_tail = np.asarray([RGR_SCORE(p_full, row, class_order=CLASS_ORDER)
                          for row in p_tail], dtype=np.float64)
    r0_gauss = np.asarray([RGR_SCORE(p_full, row, class_order=CLASS_ORDER)
                           for row in p_gauss], dtype=np.float64)
    e_fast = np.r_[1.0, [RGE_SCORE(p_full, p_mask[t], class_order=CLASS_ORDER)
                         for t in range(1, L)]]
    gap = float(np.max(np.abs(e_fast - e0)))
    if gap > 5e-10:
        raise RuntimeError(f"frozen RGE fast path disagrees with package curve: {gap}")
    arrays = (p_full, p_mask, p_tail, p_gauss, a0, e0, r0_tail, r0_gauss)
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise RuntimeError("point/precomputed predictions contain non-finite values")
    return {"p_full": p_full, "p_mask": p_mask, "p_tail": p_tail,
            "p_gaussian": p_gauss, "a0": a0, "e0": e0,
            "r0_tail": r0_tail, "r0_gaussian": r0_gauss,
            "order": order, "severity": severity, "tail_p": tail_p,
            "fast_path_gap": gap}


def bootstrap_curves(y: np.ndarray, frozen: dict[str, Any], indices: np.ndarray,
                     log: RunLog, label: str) -> tuple[np.ndarray, ...]:
    B, _n = indices.shape
    d = frozen["p_mask"].shape[0] - 1
    L = d + 1
    A = np.empty((B, L)); E = np.empty((B, L))
    Rt = np.empty((B, L)); Rg = np.empty((B, L))
    for b, idx in enumerate(indices):
        p = frozen["p_full"][idx]
        A[b] = rga_vector(y[idx], p, d)
        E[b, 0] = 1.0
        for t in range(1, L):
            E[b, t] = RGE_SCORE(p, frozen["p_mask"][t, idx],
                                class_order=CLASS_ORDER)
        for t in range(L):
            Rt[b, t] = RGR_SCORE(p, frozen["p_tail"][t, idx],
                                 class_order=CLASS_ORDER)
            Rg[b, t] = RGR_SCORE(p, frozen["p_gaussian"][t, idx],
                                 class_order=CLASS_ORDER)
        if (b + 1) % 250 == 0 or b + 1 == B:
            log.say(f"[{label}] paired bootstrap {b + 1}/{B}")
    if not all(np.all(np.isfinite(v)) for v in (A, E, Rt, Rg)):
        raise RuntimeError("bootstrap matrices contain non-finite values")
    return A, E, Rt, Rg


def manual_corr(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(xc @ xc) * float(yc @ yc))
    if denominator <= 0.0:
        raise RuntimeError("correlation is undefined for a constant summary")
    value = float((xc @ yc) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def primary_contrast(E: np.ndarray, Rt: np.ndarray, Rg: np.ndarray) -> dict[str, Any]:
    em, tm, gm = E.mean(1), Rt.mean(1), Rg.mean(1)
    ct, cg = manual_corr(em, tm), manual_corr(em, gm)
    clip = lambda r: float(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12))
    delta = float(np.arctanh(clip(ct)) - np.arctanh(clip(cg)))
    return {"corr_RGE_RGR_tail": ct, "corr_RGE_RGR_gaussian": cg,
            "fisher_z_tail_minus_gaussian": delta,
            "prespecified_direction_delta_lt_zero": bool(delta < 0.0)}


def percentile_block(values: np.ndarray) -> dict[str, Any]:
    lo, hi = np.percentile(values, [QLO, QHI])
    return {"ci95_percentile": [float(lo), float(hi)], "width": float(hi - lo),
            "mean": float(values.mean()), "sd": float(values.std(ddof=1))}


def v_arithmetic(A, E, R):
    return (A.mean(1) + E.mean(1) + R.mean(1)) / 3.0


def v_geometric(A, E, R):
    return np.cbrt(A).mean(1) * np.cbrt(E).mean(1) * np.cbrt(R).mean(1)


def v_rms(A, E, R):
    L = A.shape[1]
    chunk = max(1, min(64, int(64_000_000 / max(1, 4 * 8 * L ** 3))))
    out = np.empty(A.shape[0])
    for start in range(0, len(out), chunk):
        a = A[start:start + chunk, :, None, None]
        e = E[start:start + chunk, None, :, None]
        r = R[start:start + chunk, None, None, :]
        out[start:start + chunk] = np.sqrt((a*a + e*e + r*r) / 3.0).mean((1, 2, 3))
    return out


def topsis_reference(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pis = np.concatenate([np.ones(L), np.r_[1.0, np.zeros(L - 1)], np.ones(L)])
    nis = np.concatenate([np.zeros(L), np.linspace(1.0, 0.5, L),
                          np.r_[1.0, np.zeros(L - 1)]])
    weights = np.full(3 * L, 1.0 / 3.0)
    return pis, nis, weights


def v_topsis(A, E, R):
    L = A.shape[1]
    pis, nis, weights = topsis_reference(L)
    V = np.concatenate([A, E, R], axis=1)
    sp = np.sqrt((((V - pis) * weights) ** 2).sum(1))
    sm = np.sqrt((((V - nis) * weights) ** 2).sum(1))
    return sm / (sm + sp)


COMPOSITES: dict[str, Callable] = {
    "arithmetic": v_arithmetic, "geometric": v_geometric,
    "rms": v_rms, "topsis": v_topsis,
}


def grad_rms(a, e, r):
    L = len(a)
    M = np.sqrt((a[:, None, None]**2 + e[None, :, None]**2
                 + r[None, None, :]**2) / 3.0)
    if np.any(M == 0):
        raise RuntimeError("RMS gradient undefined at zero tensor cell")
    inv, scale = 1.0 / M, 1.0 / (3.0 * L**3)
    return (a * inv.sum((1, 2)) * scale,
            e * inv.sum((0, 2)) * scale,
            r * inv.sum((0, 1)) * scale)


def grad_topsis(a, e, r):
    L = len(a)
    pis, nis, weights = topsis_reference(L)
    v, w2 = np.concatenate([a, e, r]), weights**2
    sp = math.sqrt(float(((v - pis)**2 * w2).sum()))
    sm = math.sqrt(float(((v - nis)**2 * w2).sum()))
    if sp == 0 or sm == 0:
        raise RuntimeError("TOPSIS gradient undefined at a reference point")
    g = (sp * (w2 * (v - nis) / sm) - sm * (w2 * (v - pis) / sp)) / (sm + sp)**2
    return g[:L], g[L:2*L], g[2*L:]


def delta_inputs(kind, A, E, R, a, e, r):
    if kind == "arithmetic":
        S = np.column_stack([A.mean(1), E.mean(1), R.mean(1)])
        s0 = np.array([a.mean(), e.mean(), r.mean()])
        g = np.full(3, 1.0 / 3.0)
    elif kind == "geometric":
        S = np.column_stack([np.cbrt(A).mean(1), np.cbrt(E).mean(1),
                             np.cbrt(R).mean(1)])
        s0 = np.array([np.cbrt(a).mean(), np.cbrt(e).mean(), np.cbrt(r).mean()])
        g = np.array([s0[1]*s0[2], s0[0]*s0[2], s0[0]*s0[1]])
    else:
        ga, ge, gr = grad_rms(a, e, r) if kind == "rms" else grad_topsis(a, e, r)
        S = np.column_stack([A @ ga, E @ ge, R @ gr])
        s0 = np.array([a @ ga, e @ ge, r @ gr])
        g = np.ones(3)
    return S, s0, g


def composite_block(kind, A, E, R, a, e, r) -> dict[str, Any]:
    fn = COMPOSITES[kind]
    values = np.asarray(fn(A, E, R), dtype=np.float64)
    point = float(fn(a[None], e[None], r[None])[0])
    S, s0, g = delta_inputs(kind, A, E, R, a, e, r)
    Sigma = np.atleast_2d(np.cov(S, rowvar=False, ddof=1))
    Corr = np.atleast_2d(np.corrcoef(S, rowvar=False))
    measured = float(g @ Sigma @ g)
    zero = float(np.sum(g*g*np.diag(Sigma)))
    if min(measured, zero) < 0:
        raise RuntimeError("negative delta variance")
    wm, wz = 2 * Z * math.sqrt(measured), 2 * Z * math.sqrt(zero)
    return {"V_point": point, "paired_bootstrap": percentile_block(values),
            "delta_method": {
                "summary_point": s0.tolist(), "gradient_at_point": g.tolist(),
                "Sigma_of_summaries": Sigma.tolist(), "Corr_of_summaries": Corr.tolist(),
                "var_measured_covariance": measured,
                "var_cross_terms_declared_zero": zero,
                "width_delta_measured": wm, "width_delta_declared_zero": wz,
                "understatement_of_width_pct": (100.0 * (1.0 - wz / wm)
                                                  if wm > 0 else 0.0),
                "signed_width_change_pct_zero_vs_measured": (100.0 * (1.0 - wz / wm)
                                                               if wm > 0 else 0.0),
            }}


def summarize(A, E, Rt, Rg, frozen) -> dict[str, Any]:
    a, e = frozen["a0"], frozen["e0"]
    rt, rg = frozen["r0_tail"], frozen["r0_gaussian"]
    primary = primary_contrast(E, Rt, Rg)
    arms = {kind: composite_block(kind, A, E, Rt, a, e, rt) for kind in COMPOSITES}
    marginal = {}
    for name, matrix in (("RGA_partial", A), ("RGE_greedy_mean_baseline", E),
                         ("RGR_tail_swap", Rt), ("RGR_scaled_gaussian_noise", Rg)):
        q = np.percentile(matrix, [QLO, QHI], axis=0)
        marginal[name] = {"ci95_percentile_by_knot": np.column_stack(q).tolist(),
                          "bootstrap_sd_by_knot": matrix.std(0, ddof=1).tolist()}
    means = np.column_stack([A.mean(1), E.mean(1), Rt.mean(1), Rg.mean(1)])
    diagonal = {}
    diagonal_fns = {
        "arithmetic": lambda A,E,R: ((A+E+R)/3).mean(1),
        "geometric": lambda A,E,R: np.cbrt(A*E*R).mean(1),
        "rms": lambda A,E,R: np.sqrt((A*A+E*E+R*R)/3).mean(1),
    }
    for kind, fn in diagonal_fns.items():
        vd = fn(A, E, Rt)
        vp = COMPOSITES[kind](A, E, Rt)
        pd = float(fn(a[None], e[None], rt[None])[0])
        pp = float(COMPOSITES[kind](a[None], e[None], rt[None])[0])
        diagonal[kind] = {"V_point": pd, "product_space_V_point": pp,
                          "diagonal_minus_product_space": pd-pp,
                          "paired_bootstrap": percentile_block(vd),
                          "difference_paired_bootstrap": percentile_block(vd-vp)}
    all_corr = np.corrcoef(means, rowvar=False)
    dropped = np.column_stack([A[:, :-1].mean(1), E[:, 1:].mean(1),
                               Rt[:, 1:].mean(1)])
    base = np.column_stack([A.mean(1), E.mean(1), Rt.mean(1)])
    return {
        "curves_point": {"RGA_partial": a.tolist(),
                         "RGE_greedy_mean_baseline": e.tolist(),
                         "RGR_tail_swap": rt.tolist(),
                         "RGR_scaled_gaussian_noise": rg.tolist()},
        "marginal_intervals": marginal,
        "curve_mean_correlations": {
            "labels": ["RGA", "RGE", "RGR_tail", "RGR_gaussian"],
            "matrix": all_corr.tolist()},
        "primary_family_contrast": primary,
        "primary_endpoint": primary,
        "arms": arms,
        "composites_tail": arms,
        "matched_severity_diagonal": diagonal,
        "deterministic_knot_check": {
            "replicate_sd": {"RGA_last": float(A[:, -1].std()),
                             "RGE_first": float(E[:, 0].std()),
                             "RGR_tail_first": float(Rt[:, 0].std()),
                             "RGR_gaussian_first": float(Rg[:, 0].std())},
            "max_abs_correlation_change_after_dropping_deterministic_knots":
                float(np.max(np.abs(np.corrcoef(base, rowvar=False)
                                    - np.corrcoef(dropped, rowvar=False)))),
        },
    }


def conditioning_check(model, X, y, names, indices, frozen, A, E, Rt, Rg,
                       requested: int, log: RunLog, label: str
                       ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    Bf = min(int(requested), len(indices))
    d, L = X.shape[1], X.shape[1] + 1
    Af, Ef, Rf = np.empty((Bf,L)), np.empty((Bf,L)), np.empty((Bf,L))
    orders = []
    for b in range(Bf):
        idx = indices[b]
        Xb, yb = X[idx], y[idx]
        pb = model.predict_proba(Xb)[:, 1]
        Af[b] = rga_vector(yb, pb, d)
        Ef[b], order = rge_vector(model, Xb, names, d)
        orders.append([names[j] for j in order])
        for t, p in enumerate(frozen["tail_p"]):
            Rf[b,t] = RGR_SCORE(pb, model.predict_proba(tail_swap(Xb,float(p)))[:,1],
                                class_order=CLASS_ORDER)
        if (b+1) % 10 == 0 or b+1 == Bf:
            log.say(f"[{label}] full conditioning {b+1}/{Bf}")
    result = {"B_requested": int(requested), "B_completed": Bf, "B_full": Bf,
              "greedy_orders": orders, "composites_tail": {}}
    for kind, fn in COMPOSITES.items():
        vf, vv = fn(A[:Bf],E[:Bf],Rt[:Bf]), fn(Af,Ef,Rf)
        sf, sv = float(vf.std(ddof=1)), float(vv.std(ddof=1))
        result["composites_tail"][kind] = {
            "sd_frozen": sf, "sd_full": sv,
            "sd_ratio_full_over_frozen": sv/sf if sf > 0 else None,
            "mean_abs_paired_difference": float(np.mean(np.abs(vv-vf))),
            "max_abs_paired_difference": float(np.max(np.abs(vv-vf)))}
    if Bf >= 3:
        try:
            result["primary_endpoint_frozen"] = primary_contrast(E[:Bf],Rt[:Bf],Rg[:Bf])
            result["primary_endpoint_full_order_tail"] = primary_contrast(Ef,Rf,Rg[:Bf])
        except RuntimeError as exc:
            result["primary_endpoint_note"] = str(exc)
    return result, {"conditioning_Ab": Af, "conditioning_Eb": Ef,
                    "conditioning_Rb_tail": Rf}


def fit_with_warnings(model, X, y) -> list[dict[str, str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    return [{"category": item.category.__name__, "message": str(item.message)}
            for item in caught]


def log_fit_warnings(log: RunLog, label: str,
                     records: list[dict[str, str]]) -> None:
    if not records:
        log.say(f"[{label}] fit warnings: none")
        return
    for record in records:
        log.say(f"[{label}] fit warning {record['category']}: {record['message']}")


def run_model(tag: str, Xtr, Xte, ytr, yte, names, B: int, conditioning: int,
              log: RunLog) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n = len(yte)
    rng = np.random.default_rng(BOOT_SEED)  # reinitialised separately per model
    indices = rng.integers(0, n, size=(B, n), dtype=np.int64)
    model = make_model(tag)
    fit_warnings = fit_with_warnings(model, Xtr, ytr)
    log_fit_warnings(log, tag, fit_warnings)
    frozen = point_and_frozen(model, Xte, yte, names)
    A,E,Rt,Rg = bootstrap_curves(yte, frozen, indices, log, tag)
    result = summarize(A,E,Rt,Rg,frozen)
    result["fit_warnings"] = fit_warnings
    result["RGE_greedy_removal_order"] = [names[j] for j in frozen["order"]]
    result["fast_path_vs_package_max_abs_gap"] = frozen["fast_path_gap"]
    cond, cond_arrays = conditioning_check(model,Xte,yte,names,indices,frozen,A,E,Rt,Rg,
                                            conditioning,log,tag)
    result["conditioning_check"] = cond

    flip_model = make_model(tag)
    flip_warnings = fit_with_warnings(flip_model, Xtr, 1-ytr)
    log_fit_warnings(log, tag + "/flip", flip_warnings)
    flip_frozen = point_and_frozen(flip_model, Xte, 1-yte, names)
    Af,Ef,Rtf,Rgf = bootstrap_curves(1-yte, flip_frozen, indices, log,tag+"/flip")
    flip = summarize(Af,Ef,Rtf,Rgf,flip_frozen)
    flip["fit_warnings"] = flip_warnings
    flip["RGE_greedy_removal_order"] = [names[j] for j in flip_frozen["order"]]
    result["label_complement"] = flip

    arrays = {"Ab":A,"Eb":E,"Rb_tail":Rt,"Rb_gaussian":Rg,
              "a0":frozen["a0"],"e0":frozen["e0"],
              "r0_tail":frozen["r0_tail"],"r0_gaussian":frozen["r0_gaussian"],
              "flip_Ab":Af,"flip_Eb":Ef,"flip_Rb_tail":Rtf,"flip_Rb_gaussian":Rgf,
              "flip_a0":flip_frozen["a0"],"flip_e0":flip_frozen["e0"],
              "flip_r0_tail":flip_frozen["r0_tail"],
              "flip_r0_gaussian":flip_frozen["r0_gaussian"], **cond_arrays}
    return result, arrays


def output_suffix(raw: str) -> str:
    if not raw:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", raw):
        raise ValueError("output suffix may contain only letters, numbers, dot, dash, underscore")
    return raw if raw.startswith(("-","_")) else "-"+raw


def output_paths(spec: dict[str, Any], raw_suffix: str) -> tuple[Path, Path, Path]:
    suffix = output_suffix(raw_suffix)
    slug = str(spec["slug"])
    return (HERE / f"results-{slug}{suffix}.json",
            HERE / f"replicates-{slug}{suffix}.npz",
            HERE / f"run-{slug}{suffix}.log")


def assert_no_output_collisions(specs: list[dict[str, Any]], raw_suffix: str) -> None:
    existing: list[Path] = []
    for spec in specs:
        result_path, npz_path, log_path = output_paths(spec, raw_suffix)
        candidates = (result_path, npz_path, log_path,
                      result_path.with_name(result_path.name + ".tmp"),
                      npz_path.with_name(npz_path.stem + ".tmp.npz"))
        existing.extend(path for path in candidates if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to start because an intended output already exists: "
            + ", ".join(map(str, existing))
        )


def run_dataset(spec: dict[str, Any], args, safeai_commit: str,
                source_audit_sha256: str,
                preflight: dict[str, Any], repository_commit: str) -> bool:
    suffix = output_suffix(args.output_suffix)
    slug = spec["slug"]
    result_path, npz_path, log_path = output_paths(spec, suffix)
    existing = [p for p in (result_path,npz_path,log_path) if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite first output: " + ", ".join(map(str,existing)))
    log = RunLog(log_path)
    started = time.time()
    base = {
        "schema_version":"six-dataset-runner-v1", "status":"running", "slug":slug,
        "dataset":{"slug":slug, "name":spec["name"]},
        "generated_utc":utc_now(), "protocol_sha256":sha256_file(PROTOCOL_PATH),
        "dataset_spec_sha256":sha256_file(MANIFEST_PATH),
        "manifest_entry_sha256":sha256_json(spec),
        "manifest_sha256":sha256_file(MANIFEST_PATH),
        "runner_sha256":sha256_file(RUNNER_PATH),
        "requirements_sha256":sha256_file(REQUIREMENTS_PATH),
        "source_audit_sha256":source_audit_sha256,
        "schema_audit_sha256":sha256_file(SCHEMA_AUDIT_PATH),
        "repository_commit":repository_commit,
        "B":args.B, "alpha":ALPHA,
        "B_conditioning_check":{"logit":args.conditioning_logit,
                                "rf":args.conditioning_rf},
        "config":{"seed":SEED,"bootstrap_seed":BOOT_SEED,"test_size":TEST_SIZE,
                  "row_cap":ROW_CAP,"B":args.B,"alpha":ALPHA,"z":Z,
                  "B_conditioning_check":{"logit":args.conditioning_logit,
                                            "rf":args.conditioning_rf},
                  "conditioning":{"logit":args.conditioning_logit,
                                  "rf":args.conditioning_rf}},
        "environment":{"python":sys.version.split()[0],"platform":platform.platform(),
                       "numpy":np.__version__,"pandas":pd.__version__,
                       "scipy":scipy.__version__,"scikit_learn":sklearn.__version__,
                       "joblib":joblib.__version__,
                       "threadpoolctl":threadpoolctl.__version__,
                       "threadpools":threadpoolctl.threadpool_info(),
                       "safeai_commit":safeai_commit},
        "dataset_manifest_entry":spec,
    }
    try:
        log.say(f"[start] {slug} at {base['generated_utc']}")
        Xraw,yraw,provenance = load_raw(spec)
        base["dataset_provenance"] = provenance
        base["provenance"] = provenance
        duplicate_frame = Xraw.copy()
        duplicate_frame["__target__"] = yraw
        base["raw_duplicate_full_rows"] = int(duplicate_frame.duplicated().sum())
        selected = capped_indices(yraw,int(spec["order"]))
        X,y = Xraw.iloc[selected].reset_index(drop=True), yraw[selected]
        Xtr,Xte,ytr,yte,names,prep,tr_idx,te_idx = preprocess(X,y,selected)
        d,L = Xtr.shape[1],Xtr.shape[1]+1
        observed_preflight = build_preflight_contract(
            Xraw, yraw, selected, y, Xtr, Xte, ytr, yte, prep,
            tr_idx, te_idx, provenance
        )
        if observed_preflight != preflight:
            raise RuntimeError("dataset/preprocessing drifted after the all-six preflight")
        base["all_six_preflight_match"] = observed_preflight
        base.update({"d":d,"curve_length_L":L,"n_retained":len(y),
                     "n_train":len(ytr),"n_test":len(yte),
                     "adverse_count_retained":int(y.sum()),
                     "adverse_count_train":int(ytr.sum()),
                     "adverse_count_test":int(yte.sum()),
                     "sampling":{"applied":len(yraw)>ROW_CAP,
                                 "rng_seed":SEED+5000+int(spec["order"]),
                                 "retained_source_indices_sha256":sha256_array(selected)},
                     "preprocessing":prep})
        base["config"].update({"d":d, "encoded_d":d, "curve_length_L":L})
        log.say(f"[data] retained={len(y)} train={len(ytr)} test={len(yte)} raw_d={Xraw.shape[1]} encoded_d={d}")
        models, deposit = {}, {"severity_t_over_d":np.arange(L)/float(d),
                               "tail_p":0.5*np.arange(L)/float(d)}
        base["models"] = models
        for tag in MODEL_TAGS:
            log.say(f"[model] {tag}")
            models[tag], arrays = run_model(tag,Xtr,Xte,ytr,yte,names,args.B,
                                            args.conditioning_logit if tag=="logit"
                                            else args.conditioning_rf,log)
            for key,value in arrays.items():
                deposit[f"{tag}_{key}"] = value
        base["status"] = "completed"
        base["completed_utc"] = utc_now()
        base["runtime_seconds"] = time.time()-started
        as_json(base)  # reject non-finite secondary as well as primary results
        tmp_npz = npz_path.with_name(npz_path.stem+".tmp.npz")
        np.savez_compressed(tmp_npz,**deposit)
        os.replace(tmp_npz,npz_path)
        base["replicates_file"] = npz_path.name
        base["replicates_sha256"] = sha256_file(npz_path)
        atomic_json(result_path,base)
        log.say(f"[done] {slug} in {base['runtime_seconds']:.1f}s")
        return True
    except Exception as exc:
        if "models" in base:
            partial_models = base.pop("models")
            try:
                base["partial_models"] = as_json(partial_models)
            except (TypeError, ValueError):
                base["partial_model_results_omitted"] = True
                base["partial_model_fit_warnings"] = {
                    tag: {
                        "primary": value.get("fit_warnings", []),
                        "label_complement": value.get("label_complement", {}).get(
                            "fit_warnings", []
                        ),
                    }
                    for tag, value in partial_models.items()
                    if isinstance(value, dict)
                }
        base["status"]="failed"
        base["completed_utc"]=utc_now()
        base["runtime_seconds"]=time.time()-started
        base["failure"]={"type":type(exc).__name__,"message":str(exc),
                         "traceback":traceback.format_exc()}
        try:
            atomic_json(result_path,base)
        except Exception as record_exc:  # noqa: BLE001 -- preserve a minimal terminal record
            minimal = {
                "schema_version": "six-dataset-runner-v1",
                "status": "failed",
                "slug": slug,
                "dataset": {"slug": slug, "name": spec["name"]},
                "generated_utc": base["generated_utc"],
                "completed_utc": utc_now(),
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "dataset_spec_sha256": sha256_file(MANIFEST_PATH),
                "failure": base["failure"],
                "failure_record_reduced_because": {
                    "type": type(record_exc).__name__, "message": str(record_exc)
                },
            }
            atomic_json(result_path, minimal)
        log.say(f"[failed] {type(exc).__name__}: {exc}")
        log.say(traceback.format_exc().rstrip())
        if isinstance(exc, RunInterrupted):
            raise
        return False
    finally:
        log.close()


def parse_args(argv=None):
    manifest=json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    slugs=[d["slug"] for d in manifest["datasets"]]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["all",*slugs],default="all")
    parser.add_argument("--B",type=int,default=2000)
    parser.add_argument("--conditioning-logit",type=int,default=100)
    parser.add_argument("--conditioning-rf",type=int,default=30)
    parser.add_argument("--output-suffix",default="")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate all six sources and preprocessing without SAFE metrics")
    args=parser.parse_args(argv)
    if args.B<2 or args.conditioning_logit<2 or args.conditioning_rf<2:
        parser.error("B and conditioning counts must be at least 2")
    if (args.B!=2000 or args.conditioning_logit!=100 or args.conditioning_rf!=30) \
            and not args.output_suffix:
        parser.error("non-frozen counts require --output-suffix so primary outputs cannot be overwritten")
    output_suffix(args.output_suffix)
    return args,manifest


def write_not_run_after_interruption(spec: dict[str, Any], args,
                                     repository_commit: str,
                                     reason: str) -> None:
    result_path, _npz_path, log_path = output_paths(spec, args.output_suffix)
    slug = str(spec["slug"])
    log = RunLog(log_path)
    try:
        log.say(f"[not-run] {slug}: {reason}")
        atomic_json(result_path, {
            "schema_version": "six-dataset-runner-v1",
            "status": "not_run_due_to_interruption",
            "slug": slug,
            "dataset": {"slug": slug, "name": spec["name"]},
            "generated_utc": utc_now(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "dataset_spec_sha256": sha256_file(MANIFEST_PATH),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "repository_commit": repository_commit,
            "failure": {"type": "RunInterrupted", "message": reason},
        })
    finally:
        log.close()


def install_interrupt_handlers():
    previous = {}

    def handler(signum, _frame):
        name = signal.Signals(signum).name
        raise RunInterrupted(f"received {name}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, handler)
    return previous


def restore_interrupt_handlers(previous) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def write_startup_failure(manifest: dict[str, Any], stage: str,
                          exc: Exception) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = HERE / f"startup-failure-{timestamp}.json"
    atomic_json(path, {
        "schema_version": "six-dataset-startup-v1",
        "status": "failed_before_metrics",
        "stage": stage,
        "generated_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "dataset_spec_sha256": sha256_file(MANIFEST_PATH),
        "failure": {"type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc()},
        "datasets": [
            {"slug": spec["slug"], "status": "not_run_before_metrics"}
            for spec in manifest["datasets"]
        ],
        "requires_protocol_deviation_before_retry": True,
    })
    return path


def main(argv=None) -> int:
    args,manifest=parse_args(argv)
    specs=manifest["datasets"] if args.dataset=="all" else [
        next(d for d in manifest["datasets"] if d["slug"]==args.dataset)]
    stage = "output-collision-check"
    try:
        if not args.preflight_only:
            assert_no_output_collisions(specs, args.output_suffix)
        stage = "frozen-audit-validation"
        source_audit_sha256 = validate_source_audit(manifest)
        stage = "environment-validation"
        validate_environment()
        stage = "all-six-data-preflight"
        preflight = preflight_all_sources(manifest)
        if args.preflight_only:
            print("[preflight] PASS: exact six-dataset source and preprocessing contract",
                  flush=True)
            return 0
        stage = "repository-validation"
        repository_commit = validate_repository_state()
        stage = "safeai-substrate"
        safeai_commit=ensure_substrate()
    except Exception as exc:  # noqa: BLE001 -- no outcomes exist at this boundary
        artifact = None
        if stage not in {"output-collision-check", "all-six-data-preflight"}:
            artifact = write_startup_failure(manifest, stage, exc)
        retained = f"; retained {artifact}" if artifact is not None else ""
        print(f"[preflight-failed] {type(exc).__name__}: {exc}{retained}",
              file=sys.stderr, flush=True)
        return 2
    failures=[]
    previous_handlers = install_interrupt_handlers()
    try:
        for index, spec in enumerate(specs):
            try:
                with threadpoolctl.threadpool_limits(limits=1):
                    passed = run_dataset(
                        spec, args, safeai_commit, source_audit_sha256,
                        preflight[str(spec["slug"])], repository_commit
                    )
                if not passed:
                    failures.append(spec["slug"])
            except RunInterrupted as exc:
                failures.append(spec["slug"])
                for remaining in specs[index + 1:]:
                    write_not_run_after_interruption(
                        remaining, args, repository_commit, str(exc)
                    )
                    failures.append(remaining["slug"])
                print(f"[interrupted] {exc}", file=sys.stderr, flush=True)
                return 130
            except Exception as exc:  # noqa: BLE001 -- continue frozen intended set
                print(f"[failed-before-log] {spec['slug']}: {exc}",
                      file=sys.stderr,flush=True)
                failures.append(spec["slug"])
    finally:
        restore_interrupt_handlers(previous_handlers)
    if failures:
        print("FAILED datasets: "+", ".join(failures),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
