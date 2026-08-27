#!/usr/bin/env python3
"""Deposit outcome-blind schema and preprocessing facts for the frozen six."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import run_six_dataset as runner

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "schema-preflight-audit.json"


def main() -> int:
    runner.validate_environment()
    manifest = json.loads(runner.MANIFEST_PATH.read_text(encoding="utf-8"))
    if runner.sha256_file(runner.PROTOCOL_PATH) != runner.EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("protocol digest differs from the public freeze")
    if runner.sha256_file(runner.MANIFEST_PATH) != runner.EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest digest differs from the public freeze")
    source_audit = json.loads(runner.SOURCE_AUDIT_PATH.read_text(encoding="utf-8"))
    if (runner.sha256_file(runner.SOURCE_AUDIT_PATH)
            != runner.EXPECTED_SOURCE_AUDIT_SHA256
            or source_audit.get("status") != "PASS"):
        raise RuntimeError("source-byte audit is not the frozen PASS artifact")
    datasets = []
    for spec in manifest["datasets"]:
        slug = str(spec["slug"])
        print(f"[schema-audit] {slug}", flush=True)
        Xraw, yraw, provenance = runner.load_raw(spec)
        selected = runner.capped_indices(yraw, int(spec["order"]))
        X = Xraw.iloc[selected].reset_index(drop=True)
        y = yraw[selected]
        Xtr, Xte, ytr, yte, names, prep, train_idx, test_idx = runner.preprocess(
            X, y, selected
        )
        duplicate_frame = Xraw.copy()
        duplicate_frame["__target__"] = yraw
        identifier_names = {"id", "row_id", "rowid", "index", "customer_id",
                            "applicant_id", "loan_id", "member_id"}
        identifier_candidates = [
            str(name) for name in Xraw.columns
            if str(name).strip().lower() in identifier_names
            or str(name).strip().lower().startswith("unnamed:")
        ]
        if identifier_candidates:
            raise RuntimeError(
                f"{slug}: unhandled identifier-like columns {identifier_candidates}"
            )
        datasets.append({
            "order": int(spec["order"]),
            "slug": slug,
            "status": "PASS",
            "target_column": str(spec["target_column"]),
            "target_adverse_is_one": True,
            "openml": provenance["openml"],
            "manifest_license_note": spec.get("license", "unknown"),
            "doi": spec.get("doi"),
            "uci_id": spec.get("uci_id"),
            "source_rows": len(Xraw),
            "source_predictors": Xraw.shape[1],
            "source_adverse_count": int(yraw.sum()),
            "predictor_names": [str(value) for value in Xraw.columns],
            "predictor_dtypes": {str(name): str(dtype)
                                 for name, dtype in Xraw.dtypes.items()},
            "missing_cells_by_predictor": {
                str(name): int(value) for name, value in Xraw.isna().sum().items()
            },
            "missing_cells_total": int(Xraw.isna().sum().sum()),
            "duplicate_full_rows_including_target": int(duplicate_frame.duplicated().sum()),
            "identifier_like_columns_found": identifier_candidates,
            "identifier_columns_excluded": [],
            "cap_applied": len(Xraw) > runner.ROW_CAP,
            "retained_rows": len(y),
            "retained_adverse_count": int(y.sum()),
            "retained_source_indices_sha256": runner.sha256_array(selected),
            "train_rows": len(ytr),
            "test_rows": len(yte),
            "train_adverse_count": int(ytr.sum()),
            "test_adverse_count": int(yte.sum()),
            "train_source_indices_sha256": runner.sha256_array(train_idx),
            "test_source_indices_sha256": runner.sha256_array(test_idx),
            "numeric_columns": prep["numeric_columns"],
            "categorical_columns": prep["categorical_columns"],
            "encoded_columns": names,
            "encoded_d": Xtr.shape[1],
            "curve_length_L": Xte.shape[1] + 1,
            "train_imputed_cells": prep["train_imputed_cells"],
            "test_imputed_cells": prep["test_imputed_cells"],
            "unseen_test_levels": prep["unseen_test_levels"],
            "zero_variance_columns_removed": prep["zero_variance_columns_removed"],
            "processed_train_sha256": prep["processed_train_sha256"],
            "processed_test_sha256": prep["processed_test_sha256"],
            "raw_frame_sha256": provenance["raw_frame_sha256"],
        })
    report = {
        "audit": "outcome-blind six-dataset schema and preprocessing preflight",
        "audit_date": "2026-08-27",
        "status": "PASS",
        "safe_metrics_or_models_computed": False,
        "raw_data_retained": False,
        "protocol_sha256": runner.sha256_file(runner.PROTOCOL_PATH),
        "manifest_sha256": runner.sha256_file(runner.MANIFEST_PATH),
        "pandas": pd.__version__,
        "datasets": datasets,
    }
    runner.atomic_json(OUTPUT, report)
    print(f"PASS: wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
