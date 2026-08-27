#!/usr/bin/env python3
"""Verify all six production deposits and render the prespecified public summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import verify_six_dataset as verifier

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "dataset-manifest.json"
VERIFY_REPORT = HERE / "verify-six-dataset.json"
SUMMARY_JSON = HERE / "summary-six-dataset.json"
SUMMARY_CSV = HERE / "summary-six-dataset.csv"
SUMMARY_MD = HERE / "SIX-DATASET-SUMMARY.md"
MODEL_TAGS = ("logit", "rf")
COMPOSITES = ("arithmetic", "geometric", "rms", "topsis")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REPOSITORY = "https://github.com/safe-composed-uncertainty/composed-uncertainty"


def classify(deltas: dict[str, float]) -> str:
    count = sum(value < 0.0 for value in deltas.values())
    if count == 2:
        return "directionally concordant"
    if count == 1:
        return "mixed"
    return "contrary"


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def load_manifest() -> list[dict[str, Any]]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return sorted(payload["datasets"], key=lambda item: int(item["order"]))


def resolve_annotated_tag(tag: str) -> tuple[Path, str]:
    root = Path(subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True
    ).strip())
    ref = f"refs/tags/{tag}"
    object_type = subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "-t", ref], text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if object_type != "tag":
        raise ValueError("release tag must exist locally as an annotated tag")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{ref}^{{commit}}"], text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return root, commit


def validate_immutable_results_tag(tag: str, specs: list[dict[str, Any]]) -> str:
    root, commit = resolve_annotated_tag(tag)
    required = [
        HERE / "PROTOCOL.md", HERE / "dataset-manifest.json",
        HERE / "requirements-six-dataset.txt", HERE / "data-source-audit.json",
        HERE / "schema-preflight-audit.json", HERE / "IMPLEMENTATION-NOTES.md",
        HERE / "run_six_dataset.py", HERE / "verify_six_dataset.py",
        HERE / "VERIFIER-SCHEMA.md", HERE / "verify-six-dataset.json",
    ]
    for spec in specs:
        slug = str(spec["slug"])
        required.extend([
            HERE / f"results-{slug}.json",
            HERE / f"replicates-{slug}.npz",
            HERE / f"run-{slug}.log",
        ])
    for path in required:
        relative = path.relative_to(root)
        tagged_blob = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", f"{commit}:{relative}"], text=True
        ).strip()
        working_blob = subprocess.check_output(
            ["git", "-C", str(root), "hash-object", str(path)], text=True
        ).strip()
        if tagged_blob != working_blob:
            raise ValueError(f"release tag does not bind current artifact {relative}")
    return commit


def inventory_results(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = []
    for spec in specs:
        slug = str(spec["slug"])
        path = HERE / f"results-{slug}.json"
        if not path.is_file():
            inventory.append({"order": int(spec["order"]), "slug": slug,
                              "name": spec["name"], "status": "missing",
                              "failure_type": "MissingResult",
                              "failure_message": str(path.name)})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            failure = payload.get("failure") or {}
            inventory.append({"order": int(spec["order"]), "slug": slug,
                              "name": spec["name"],
                              "status": str(payload.get("status", "unknown")),
                              "failure_type": str(failure.get("type", "")),
                              "failure_message": str(failure.get("message", ""))})
        except (OSError, json.JSONDecodeError) as exc:
            inventory.append({"order": int(spec["order"]), "slug": slug,
                              "name": spec["name"], "status": "invalid",
                              "failure_type": type(exc).__name__,
                              "failure_message": str(exc)})
    return inventory


def write_failure_summary(release_tag: str, specs: list[dict[str, Any]],
                          exc: Exception) -> None:
    inventory = inventory_results(specs)
    failures = [item for item in inventory if item["status"].lower()
                not in {"complete", "completed", "ok", "pass", "success"}]
    try:
        _root, release_commit = resolve_annotated_tag(release_tag)
        release_url = f"{REPOSITORY}/tree/{release_tag}"
    except (OSError, ValueError, subprocess.CalledProcessError):
        release_commit = None
        release_url = None
    summary = {
        "schema_version": "six-dataset-summary-v1.1",
        "status": "FAIL",
        "release_tag": release_tag,
        "release_url": release_url,
        "release_commit": release_commit,
        "intended_dataset_count": 6,
        "completed_but_not_set_verified_count": 6 - len(failures),
        "failures": failures,
        "all_six_primary_counts": None,
        "all_48_cross_covariance_effect_range": None,
        "verification_error": {"type": type(exc).__name__, "message": str(exc)},
        "claim_boundary": "No all-six result may be claimed because verification failed.",
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(inventory[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(inventory)
    lines = [
        "# Six-dataset credit robustness extension — incomplete",
        "",
        ("The intended six-dataset set did not pass independent verification. No "
         "all-six directional count or 48-cell covariance-effect range is claimed."),
        "",
        f"Verifier error: `{type(exc).__name__}: {exc}`",
        "",
        "## Intended-set status",
        "",
        "| dataset | status | failure |",
        "|:--|:--|:--|",
    ]
    for item in inventory:
        failure = " — ".join(value for value in
                             (item["failure_type"], item["failure_message"]) if value)
        lines.append(f"| {item['name']} | {item['status']} | {failure} |")
    lines.append("")
    if release_url is not None:
        lines.append(
            f"The retained artifacts are available at [{release_tag}]({release_url})."
        )
    else:
        lines.append("No validated annotated release tag was available for this failure set.")
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def build_rows(report: dict[str, Any], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        slug = str(spec["slug"])
        verified = report["datasets"][slug]["computed"]
        deltas = {
            tag: float(verified["models"][tag]["fisher_z_tail_minus_gaussian"])
            for tag in MODEL_TAGS
        }
        rows.append({
            "order": int(spec["order"]),
            "slug": slug,
            "name": str(spec["name"]),
            "source_rows": int(spec["rows"]),
            "retained_rows": int(verified["n_retained"]),
            "test_rows": int(verified["n_test"]),
            "raw_d": int(verified["raw_d"]),
            "encoded_d": int(verified["d"]),
            "delta_logit": deltas["logit"],
            "delta_rf": deltas["rf"],
            "classification": classify(deltas),
            "verification": "PASS",
        })
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    effect = aggregate["cross_covariance_effect_signed_pct_range"]
    minimum = effect["minimum"]
    maximum = effect["maximum"]
    release_url = summary["release_url"]
    lines = [
        "# Six-dataset credit robustness extension",
        "",
        ("This is a prospectively specified six-dataset extension performed after "
         "the German-credit and Taiwan analyses. It is post hoc relative to those "
         "studies and is not part of the original Taiwan pre-registration."),
        "",
        "## Prespecified endpoint",
        "",
        (f"The Fisher-z tail-minus-Gaussian contrast had the expected negative "
         f"direction in **{aggregate['model_arms_delta_lt_zero_out_of_12']}/12 "
         f"model arms**. Both models were negative in "
         f"**{aggregate['datasets_directionally_concordant_out_of_6']}/6 datasets**."),
        "",
        ("Across all 48 dataset/model/composite delta-width calculations, setting "
         "cross-covariances to zero changed width by a signed "
         f"**{minimum['signed_width_change_pct_zero_vs_measured']:+.2f}% to "
         f"{maximum['signed_width_change_pct_zero_vs_measured']:+.2f}%**. The "
         "minimum occurred for "
         f"`{minimum['slug']}`/{minimum['model']}/{minimum['composite']}; the maximum "
         f"for `{maximum['slug']}`/{maximum['model']}/{maximum['composite']}. "
         "Positive means the zero-cross-covariance interval is narrower."),
        "",
        "## All intended datasets",
        "",
        "| dataset | retained/test | raw/encoded d | Delta logit | Delta rf | class |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for row in summary["datasets"]:
        lines.append(
            f"| {row['name']} | {row['retained_rows']}/{row['test_rows']} | "
            f"{row['raw_d']}/{row['encoded_d']} | {fmt(row['delta_logit'])} | "
            f"{fmt(row['delta_rf'])} | {row['classification']} |"
        )
    lines.extend([
        "",
        "## Execution status and verifier correction",
        "",
        ("All six dataset runs completed on their first production attempt; no "
         "outcome run was retried. The first directory-wide verifier 1.2 attempt "
         "stopped before aggregate verification because it compared an 11-field "
         "expected subset for exact equality with the runner's 21-field preflight "
         "contract. That FAIL report is retained as "
         "`verify-six-dataset-v1.2-initial-FAIL.json`. Verifier 1.3 instead requires "
         "the complete frozen 21-field contract and passed all deposits; the "
         "correction changed no dataset, model, seed, endpoint, replicate, metric "
         "formula, tolerance, or aggregate logic. Full details are in "
         "`PROTOCOL-DEVIATIONS.md`."),
        "",
        "## Claim boundary and audit",
        "",
        ("The dataset is the replication unit; the two fitted models are sensitivity "
         "arms, not independent replications. These results describe only the six "
         "frozen, publicly retrievable benchmark cohorts under the recorded "
         "representation, split, model, and perturbation design. Raw source files "
         "are not redistributed. Mixed-data one-hot runs change the feature unit and "
         "are an adapted stratum. No statement of generality across credit data, "
         "interval calibration, or legal validity is made."),
        "",
        (f"All six production JSON/NPZ deposits passed the independent verifier. "
         f"Protocol, source/schema audits, code, logs, deposits, and verifier output "
         f"are in the immutable [{summary['release_tag']}]({release_url}) tagged "
         "snapshot."),
        "",
        "## Manuscript-ready paragraph",
        "",
        summary["manuscript_ready_paragraph"],
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args(argv)
    if not TAG_RE.fullmatch(args.release_tag):
        parser.error("release tag contains unsupported characters")

    specs = load_manifest()
    try:
        report = verifier.verify_results_dir(HERE)
    except (OSError, ValueError, verifier.VerificationError) as exc:
        write_failure_summary(args.release_tag, specs, exc)
        print(f"FAIL: wrote transparent intended-set failure summary: {exc}")
        return 1
    verifier.write_report(VERIFY_REPORT, report)
    release_commit = validate_immutable_results_tag(args.release_tag, specs)
    rows = build_rows(report, specs)
    aggregate = report["aggregate"]
    minimum_effect = aggregate["cross_covariance_effect_signed_pct_range"]["minimum"]
    maximum_effect = aggregate["cross_covariance_effect_signed_pct_range"]["maximum"]
    release_url = f"{REPOSITORY}/tree/{args.release_tag}"
    paragraph = (
        "After completing the German-credit and Taiwan analyses, we prospectively "
        "fixed an extension protocol and applied the same estimators to six additional "
        "publicly retrievable tabular consumer-credit benchmark cohorts. The "
        "prespecified tail-swap-versus-Gaussian Fisher-z contrast had the expected "
        "negative direction in "
        f"{aggregate['datasets_directionally_concordant_out_of_6']}/6 datasets "
        f"({aggregate['model_arms_delta_lt_zero_out_of_12']}/12 model fits). Across "
        "the 48 dataset/model/composite calculations, the signed effect of setting "
        "cross-covariances to zero ranged from "
        f"{minimum_effect['signed_width_change_pct_zero_vs_measured']:+.2f}% to "
        f"{maximum_effect['signed_width_change_pct_zero_vs_measured']:+.2f}%. "
        "Complete selection, "
        "preprocessing, failures, results, replicate deposits, and independent "
        f"verification are available in the immutable {args.release_tag} tagged "
        "repository snapshot. This extension is separate from the original Taiwan "
        "pre-registration."
    )
    summary = {
        "schema_version": "six-dataset-summary-v1.1",
        "status": "PASS",
        "release_tag": args.release_tag,
        "release_url": release_url,
        "release_commit": release_commit,
        "verification_report": VERIFY_REPORT.name,
        "verification_report_sha256": verifier.sha256_file(VERIFY_REPORT),
        "protocol_sha256": report["protocol_sha256"],
        "dataset_spec_sha256": report["dataset_spec_sha256"],
        "aggregate": aggregate,
        "intended_dataset_count": 6,
        "completed_and_verified_dataset_count": 6,
        "failures": [],
        "deviations": [{
            "type": "verifier-preflight-contract-correction",
            "initial_verifier_schema": "1.2",
            "initial_status": "FAIL",
            "initial_report": "verify-six-dataset-v1.2-initial-FAIL.json",
            "corrected_verifier_schema": "1.3",
            "corrected_status": "PASS",
            "outcome_run_retried": False,
            "outcome_logic_changed": False,
            "details": "PROTOCOL-DEVIATIONS.md",
        }],
        "datasets": rows,
        "manuscript_ready_paragraph": paragraph,
        "claim_boundary": (
            "Finite-set six-cohort result; datasets are replication units; model "
            "arms are sensitivities; mixed-data one-hot runs are an adapted stratum; "
            "raw source files are not redistributed."
        ),
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(f"PASS: wrote {VERIFY_REPORT.name}, {SUMMARY_JSON.name}, "
          f"{SUMMARY_CSV.name}, and {SUMMARY_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
