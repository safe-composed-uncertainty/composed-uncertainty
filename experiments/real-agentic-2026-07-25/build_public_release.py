#!/usr/bin/env python3
"""Build the sanitized public artifact for the executed-system example."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE = HERE
PUBLIC = HERE.parent / "real-system-public"
SOURCE_RESULTS = SOURCE / "results-real-agentic.json"
SOURCE_EPISODES = SOURCE / "episodes-real-agentic.csv"
SOURCE_VERIFICATION = SOURCE / "verify-chain.log"
RESULTS_OUT = PUBLIC / "results.json"
EPISODES_OUT = PUBLIC / "episodes.csv"
VERIFICATION_OUT = PUBLIC / "verification.log"
MANIFEST_OUT = PUBLIC / "artifact-manifest.json"
CHANNEL_NAMES = {
    "dtbs_field_tamper": "evidence_field_tamper",
    "tampered_ear": "tampered_execution_record",
    "untrusted_sam": "untrusted_evidence_signer",
}


def public_channel(name: str) -> str:
    return CHANNEL_NAMES.get(name, name)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize_results(source: dict[str, Any]) -> dict[str, Any]:
    sorted_module_hashes = [
        value for _, value in sorted(source["pipeline_module_sha256"].items())
    ]
    module_set_digest = hashlib.sha256(
        canonical_bytes(sorted_module_hashes)
    ).hexdigest()
    module_digests = [
        {"module_id": f"module-{index:02d}", "sha256": value}
        for index, (_, value) in enumerate(
            sorted(source["pipeline_module_sha256"].items()), start=1
        )
    ]
    channels = [
        {
            "id": public_channel(item["id"]),
            "probability": item["prob"],
            "expected_cell": item["expected_cell"],
        }
        for item in source["channels"]
    ]
    return {
        "artifact_profile": "executed-two-stage-uncertainty-example/1.0",
        "design": (
            "seeded randomized episodes executed by one two-stage "
            "authorization-and-evidence verifier"
        ),
        "N": source["N"],
        "n_train": source["n_train"],
        "n_eval": source["n_eval"],
        "B": source["B"],
        "alpha": source["alpha"],
        "noise_sd": source["noise_sd"],
        "seed": source["seed"],
        "episode_seed": source["episode_seed"],
        "episode_seed_rule": source["episode_seed_rule"],
        "redraw_seed_rule": source["redraw_seed_rule"],
        "index_streams": source["index_streams"],
        "safeai_commit": source["safeai_commit"],
        "pipeline_module_set_sha256": module_set_digest,
        "pipeline_modules": module_digests,
        "versions": source["versions"],
        "channels": channels,
        "features": source["features"],
        "channel_counts": {
            public_channel(name): count
            for name, count in source["channel_counts"].items()
        },
        "eval_cells_2x2": source["eval_cells_2x2"],
        "prevalence": source["prevalence"],
        "scorer_auc_eval": source["scorer_auc_eval"],
        "stages": source["stages"],
        "joint": source["joint"],
        "chain": source["chain"],
        "runtime_seconds": source["runtime_seconds"],
        "scope_note": (
            "Randomized executed episodes under a frozen stress mixture; "
            "not production-traffic observations or field prevalence."
        ),
    }


def sanitize_episodes() -> None:
    with SOURCE_EPISODES.open(newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError("episode table has no header")
        dropped = {"sources", "tamper_detail"}
        fieldnames = [name for name in reader.fieldnames if name not in dropped]
        rows = []
        for row in reader:
            public_row = {name: row[name] for name in fieldnames}
            public_row["channel"] = public_channel(public_row["channel"])
            rows.append(public_row)
    with EPISODES_OUT.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_verification_log() -> None:
    text = SOURCE_VERIFICATION.read_text()
    if "VERIFY PASS" not in text:
        raise ValueError("independent verification has not completed successfully")
    forbidden = (
        "/srv/",
        "accountable-agentic-action",
        "machine-mandate",
        "aep-sandbox",
    )
    if any(token in text for token in forbidden):
        raise ValueError("verification log contains an internal source label")
    VERIFICATION_OUT.write_text(
        "# Independent statistical recomputation\n\n" + text
    )


def main() -> None:
    source_results = json.loads(SOURCE_RESULTS.read_text())
    RESULTS_OUT.write_text(
        json.dumps(sanitize_results(source_results), indent=2, ensure_ascii=False)
        + "\n"
    )
    sanitize_episodes()
    sanitize_verification_log()
    artifacts = {
        path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in (
            PUBLIC / "METHODS.md",
            RESULTS_OUT,
            EPISODES_OUT,
            VERIFICATION_OUT,
        )
    }
    manifest = {
        "artifact_profile": "executed-two-stage-uncertainty-example/1.0",
        "generated_utc": "2026-07-27T00:00:00Z",
        "files": artifacts,
        "source_boundary": (
            "Operational keys, internal repository names, host paths, and "
            "per-module filenames are excluded."
        ),
    }
    MANIFEST_OUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"wrote sanitized release: {len(artifacts)} hashed files; "
        f"{sum(1 for _ in EPISODES_OUT.open()) - 1} episodes"
    )


if __name__ == "__main__":
    main()
