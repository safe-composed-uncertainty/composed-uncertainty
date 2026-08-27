#!/usr/bin/env python3
"""Verify the frozen six OpenML snapshots without retaining raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "dataset-manifest.json"
SAS_HMEQ_URL = (
    "https://support.sas.com/documentation/onlinedoc/viya/"
    "exampledatasets/hmeq.csv"
)


def digest_bytes(payload: bytes) -> tuple[str, str]:
    return hashlib.md5(payload).hexdigest(), hashlib.sha256(payload).hexdigest()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "data-source-audit.json")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = {
        "audit": "six-dataset source bytes and OpenML metadata",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "raw_data_retained": False,
        "datasets": [],
    }
    failures: list[str] = []
    for spec in manifest["datasets"]:
        data_id = spec["openml_data_id"]
        metadata_url = f"https://www.openml.org/api/v1/json/data/{data_id}"
        metadata = fetch_json(metadata_url)["data_set_description"]
        raw = fetch_bytes(metadata["url"])
        md5, sha256 = digest_bytes(raw)
        checks = {
            "id": int(metadata["id"]) == data_id,
            "version": int(metadata["version"]) == spec["openml_version"],
            "name": metadata["name"] == spec["openml_name"],
            "api_md5_matches_manifest": metadata["md5_checksum"] == spec["openml_md5"],
            "download_md5_matches_api": md5 == metadata["md5_checksum"],
            "download_sha256_matches_manifest": sha256 == spec["download_sha256"],
        }
        entry = {
            "slug": spec["slug"],
            "openml_data_id": data_id,
            "metadata_url": metadata_url,
            "download_url": metadata["url"],
            "download_bytes": len(raw),
            "download_md5": md5,
            "download_sha256": sha256,
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        }
        if not entry["all_checks_pass"]:
            failures.append(spec["slug"])
        report["datasets"].append(entry)

    hmeq = next(x for x in manifest["datasets"] if x["slug"] == "hmeq")
    sas_raw = fetch_bytes(SAS_HMEQ_URL)
    sas_md5, sas_sha256 = digest_bytes(sas_raw)
    report["hmeq_official_sas_cross_pointer"] = {
        "url": SAS_HMEQ_URL,
        "download_bytes": len(sas_raw),
        "download_md5": sas_md5,
        "download_sha256": sas_sha256,
        "sha256_matches_manifest": sas_sha256 == hmeq["official_sas_csv_sha256"],
    }
    if not report["hmeq_official_sas_cross_pointer"]["sha256_matches_manifest"]:
        failures.append("hmeq-official-sas-pointer")

    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(report['datasets'])} OpenML snapshots audited")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
