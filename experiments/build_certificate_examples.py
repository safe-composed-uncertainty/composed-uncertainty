#!/usr/bin/env python3
"""Build and verify the executable certificate examples for profile 0.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import jsonschema
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "certificate-profile-0.3.schema.json"
OUTPUT_DIR = HERE / "certificate-examples"
GERMAN_RESULTS = HERE / "redraw-2026-07-24" / "results-redraw.json"
REAL_DIR = HERE / "real-agentic-2026-07-25"
REAL_RESULTS = REAL_DIR / "results-real-agentic.json"
REAL_EPISODES = HERE / "real-system-public" / "episodes.csv"

ISSUED = "2026-07-27T00:00:00Z"
RECALIBRATE_BY = "2026-10-25T00:00:00Z"
KEY_USE = "PUBLICATION EXAMPLE KEY — NEVER OPERATIONAL"
CANONICALIZATION = (
    "UTF-8 JSON; sorted keys; separators comma/colon; ensure_ascii=false"
)
COMPONENT_ORDER = ["RGA", "RGE", "RGR"]
RECALIBRATION_TRIGGERS = [
    "model retrain",
    "substrate version or module-set change",
    "evaluation-sample change",
    "perturbation-family or conditioning-scheme change",
    "detected distribution shift",
]
GERMAN_REDRAW_RULE = (
    "fresh perturbation per replicate; seed = model base + replicate index; "
    "bases: logistic 20360723, random forest 20460723"
)
REAL_REDRAW_RULE = (
    "fresh perturbation per replicate; seed = stage/model base + replicate "
    "index; bases: A/logistic 20360727, B/logistic 20460727, "
    "A/random forest 20560727, B/random forest 20660727"
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example_private_key() -> Ed25519PrivateKey:
    seed = hashlib.sha256(
        b"composed-uncertainty-certificate-publication-example"
    ).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def sign(certificate: dict[str, Any]) -> dict[str, Any]:
    payload = canonical_bytes(certificate)
    private_key = example_private_key()
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return {
        "certificate": certificate,
        "signature": {
            "algorithm": "Ed25519",
            "canonicalization": CANONICALIZATION,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "public_key_hex": public_key.hex(),
            "signature_hex": private_key.sign(payload).hex(),
            "key_use": KEY_USE,
        },
    }


def validity() -> dict[str, Any]:
    return {
        "issued_utc": ISSUED,
        "recalibrate_by_utc": RECALIBRATE_BY,
        "recalibration_triggers": RECALIBRATION_TRIGGERS,
        "horizon_rule": (
            "90 days maximum; a chain expires at the earliest input-link expiry"
        ),
    }


def estimation(seed: int, redraw_rule: str) -> dict[str, Any]:
    return {
        "method": (
            "paired nonparametric bootstrap over shared evaluation rows; "
            + redraw_rule
        ),
        "resampling_unit": "evaluation row",
        "replicates": 2000,
        "alpha": 0.05,
        "seed": seed,
        "perturbation_family": "Gaussian feature noise, scale 0.5 column SD",
        "conditioning_scheme": "redraw_per_replicate",
        # Profile 0.3: the interval covers evaluation sampling for one frozen
        # trained artifact. It does not cover training-split, seed or
        # hyperparameter variability; a retrain is a recalibration trigger.
        "uncertainty_scope": "fixed_artifact_evaluation",
    }


def link_certificate(
    *,
    certificate_id: str,
    system_id: str,
    substrate: dict[str, str],
    dataset: str,
    sample_digest: str,
    point: dict[str, float],
    redraw: dict[str, Any],
    seed: int,
    redraw_rule: str,
) -> dict[str, Any]:
    return sign(
        {
            "schema": "composed-uncertainty-certificate/0.3",
            "certificate_id": certificate_id,
            "certificate_type": "link",
            "subject": {
                "system_id": system_id,
                "substrate": substrate,
                "dataset": dataset,
                "test_sample_digest": sample_digest,
            },
            "measurement": {
                "component_order": COMPONENT_ORDER,
                "component_point_estimates": [point[name] for name in COMPONENT_ORDER],
                "component_covariance": redraw["Sigma"],
                "aggregator": "geometric mean of RGA, RGE and RGR",
                "composed_score": {
                    "estimator": (
                        "plug-in geometric mean of full-sample component estimates"
                    ),
                    "value": point["C_geomean"],
                    "interval95": redraw["ci_C_paired_percentile"],
                    # Percentile endpoints of the paired bootstrap: not a
                    # function of component_covariance, hence not recomputable
                    # from it. Profile 0.3 records that fact here.
                    "interval_method": "paired_bootstrap_percentile",
                },
            },
            "estimation": estimation(seed, redraw_rule),
            "validity": validity(),
        }
    )


def chain_covariance(
    chain: dict[str, Any],
    link_a: dict[str, Any],
    link_b: dict[str, Any],
) -> list[list[float]]:
    if "link_covariance" in chain:
        return chain["link_covariance"]
    sd_a = link_a["sd_C_paired"]
    sd_b = link_b["sd_C_paired"]
    cov_ab = chain["cross_link_correlation"] * sd_a * sd_b
    return [[sd_a**2, cov_ab], [cov_ab, sd_b**2]]


def chain_certificate(
    *,
    certificate_id: str,
    system_id: str,
    substrate: dict[str, str],
    dataset: str,
    sample_digest: str,
    chain: dict[str, Any],
    link_a_result: dict[str, Any],
    link_b_result: dict[str, Any],
    link_envelopes: list[dict[str, Any]],
    seed: int,
    redraw_rule: str,
) -> dict[str, Any]:
    covariance = chain_covariance(chain, link_a_result, link_b_result)
    return sign(
        {
            "schema": "composed-uncertainty-certificate/0.3",
            "certificate_id": certificate_id,
            "certificate_type": "chain",
            "subject": {
                "system_id": system_id,
                "substrate": substrate,
                "dataset": dataset,
                "test_sample_digest": sample_digest,
            },
            "measurement": {
                "chain_functional": "h = C_A × C_B (both-must-hold)",
                "estimator": chain.get(
                    "estimator",
                    "plug-in product of the two full-sample link scores",
                ),
                "link_certificate_payload_sha256": [
                    envelope["signature"]["payload_sha256"]
                    for envelope in link_envelopes
                ],
                "link_point_values": [
                    chain["link_point_values"]["A"],
                    chain["link_point_values"]["B"],
                ],
                "link_covariance": covariance,
                "cross_link_covariance": {
                    "status": "measured",
                    "value": covariance[0][1],
                    "correlation": chain["cross_link_correlation"],
                    "provenance": (
                        "same paired bootstrap index stream over the shared sample; "
                        f"{redraw_rule}"
                    ),
                    "shared_sample_digest": sample_digest,
                },
                "composed_score": {
                    "value": chain["h_point"],
                    "interval95": chain["ci95_measured_covariance"],
                    # First-order interval: point +/- z * sqrt(g' Sigma g) with
                    # g the gradient of the chain functional. Recomputable from
                    # link_point_values and link_covariance alone; the
                    # zero_cross_term_sensitivity interval below is the same
                    # construction with the off-diagonal entry set to zero.
                    "interval_method": "first_order_delta",
                },
                "zero_cross_term_sensitivity": {
                    "interval95": chain["ci95_cross_term_declared_zero"],
                    "width_change_pct": chain["understatement_of_width_pct"],
                },
            },
            "estimation": estimation(seed, redraw_rule),
            "validity": validity(),
        }
    )


def build_examples() -> dict[str, dict[str, Any]]:
    german = json.loads(GERMAN_RESULTS.read_text())
    real = json.loads(REAL_RESULTS.read_text())

    german_digest = (
        "fabffe0abe7180657ef28dd4b0d4f3a16d897c0566c0cd1e4ba3a2e6c672ddbf"
    )
    german_substrate = {
        "name": "safeai",
        "version_or_commit": german["safeai_commit"],
    }
    german_links = []
    for tag, label in (
        ("logit", "standardised logistic-regression link"),
        ("rf", "300-tree random-forest link"),
    ):
        model = german["models"][tag]
        german_links.append(
            link_certificate(
                certificate_id=f"example-german-{tag}-link-20260727",
                system_id=label,
                substrate=german_substrate,
                dataset="Statlog German credit (OpenML credit-g v1), held-out rows",
                sample_digest=german_digest,
                point=model["point_fixed_draw"],
                redraw=model["redraw"],
                seed=german["seed"],
                redraw_rule=GERMAN_REDRAW_RULE,
            )
        )
    german_chain = chain_certificate(
        certificate_id="example-german-two-link-chain-20260727",
        system_id="two-model shared-sample stand-in",
        substrate=german_substrate,
        dataset="Statlog German credit (OpenML credit-g v1), held-out rows",
        sample_digest=german_digest,
        chain=german["chain"]["redraw"],
        link_a_result=german["models"]["logit"]["redraw"],
        link_b_result=german["models"]["rf"]["redraw"],
        link_envelopes=german_links,
        seed=german["seed"],
        redraw_rule=GERMAN_REDRAW_RULE,
    )

    real_digest = sha256_file(REAL_EPISODES)
    module_manifest = canonical_bytes(
        [
            value
            for _, value in sorted(real["pipeline_module_sha256"].items())
        ]
    )
    real_substrate = {
        "name": "executed authorization-and-evidence pipeline with safeai metric layer",
        "version_or_commit": real["safeai_commit"],
        "module_set_sha256": hashlib.sha256(module_manifest).hexdigest(),
    }
    real_links = []
    for stage, label in (
        ("A", "authorization-stage probabilistic scorer"),
        ("B", "evidence-verification-stage probabilistic scorer"),
    ):
        result = real["stages"][stage]["logit"]
        real_links.append(
            link_certificate(
                certificate_id=f"example-real-stage-{stage.lower()}-link-20260727",
                system_id=label,
                substrate=real_substrate,
                dataset="1,000 executed randomized episodes; held-out evaluation rows",
                sample_digest=real_digest,
                point=result["point_fixed_draw"],
                redraw=result["redraw"],
                seed=real["seed"],
                redraw_rule=REAL_REDRAW_RULE,
            )
        )
    real_chain_result = real["chain"]["logit"]["redraw"]
    if "link_point_values" not in real_chain_result:
        real_chain_result = {
            **real_chain_result,
            "estimator": "plug-in product of the two full-sample link scores",
            "link_point_values": {
                "A": real["stages"]["A"]["logit"]["point_fixed_draw"]["C_geomean"],
                "B": real["stages"]["B"]["logit"]["point_fixed_draw"]["C_geomean"],
            },
        }
    real_chain = chain_certificate(
        certificate_id="example-real-two-stage-chain-20260727",
        system_id="executed two-stage agentic research pipeline",
        substrate=real_substrate,
        dataset="1,000 executed randomized episodes; held-out evaluation rows",
        sample_digest=real_digest,
        chain=real_chain_result,
        link_a_result=real["stages"]["A"]["logit"]["redraw"],
        link_b_result=real["stages"]["B"]["logit"]["redraw"],
        link_envelopes=real_links,
        seed=real["seed"],
        redraw_rule=REAL_REDRAW_RULE,
    )

    return {
        "german-logit-link.json": german_links[0],
        "german-random-forest-link.json": german_links[1],
        "german-chain.json": german_chain,
        "real-stage-a-link.json": real_links[0],
        "real-stage-b-link.json": real_links[1],
        "real-chain.json": real_chain,
    }


def verify_envelope(
    envelope: dict[str, Any],
    validator: jsonschema.Draft202012Validator,
) -> str:
    validator.validate(envelope)
    payload = canonical_bytes(envelope["certificate"])
    signature = envelope["signature"]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != signature["payload_sha256"]:
        raise ValueError("payload digest mismatch")
    public_key = Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(signature["public_key_hex"])
    )
    public_key.verify(bytes.fromhex(signature["signature_hex"]), payload)
    return digest


def first_order_interval(
    point: float,
    gradient: list[float],
    covariance: list[list[float]],
    alpha: float,
) -> tuple[list[float], float]:
    """point +/- z * sqrt(g' Sigma g); the delta-method interval on the raw
    scale, which is what interval_method = first_order_delta names."""
    variance = sum(
        gradient[i] * covariance[i][j] * gradient[j]
        for i in range(len(gradient))
        for j in range(len(gradient))
    )
    half = NormalDist().inv_cdf(1.0 - alpha / 2.0) * math.sqrt(variance)
    return [point - half, point + half], 2.0 * half


def recomputation_report(paths: list[Path]) -> None:
    """Check every certificate against its own declared interval_method.

    A chain record declares first_order_delta, so its interval must be an exact
    function of link_point_values and link_covariance; we require agreement to
    1e-12. A link record declares paired_bootstrap_percentile, so its endpoints
    are NOT a function of component_covariance; we report the residual gap so
    that the difference between the two classes is measured, not assumed.
    """
    for path in paths:
        certificate = json.loads(path.read_text())["certificate"]
        measurement = certificate["measurement"]
        composed = measurement["composed_score"]
        alpha = certificate["estimation"]["alpha"]
        recorded = composed["interval95"]
        width = recorded[1] - recorded[0]
        if certificate["certificate_type"] == "chain":
            link_a, link_b = measurement["link_point_values"]
            gradient = [link_b, link_a]  # d(C_A C_B) / d(C_A, C_B)
            covariance = measurement["link_covariance"]
            rebuilt, _ = first_order_interval(
                composed["value"], gradient, covariance, alpha
            )
            gap = max(abs(rebuilt[k] - recorded[k]) for k in (0, 1))
            zeroed = [[covariance[0][0], 0.0], [0.0, covariance[1][1]]]
            sensitivity = measurement["zero_cross_term_sensitivity"]
            rebuilt_zero, width_zero = first_order_interval(
                composed["value"], gradient, zeroed, alpha
            )
            gap_zero = max(
                abs(rebuilt_zero[k] - sensitivity["interval95"][k]) for k in (0, 1)
            )
            change = 100.0 * (width - width_zero) / width
            gap_change = abs(change - sensitivity["width_change_pct"])
            if composed["interval_method"] != "first_order_delta":
                raise ValueError(f"{path.name}: unexpected chain interval_method")
            if max(gap, gap_zero) > 1e-12 or gap_change > 1e-9:
                raise ValueError(f"{path.name}: chain interval not self-consistent")
            print(
                f"  {path.name}: chain, first_order_delta, "
                f"recomputed from its own covariance, max endpoint gap "
                f"{max(gap, gap_zero):.1e}"
            )
        else:
            estimates = measurement["component_point_estimates"]
            value = composed["value"]
            # gradient of the geometric mean: dC/dtheta_k = C / (3 theta_k)
            gradient = [value / (3.0 * theta) for theta in estimates]
            rebuilt, _ = first_order_interval(
                value, gradient, measurement["component_covariance"], alpha
            )
            gap = max(abs(rebuilt[k] - recorded[k]) for k in (0, 1))
            if composed["interval_method"] != "paired_bootstrap_percentile":
                raise ValueError(f"{path.name}: unexpected link interval_method")
            print(
                f"  {path.name}: link, paired_bootstrap_percentile, "
                f"a first-order recomputation from component_covariance misses "
                f"the recorded endpoints by {gap:.2e} "
                f"({100.0 * gap / width:.2f} per cent of the width)"
            )


def verify_outputs(expected: dict[str, dict[str, Any]] | None = None) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    paths = sorted(OUTPUT_DIR.glob("*.json"))
    if expected is not None and {path.name for path in paths} != set(expected):
        raise ValueError("certificate output set does not match the builder")
    digests = {}
    for path in paths:
        envelope = json.loads(path.read_text())
        digests[path.name] = verify_envelope(envelope, validator)
        if expected is not None and envelope != expected[path.name]:
            raise ValueError(f"{path.name} differs from deterministic rebuild")
    link_digests = {
        digest for name, digest in digests.items() if name.endswith("-link.json")
    }
    for path in paths:
        envelope = json.loads(path.read_text())
        if envelope["certificate"]["certificate_type"] == "chain":
            declared = set(
                envelope["certificate"]["measurement"][
                    "link_certificate_payload_sha256"
                ]
            )
            if not declared.issubset(link_digests):
                raise ValueError(f"{path.name} references an unknown link payload")
    print(f"verified {len(paths)} schema-valid, digest-valid, signed certificates")
    print("interval self-consistency against the declared interval_method:")
    recomputation_report(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_outputs()
        return
    examples = build_examples()
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, envelope in examples.items():
        (OUTPUT_DIR / name).write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n"
        )
    verify_outputs(examples)


if __name__ == "__main__":
    main()
