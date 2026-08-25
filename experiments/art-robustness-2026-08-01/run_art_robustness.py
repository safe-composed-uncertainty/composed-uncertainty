"""Full frozen-artifact ART robustness run (entry point skeleton).

Fits the deposited German-credit (and, when its spec is filled, Taiwan) models,
loads the frozen severity grids and per-dataset constraints, runs each attack
adapter ONCE per model, and emits the RGR_* curve keys of OUTPUT-SCHEMA.md plus
the p_perturbed tensor for the existing bootstrap/covariance machinery.

This is intentionally a skeleton: it wires the verified pieces together and is
the command in PRE-BUILD-CHECKLIST.md step 3. The methodological choices (which
arms are headline, mutable-field specs, success metric) are set by
severity_grids/grids.json and constraints/*.json, not hard-coded here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from constraints.constraints import FeatureConstraints, FeatureSpec
from adapters.hopskipjump_adapter import HopSkipJumpAdapter


def load_grids():
    return json.loads((HERE / "severity_grids" / "grids.json").read_text())


def main():
    print("run_art_robustness skeleton — see PRE-BUILD-CHECKLIST.md.")
    print("Confirmed working pieces: env, FeatureConstraints (9/9 tests),")
    print("HopSkipJumpAdapter, smoke test on real models.")
    print("To run the full arm: fill/confirm the constraint specs, choose")
    print("n_rows and grid, then loop the adapters over {logit, rf} and dump")
    print("results-art-robustness.json + .npz in the curves_point convention.")
    grids = load_grids()
    print("severity grids loaded:", list(grids.keys()))


if __name__ == "__main__":
    main()
