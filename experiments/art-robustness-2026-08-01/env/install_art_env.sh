#!/usr/bin/env bash
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -m venv "$here/../.venv-art"
# shellcheck disable=SC1091
. "$here/../.venv-art/bin/activate"
pip install --upgrade pip
pip install -r "$here/requirements-art.lock.txt"
python - <<'PY'
import art, torch, sklearn
from art.attacks.evasion import HopSkipJump, FastGradientMethod
from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor
print("ART", art.__version__, "torch", torch.__version__, "sklearn", sklearn.__version__, "- imports OK")
PY
