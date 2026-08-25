# ART robustness environment digest

Verified by live install + smoke run on 2026-08-01 (Zeus1, CPython 3.12.3).

- adversarial-robustness-toolbox==1.20.1
- torch==2.13.0+cpu  (CPU wheel: pip install --index-url https://download.pytorch.org/whl/cpu torch)
- numpy==2.5.1, scipy==1.18.0, scikit-learn==1.9.0, packaging==26.2
- safeai pinned commit 39768fc (koleso500/safeai), same as redraw-2026-07-24/safeai-src

## Gotchas confirmed live (not from docs)

1. ART 1.20.1 __init__ import chain unconditionally imports torch (reaches
   art.estimators.certification which subclasses torch.nn.Module). The inert
   torch stub in redraw-2026-07-24/_stubs/ CANNOT run real ART — install the
   genuine CPU torch wheel (~11s, no CUDA).
2. ART also needs `packaging` explicitly (its object_detection module imports
   packaging.version); a bare ART install misses it. Pinned in the lockfile.
3. SklearnClassifier(model=...) wraps predict(), so HopSkipJump works on the
   random forest with NO gradients. FGM needs class/loss gradients -> logit
   pipeline only (the RF wrapper does not expose them).

## install

    python3 -m venv venv-art && . venv-art/bin/activate
    pip install -r requirements-art.lock.txt
    # if installing fresh instead of from the lock:
    pip install adversarial-robustness-toolbox==1.20.1 scikit-learn scipy numpy packaging
    pip install --index-url https://download.pytorch.org/whl/cpu torch
