"""Fabrication screen, written during the adversarial verification of 28 July 2026.

Every numeric literal in SECOND-DATASET-SUMMARY.md must be traceable to a deposited
artefact: results-second-dataset.json, either verification JSON, one of the probe
JSONs, any deposited log, or the German-credit deposits that the comparison rests on.
The screen pools every number in those files, then scans the summary token by token
and accepts a token only if some pooled value equals it, rounds to it at any
precision, or equals it after a x100 percentage conversion.

Two tokens are expected to remain and are legitimate one-step derivations rather
than quoted values: "1,005" is 542.7 + 461.9 from probe-03-projection.log, and
"8,999" is 9,000 - 1 from probe-05-rf-determinism.log. Any third survivor is a
number that entered the prose from nowhere and must be chased."""
import json, re, os, sys, math

D = "/srv/tyche/repos/tyche-research-vault/papers/safe-composed-uncertainty/experiment/second-dataset-2026-07-27"
P = "/srv/tyche/repos/tyche-research-vault/papers/safe-composed-uncertainty/experiment/pavia-composite-2026-07-25"
R = "/srv/tyche/repos/tyche-research-vault/papers/safe-composed-uncertainty/experiment/redraw-2026-07-24"

pool = set()

def harvest(o):
    if isinstance(o, dict):
        for v in o.values():
            harvest(v)
    elif isinstance(o, list):
        for v in o:
            harvest(v)
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        pool.add(float(o))
    elif isinstance(o, str):
        for m in re.finditer(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", o):
            try:
                pool.add(float(m.group()))
            except ValueError:
                pass

jsons = [
    os.path.join(D, "results-second-dataset.json"),
    os.path.join(D, "verify-second-dataset.json"),
    os.path.join(D, "verify2-from-definition-curves.json"),
    os.path.join(D, "verify2-from-definition-chain.json"),
    os.path.join(D, "probe-01-data.json"),
    os.path.join(D, "probe-02-cost.json"),
    os.path.join(D, "probe-03-projection.json"),
    os.path.join(D, "probe-04-comparability.json"),
    os.path.join(P, "results-pavia-composite.json"),
    os.path.join(R, "results-redraw.json"),
]
for j in jsons:
    harvest(json.load(open(j)))

logs = [
    os.path.join(D, "run-second-dataset.log"),
    os.path.join(D, "main-run-stdout.txt"),
    os.path.join(D, "verify-second-dataset.log"),
    os.path.join(D, "verify-run-stdout.txt"),
    os.path.join(D, "probe-01-data.log"),
    os.path.join(D, "probe-02-cost.log"),
    os.path.join(D, "probe-03-projection.log"),
    os.path.join(D, "probe-04-comparability.log"),
    os.path.join(D, "probe-06-comparability-of-the-two-studies.log"),
    os.path.join(D, "verify2-from-definition-curves.log"),
    os.path.join(D, "verify2-from-definition-chain.log"),
    os.path.join(D, "probe-05-rf-determinism.log"),
    os.path.join(D, "prereg-number-audit.log"),
    os.path.join(D, "tables.md"),
    os.path.join(P, "verify-cross-terms.log"),
    os.path.join(P, "run-pavia-composite.log"),
    os.path.join(D, "PRE-REGISTRATION.md"),
]
for f in logs:
    if os.path.exists(f):
        txt = open(f, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", txt):
            try:
                pool.add(float(m.group()))
            except ValueError:
                pass

import math as _mm
pool = sorted(v for v in pool if _mm.isfinite(v))
print(f"pool size {len(pool)}")

# derived: also allow rounded forms and simple sign flips
def traceable(x):
    """A summary token is traceable if some pooled value equals it, or rounds to it
    at any precision the summary might have used, or equals it after a x100
    percentage conversion. Comparisons are relative, so that round-off in the
    screen itself cannot manufacture a miss."""
    def close(a, b):
        return abs(a - b) <= max(abs(b), abs(a)) * 1e-9 + 1e-300
    for cand in (x, -x):
        for p in pool:
            for base in (p, p * 100.0):
                if close(base, cand):
                    return True
                for nd in range(0, 12):
                    if close(round(base, nd), cand):
                        return True
                for sf in range(1, 13):
                    if base != 0:
                        import math as _m
                        e = _m.floor(_m.log10(abs(base)))
                        if close(round(base, -(e - sf + 1)), cand):
                            return True
    return False

summary = open(os.path.join(D, "SECOND-DATASET-SUMMARY.md"), encoding="utf-8").read()
lines = summary.split("\n")

untraced = []
seen = 0
skip_small = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0}
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith("|") and set(ln.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
        continue
    for m in re.finditer(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:e[-+]?\d+)?(?![\w])", ln):
        tok = m.group().replace(",", "")
        try:
            x = float(tok)
        except ValueError:
            continue
        seen += 1
        if abs(x) < 1e-30:
            continue
        if not traceable(x):
            untraced.append((i, tok, ln.strip()[:150]))

print(f"numeric tokens scanned: {seen}")
print(f"UNTRACED: {len(untraced)}")
for i, tok, ln in untraced:
    print(f"  L{i}: {tok!r}   << {ln}")
