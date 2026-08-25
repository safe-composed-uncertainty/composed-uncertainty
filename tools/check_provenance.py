#!/usr/bin/env python3
"""Fail if a result file does not record how it was produced.

Rule three of the README: every result file carries its seed, the pinned
safeai commit, the NumPy version and a UTC timestamp.

Field names are matched through an alias table because the runs predate the
rule and used several spellings. A file may also DERIVE its provenance from an
upstream artifact; declare that in PROVENANCE-EXEMPTIONS.md and it is accepted
with the upstream named, not silently skipped.
"""
import json, sys, glob, os, re

ALIASES = {
    "seed":          ("seed", "master_seed"),
    "safeai_commit": ("safeai_commit", "safeai", "commit"),
    "numpy":         ("numpy", "numpy_version"),
    "timestamp":     ("generated_utc", "timestamp", "generated"),
}
EXEMPT_FILE = "PROVENANCE-EXEMPTIONS.md"


def load_exemptions(root):
    """Parse `- path — reason` lines. Returns {path: reason}."""
    p = os.path.join(root, EXEMPT_FILE)
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^\s*-\s+`([^`]+)`\s+[—-]\s+(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def find(d, names):
    for n in names:
        if n in d:
            return d[n]
    for v in d.values():
        if isinstance(v, dict):
            for n in names:
                if n in v:
                    return v[n]
    return None


def main(root="."):
    exempt = load_exemptions(root)
    files = sorted(glob.glob(os.path.join(root, "results", "**", "*.json"), recursive=True))
    files += sorted(glob.glob(os.path.join(root, "experiments", "**", "results-*.json"), recursive=True))
    if not files:
        print("check_provenance: no result files found — nothing to check")
        return 0
    bad = []
    for f in files:
        rel = os.path.relpath(f, root)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            bad.append((rel, f"unreadable: {e}"))
            continue
        missing = [k for k, names in ALIASES.items() if find(d, names) is None]
        if not missing:
            continue
        if rel in exempt:
            print(f"  exempt  {rel}  ({', '.join(missing)}) — {exempt[rel]}")
            continue
        bad.append((rel, "missing " + ", ".join(missing)))
    for rel, why in bad:
        print(f"  FAIL    {rel}  {why}")
    print(f"\ncheck_provenance: {len(files) - len(bad)}/{len(files)} files carry full provenance")
    if bad:
        print("Record the missing fields, or declare the file in "
              f"{EXEMPT_FILE} with the upstream artifact it inherits from.")
        print("Do NOT invent a timestamp or a version for a run that has already happened.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
