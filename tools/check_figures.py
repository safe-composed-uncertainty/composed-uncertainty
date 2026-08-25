#!/usr/bin/env python3
"""Fail if a figure in the paper has no generator, or a generator has no inputs.

"Notebooks that produce every figure" is true on the day it is written and
quietly false a month later. This is the check that keeps it true.

Not every figure is a result. A figure that illustrates the argument rather
than reporting a measurement belongs in FIGURE-EXEMPTIONS.md, so that it is a
decision on the record rather than a gap.
"""
import os, sys, re, glob

EXEMPT_FILE = "FIGURE-EXEMPTIONS.md"
# .tex is a real figure output here: the TikZ generators emit LaTeX that the
# manuscript compiles. Omitting it made the check fail two generators that
# were doing exactly what they should.
FIG_EXT = (".pdf", ".png", ".svg", ".tex")


def load_exemptions(root):
    p = os.path.join(root, EXEMPT_FILE)
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^\s*-\s+`([^`]+)`\s+[—-]\s+(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def stem(name):
    b = os.path.basename(name)
    for e in FIG_EXT:
        if b.endswith(e):
            b = b[: -len(e)]
    return re.sub(r"^(fig|figure)[-_]?", "", b).replace("_", "-").lower()


def main(root="."):
    exempt = load_exemptions(root)
    figs, gens = {}, {}
    for f in glob.glob(os.path.join(root, "figures", "**", "*"), recursive=True):
        b = os.path.basename(f)
        if b.endswith(FIG_EXT):
            figs.setdefault(stem(b), []).append(os.path.relpath(f, root))
        elif b.startswith("make_") and b.endswith(".py"):
            gens[stem(b[len("make_"):-3])] = os.path.relpath(f, root)
    problems = []
    for s, paths in sorted(figs.items()):
        if s in gens:
            continue
        rel = paths[0]
        if rel in exempt or s in exempt:
            print(f"  exempt  {rel} — {exempt.get(rel, exempt.get(s))}")
            continue
        problems.append((rel, "no generator (expected figures/make_%s.py)" % s))
    for s, g in sorted(gens.items()):
        if s not in figs:
            problems.append((g, "generator produces no committed figure"))
    for rel, why in problems:
        print(f"  FAIL    {rel}  {why}")
    print(f"\ncheck_figures: {len(figs)} figures, {len(gens)} generators, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
