# Composed uncertainty for SAFE compliance scores — code and run records

Code and run records for the article by Paolo Giudici, Vasily Kolesnikov and
Anton Sokolov. The manuscript itself is **not** here: Overleaf is the
authoritative copy, and a second one in this repository would become a third
version.

## Layout

| Directory | Holds |
|:--|:--|
| `episodes/` | the episode generator and the seed manifest |
| `scorers/` | thin wrappers over pinned `safeai` |
| `experiments/` | one self-contained directory per study |
| `results/` | the committed result JSON |
| `figures/` | one generator per figure in the paper |
| `data/` | pointers and checksums, never copies of the public datasets |

## Three rules

These are rules rather than habits, and `tools/` enforces them.

**1. Numbers live in the result JSON and nowhere else.** A figure that
hard-codes a measured number will silently survive a rerun that changes it.
Figure generators read every measured quantity out of the result files.

**2. `safeai` stays pinned and is never vendored.** It is pinned by commit and
cloned at run time, and the pin is recorded in every result file. The package
belongs to its author; a frozen copy here would fork it by accident.

**3. Every result file carries its seed.**

## Checks

```
python3 tools/check_provenance.py .   # rule 3, and the pin from rule 2
python3 tools/check_figures.py .      # rule 1, structurally
```

Both accept declared exemptions, in `PROVENANCE-EXEMPTIONS.md` and
`FIGURE-EXEMPTIONS.md`. An exemption names the file and the reason, so it is a
decision on the record rather than a gap. The checks print exemptions rather
than hiding them.

⛔ **Never satisfy `check_provenance` by writing in a value you did not
measure.** A seed can be recovered from the run script; a timestamp and a
library version cannot be recovered after the fact. If they were not recorded,
either re-run the experiment or declare the exemption honestly.

## Licence

MIT, matching `safeai`.
