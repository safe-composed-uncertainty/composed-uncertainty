# INTERNAL WORKING DIRECTORY - real-agentic worked example (insurance for Section 7.2)

```
INTERNAL WORKING ARTIFACT - applies to EVERY file in this directory,
including the machine artifacts that cannot carry their own header
(episodes-real-agentic.csv, results-real-agentic.json, run-real-agentic.log,
verify-chain.log, pilot/*). These files name repositories, hosts, source
vector ids and project codenames for reproducibility. None of those names
may appear in the manuscript or in any article-facing text. In the article
the substrate is described only as "a live agentic research pipeline
operated by the first author". Do not commit this directory without an
explicit go-ahead. Do not paste from any file here into the manuscript;
write Section 7.2 fresh through the naming filter of DESIGN.md Section 9.
```

Run of record: REAL-AGENTIC-SUMMARY.md. Design: DESIGN.md. Freeze:
CONFIG-real-agentic.md. Adversarial verification: verify-chain.log.

Adversarially verified 2026-07-27 (independent session): all summary
numbers traced to results-real-agentic.json / run-real-agentic.log; stage
A/logit point metrics reproduced from the frozen CSV by a from-scratch
reimplementation of the 1 - CvM/Gini rank-graduation formula (gap 0.0);
full production harness re-run in a scratchpad reproduced the run of
record; episode labels re-checked row-by-row against the frozen
channel-to-cell mapping (0 mismatches in 1,000); vendored pipeline-src
modules byte-identical to their origins and to the recorded SHA-256 set.
