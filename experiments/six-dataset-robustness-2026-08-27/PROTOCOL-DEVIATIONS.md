# Protocol deviations

## 2026-08-27 — pandas pin changed before any outcome computation

The freeze commit pinned pandas 3.0.4 because that was the host version used
for the outcome-blind schema probe. During construction of the isolated run
environment, pip reported that 3.0.4 had been yanked for datetime-related
segmentation faults. Before any model or SAFE metric was run, the environment
pin was changed to pandas 3.0.5. This is an environment-safety change; the
dataset set, preprocessing recipe, endpoints, seeds, and analysis are
unchanged. The initially created 3.0.4 environment was not used for outcomes.

## 2026-08-27 — HMEQ checksum field corrected before outcomes

The initial manifest accidentally placed the SHA-256 of SAS's official
`hmeq.csv` in the field intended for the selected OpenML ARFF snapshot. Both
files were outcome-blind provenance checks of the same named dataset. Before
any model or SAFE metric was run, `download_sha256` was corrected to the exact
OpenML 43337 v2 bytes and the SAS checksum was retained separately as
`official_sas_csv_sha256`. No data choice or analysis setting changed.

## 2026-08-27 — verifier preflight-contract correction after outcomes

The first directory-wide verification attempt, using verifier schema 1.2,
stopped before checking any claim-bearing aggregate with the error
`australian-credit-approval: run-time preflight binding disagrees with the
frozen audit`. The failure report is retained as
`verify-six-dataset-v1.2-initial-FAIL.json`.

Diagnosis showed that verifier 1.2 constructed an 11-field expected preflight
subset and then required it to equal the runner's complete 21-field preflight
contract. All 11 overlapping fields agreed; exact dictionary equality failed
because the runner also deposited ten frozen provenance and count fields. The
verifier was corrected to construct and require the complete 21-field contract
from `schema-preflight-audit.json`, matching the outcome-blind runner check.
This strengthens rather than relaxes the binding and does not change a
dataset, model, seed, endpoint, replicate, or deposited outcome array. The
verifier schema was advanced to 1.3 and the production analysis was not rerun.
