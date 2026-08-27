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
