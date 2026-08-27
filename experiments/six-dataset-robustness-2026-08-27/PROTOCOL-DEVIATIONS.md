# Protocol deviations

## 2026-08-27 — pandas pin changed before any outcome computation

The freeze commit pinned pandas 3.0.4 because that was the host version used
for the outcome-blind schema probe. During construction of the isolated run
environment, pip reported that 3.0.4 had been yanked for datetime-related
segmentation faults. Before any model or SAFE metric was run, the environment
pin was changed to pandas 3.0.5. This is an environment-safety change; the
dataset set, preprocessing recipe, endpoints, seeds, and analysis are
unchanged. The initially created 3.0.4 environment was not used for outcomes.
