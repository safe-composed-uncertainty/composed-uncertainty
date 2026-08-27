# Six-dataset runbook

Run from the repository root on a clean checkout. Python 3.12.3 was used for the
frozen run. The production runner refuses modified or staged tracked files and
asserts the six package versions before loading data.

```bash
python3 -m venv /tmp/composed-uncertainty-six
/tmp/composed-uncertainty-six/bin/pip install -r \
  experiments/six-dataset-robustness-2026-08-27/requirements-six-dataset.txt
```

Outcome-free gates:

```bash
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/run_six_dataset.py \
  --preflight-only
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/audit_schema_preflight.py
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/verify_six_dataset.py \
  --self-test
```

Tiny-B structural smoke run (not a production estimate):

```bash
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/run_six_dataset.py \
  --dataset australian-credit-approval --B 12 \
  --conditioning-logit 3 --conditioning-rf 2 --output-suffix SMOKE01
```

Frozen production run and independent verification:

```bash
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/run_six_dataset.py --dataset all
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/verify_six_dataset.py \
  --results-dir experiments/six-dataset-robustness-2026-08-27
```

After verification passes, commit the exact production results, logs, deposits,
and verifier report and create the annotated immutable-results tag. The summary
refuses a missing/lightweight tag or one whose blobs differ from the working
artifacts.

```bash
git add experiments/six-dataset-robustness-2026-08-27/results-*.json \
  experiments/six-dataset-robustness-2026-08-27/replicates-*.npz \
  experiments/six-dataset-robustness-2026-08-27/run-*.log \
  experiments/six-dataset-robustness-2026-08-27/verify-six-dataset.json
git commit -m "Deposit verified six-dataset robustness results"
git tag -a six-dataset-robustness-results-v1 -m \
  "Verified six-dataset robustness results v1"
/tmp/composed-uncertainty-six/bin/python \
  experiments/six-dataset-robustness-2026-08-27/make_six_dataset_summary.py \
  --release-tag six-dataset-robustness-results-v1
```

The runner is serial by design. Expect hours rather than minutes for the full
six-dataset run; progress is written after each 250 paired replicates and each 10
full-conditioning replicates. Deposits are derived curves only and should remain
well below GitHub's per-file size limit. OpenML's local download cache lives
outside this study directory. Do not copy raw ARFF/CSV files into the repository.

Every non-frozen count requires a suffix, and no output is overwritten. If any
run fails or is interrupted, retain its first log/result and record the event in
`PROTOCOL-DEVIATIONS.md` before a suffixed retry.
