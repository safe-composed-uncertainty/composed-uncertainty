# Six-dataset credit robustness extension

This is a prospectively specified six-dataset extension performed after the German-credit and Taiwan analyses. It is post hoc relative to those studies and is not part of the original Taiwan pre-registration.

## Prespecified endpoint

The Fisher-z tail-minus-Gaussian contrast had the expected negative direction in **12/12 model arms**. Both models were negative in **6/6 datasets**.

Across all 48 dataset/model/composite delta-width calculations, setting cross-covariances to zero changed width by a signed **-8.38% to +4.97%**. The minimum occurred for `fico-heloc-cleaned`/rf/geometric; the maximum for `australian-credit-approval`/logit/geometric. Positive means the zero-cross-covariance interval is narrower.

## All intended datasets

| dataset | retained/test | raw/encoded d | Delta logit | Delta rf | class |
|:--|--:|--:|--:|--:|:--|
| Statlog Australian Credit Approval | 690/207 | 14/42 | -1.0918 | -0.8149 | directionally concordant |
| SAS HMEQ Home Equity | 5960/1788 | 12/18 | -0.7375 | -0.3477 | directionally concordant |
| FICO HELOC cleaned | 9871/2962 | 23/37 | -1.1373 | -0.8018 | directionally concordant |
| Give Me Some Credit | 30000/9000 | 10/10 | -0.6963 | -0.7217 | directionally concordant |
| Lending Club Loan Data | 9578/2874 | 13/19 | -1.1154 | -0.8499 | directionally concordant |
| Credit Risk Dataset | 30000/9000 | 11/26 | -0.8612 | -0.4586 | directionally concordant |

## Execution status and verifier correction

All six dataset runs completed on their first production attempt; no outcome run was retried. The first directory-wide verifier 1.2 attempt stopped before aggregate verification because it compared an 11-field expected subset for exact equality with the runner's 21-field preflight contract. That FAIL report is retained as `verify-six-dataset-v1.2-initial-FAIL.json`. Verifier 1.3 instead requires the complete frozen 21-field contract and passed all deposits; the correction changed no dataset, model, seed, endpoint, replicate, metric formula, tolerance, or aggregate logic. Full details are in `PROTOCOL-DEVIATIONS.md`.

## Claim boundary and audit

The dataset is the replication unit; the two fitted models are sensitivity arms, not independent replications. These results describe only the six frozen, publicly retrievable benchmark cohorts under the recorded representation, split, model, and perturbation design. Raw source files are not redistributed. Mixed-data one-hot runs change the feature unit and are an adapted stratum. No statement of generality across credit data, interval calibration, or legal validity is made.

All six production JSON/NPZ deposits passed the independent verifier. Protocol, source/schema audits, code, logs, deposits, and verifier output are in the immutable [six-dataset-robustness-results-v1](https://github.com/safe-composed-uncertainty/composed-uncertainty/tree/six-dataset-robustness-results-v1) tagged snapshot.

## Manuscript-ready paragraph

After completing the German-credit and Taiwan analyses, we prospectively fixed an extension protocol and applied the same estimators to six additional publicly retrievable tabular consumer-credit benchmark cohorts. The prespecified tail-swap-versus-Gaussian Fisher-z contrast had the expected negative direction in 6/6 datasets (12/12 model fits). Across the 48 dataset/model/composite calculations, the signed effect of setting cross-covariances to zero ranged from -8.38% to +4.97%. Complete selection, preprocessing, failures, results, replicate deposits, and independent verification are available in the immutable six-dataset-robustness-results-v1 tagged repository snapshot. This extension is separate from the original Taiwan pre-registration.
