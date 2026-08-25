# Finite-sample coverage study

Run date: 2026-07-27.
Outer replications per scenario: 1000. Bootstrap replications per dataset: 999.
Master seed: 20260727. Runtime: 114.7 seconds.

The data-generating process is a bounded Gaussian-copula model with analytically known component means. Components share each sampled row; the third can add independently redrawn perturbation noise. The paired bootstrap uses one index stream for all components. The independent negative control uses a separate stream per component.

## Joint-region coverage (%)

| Scenario | n | raw chi-squared | raw Hotelling | logit chi-squared | logit Hotelling | raw bootstrap radial | logit bootstrap radial | independent control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| central-positive-n100 | 100 | 94.7 | 95.6 | 94.9 | 95.7 | 94.5 | 95.1 | 93.3 |
| central-positive-n300 | 300 | 94.9 | 95.4 | 95.2 | 95.6 | 94.7 | 95.0 | 90.9 |
| central-positive-n800 | 800 | 95.0 | 95.1 | 95.0 | 95.2 | 95.0 | 95.1 | 91.8 |
| upper-bound-positive-n300 | 300 | 92.7 | 93.0 | 94.3 | 94.5 | 92.6 | 94.4 | 93.0 |
| central-zero-n300 | 300 | 94.7 | 95.1 | 94.7 | 95.2 | 94.8 | 95.0 | 94.4 |
| central-negative-n300 | 300 | 94.9 | 95.0 | 95.0 | 95.1 | 94.6 | 94.7 | 94.5 |
| mixed-sign-n300 | 300 | 95.2 | 95.5 | 95.0 | 95.8 | 95.0 | 95.3 | 93.8 |
| central-positive-noise-n300 | 300 | 94.6 | 94.9 | 94.9 | 94.9 | 97.1 | 97.3 | 92.0 |
| upper-bound-positive-noise-n300 | 300 | 93.7 | 94.3 | 93.8 | 93.9 | 97.2 | 97.4 | 93.4 |

## Composite-interval coverage (%)

| Scenario | delta | logit delta | percentile | BCa | independent percentile | paired width | independent width |
|---|---:|---:|---:|---:|---:|---:|---:|
| central-positive-n100 | 95.0 | 95.1 | 95.2 | 95.1 | 80.9 | 0.0943 | 0.0647 |
| central-positive-n300 | 94.1 | 94.1 | 94.2 | 94.0 | 80.2 | 0.0549 | 0.0375 |
| central-positive-n800 | 94.4 | 94.4 | 94.1 | 93.8 | 80.4 | 0.0336 | 0.0230 |
| upper-bound-positive-n300 | 95.7 | 95.9 | 95.8 | 95.8 | 84.0 | 0.0240 | 0.0171 |
| central-zero-n300 | 95.4 | 95.4 | 95.4 | 95.6 | 95.4 | 0.0375 | 0.0375 |
| central-negative-n300 | 95.0 | 95.0 | 94.7 | 95.0 | 99.8 | 0.0219 | 0.0375 |
| mixed-sign-n300 | 95.6 | 95.6 | 95.2 | 95.3 | 95.8 | 0.0371 | 0.0374 |
| central-positive-noise-n300 | 95.3 | 95.3 | 95.6 | 91.5 | 85.6 | 0.0549 | 0.0396 |
| upper-bound-positive-noise-n300 | 94.7 | 95.2 | 97.1 | 89.3 | 90.8 | 0.0265 | 0.0204 |

## Reading

Across the nine scenarios, paired-percentile composite coverage ranges from 94.1% to 97.1%. The independent-stream control ranges from 80.2% to 99.8%. Joint chi-squared coverage ranges from 92.7% to 95.2%, and Hotelling coverage from 93.0% to 95.6%. On the logit scale, the corresponding ranges are 93.8% to 95.2% and 93.9% to 95.8%.

The Monte Carlo standard error of a 95% coverage estimate is 0.69 percentage points. Results should therefore be read in bands, not as exact rankings separated by one percentage point.

Raw results: `results-coverage.json` and `results-coverage.csv`. Driver: `coverage_simulation.py`.
