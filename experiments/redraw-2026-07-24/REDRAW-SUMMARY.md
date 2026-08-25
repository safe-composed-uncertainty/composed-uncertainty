# Redraw check: are the cross-metric correlations an artefact of a fixed perturbation seed?

Run 2026-07-24, zeus1. German credit (OpenML credit-g v1), train 700 / test 300, d = 20,
B = 2000 paired bootstrap replicates, alpha = 0.05, noise sd 0.5, pinned safeai
`39768fcd5264c881f7174268bbffda52b298ae89`. Latest runtime 217.2 s.
Raw output: `results-redraw.json`; log: `run-redraw.log`; driver: `redraw_experiment.py`.
The original v0.1 experiment directory was not modified (it backs a published deposit).

## The question

In the deposited experiment (DOI 10.5281/zenodo.21516475) the RGR perturbations were drawn
once with a fixed seed and held across bootstrap replicates. The measured cross-metric
correlations are therefore conditional on that auxiliary draw. Paolo's independence
argument (24 July) is that perturbations are sampled independently of the data, so the
honest question is whether the correlations survive redrawing them per replicate.

Redraw scheme: fresh perturbations on every replicate, seeded per replicate index
(`REDRAW_BASE[model] + b`) so the run stays reproducible.

## Result: they survive

| Pair | logit fixed-draw | logit redraw | rf fixed-draw | rf redraw |
|---|---|---|---|---|
| RGA-RGE | +0.286 | +0.286 | +0.292 | +0.292 |
| RGA-RGR | +0.160 | +0.127 | +0.153 | +0.139 |
| RGE-RGR | +0.597 | +0.430 | +0.342 | +0.329 |

Two-link chain (logit then rf, h = C_A * C_B):

| | fixed-draw | redraw |
|---|---|---|
| cross-link correlation | +0.722 | +0.712 |
| plug-in point, full-sample link-score product | 0.8266 | 0.8266 |
| composed 95% interval width, measured covariance | 0.0798 | 0.0800 |
| same, cross-term declared zero | 0.0608 | 0.0611 |
| understatement of width | 23.8% | 23.6% |

## Reading

The RGA-RGE correlation is **identical to the digit** under both schemes. That is not luck:
RGE carries no auxiliary randomness at the default settings, so redrawing perturbations
cannot touch it (see below). Only the pairs involving RGR move, and they move down
moderately: the fixed-draw values are conditional on one perturbation realisation and sit
above the unconditional ones, exactly as expected. The largest single drop is RGE-RGR on
the logit model, +0.597 to +0.430.

The headline claim is unaffected. The chain result is what carries the paper, and it barely
moves: correlation +0.722 to +0.712, understatement 23.8% to 23.6%. "About a quarter of the
interval width" holds under both schemes.

For the article, the correct wording is the redraw column: it is the unconditional
correlation, and it is the one a referee will ask for. The peak "up to 0.60" from the
deposit should become "up to 0.43 unconditionally, 0.60 conditional on a fixed perturbation
draw" — or simply be replaced by the chain number, which is stronger and stable.

## RGE determinism (the second question)

Confirmed deterministic at default settings, from the pinned code: the RGE random generator
is `np.random.default_rng(random_seed if random_seed is not None else 42)`
(`safeai/rge.py:1560` tabular, `rge.py:1372` text) and is used **only** by the `'random'`
masking strategy, while the default is `'greedy'` (`rge.py:1635-1671`); the tabular curve is
pinned to start at 1.0 (`rge.py:1578`). Image workflows with random occlusion are the
exception and are seeded only if a seed is passed (`utils.py:725-727`).

The empirical run corroborates this: RGA-RGE is unchanged to 16 digits across schemes.

Consequence for the paper: write "explanation terms", not "explanation draws". RGE depends
on the test sample alone, like RGA — so the shared-sample channel is not one of three
symmetric sources but the dominant one, carrying two of the three metrics outright.

Related asymmetry, worth one sentence in the methods: RGR's noise path defaults to
`default_rng(None)`, i.e. **nondeterministic** (`rgr.py:914`), and the PyTorch path ignores
the passed generator entirely (`rgr.py:1613-1618`). Reproducible covariance estimation with
common random numbers needs that seed plumbing fixed — a natural first contribution to
`koleso500/safeai` while Vasily is away.
