# Figure 6: spike-timing-dependent plasticity

This directory independently reproduces Figure 6. `figure6.py` generates
panels a–h, while `fit_stdp.py` evaluates the recorded fits for panels i–j.
`run_search.py` repeats the two parameter searches with the public
[`mini-radas`](https://github.com/YuhangSong/mini-radas) package. Source
notebooks and the pinned migration revision are listed in
[`provenance.json`](provenance.json).

## 1. System requirements

Python 3.11.14 on a 64-bit Linux or macOS system is sufficient. The scripts
run on CPU; package versions are pinned in [`requirements.txt`](requirements.txt).

## 2. Installation

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Installation typically takes 10–20 minutes; the tests take under 1 minute.

## 3. Reproduction

Run these commands from `figure_6_stdp/` with the environment activated:

| Panels | Command | Output | Estimated runtime |
| --- | --- | --- | --- |
| 6a–h | `python figure6.py` | `results/figure6.*` and `plots/figure6*` | 5–20 min |
| 6h only | `python analytic.py` | `results/figure6h.*` and `plots/figure6h.*` | under 1 min |
| 6i–j | `python fit_stdp.py` | `generated/fig6_fits/` | 1–5 min |
| 6i fresh search | `python run_search.py fig6i --seed 0` | `generated/radas/fig6i/<run-id>/` | 1–12 h |
| 6j fresh search | `python run_search.py fig6j --seed 0` | `generated/radas/fig6j/<run-id>/` | 1–12 h |

The default `fit_stdp.py` command evaluates the best configurations recorded
by the original searches; it does not rerun optimization. Each fresh search
runs 1,000 trials and writes the fitted curves, trial table, plots, and run
metadata. Quick integration checks are available with `--smoke --no-plots`.

Existing a–h results can be replotted without rerunning simulations:

```bash
python figure6.py --action plot
python analytic.py --action plot
```

## 4. Command-line options

Examples of the principal scientific controls are:

```bash
# Change the timing grid and learning rates used in panels a–h.
python figure6.py \
  --delay-start-ms -60 --delay-stop-ms 60 --delay-step-ms 2 \
  --alpha-values 0.005 0.01 0.02

# Use the strict ±5 ms Methods weighting for a new seeded search.
python run_search.py fig6i \
  --weighting manuscript --variance-floor 1e-12 \
  --num-samples 2000 --seed 7

# Run a reduced 32-trial search.
python run_search.py fig6j --quick --seed 0
```

Use `python <script>.py --help` for all options. The source fitting notebooks
used five neighboring observations on either side of each sorted observation,
not the literal ±5 ms window described in Methods. Therefore `legacy` is the
default weighting for compatibility; `manuscript` implements the stated time
window and records its variance floor. Both modes normalize model predictions
to the sum of the observations, matching the source notebooks.
