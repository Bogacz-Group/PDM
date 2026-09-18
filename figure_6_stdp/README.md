# Figure 6: spike-timing-dependent plasticity

This directory independently reproduces Figure 6. `figure6.py` generates
panels a–h, while `fit_stdp.py` evaluates the recorded fits for panels i–j.
The sources at predictive_dendrite commit
[`370b8da`](https://github.com/YuhangSong/predictive_dendrite/commit/370b8daebcd6d52e37075f362f29e3a6c59e1ca3)
are `stdp_dx.ipynb` (panels a and c), `stdp_dw.ipynb` (b and d–g),
`rafal_analytic/` (h), `fit_stdp_data_OptunaSearch_Bi2002.ipynb` (i), and
`fit_stdp_data_OptunaSearch_Melanie2003.ipynb` (j), all under
`experiments/spiking/`.

## 1. System requirements

Python 3.11.14 on a 64-bit Linux or macOS system is sufficient. The
reproduction scripts run on CPU; their minimal package versions are pinned in
[`requirements.txt`](requirements.txt).

## 2. Installation

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Installation typically takes 5–10 minutes.

The reported figures do not require radas. To repeat the optional fresh
parameter searches, install [`requirements-search.txt`](requirements-search.txt)
in the same environment:

```bash
python -m pip install -r requirements-search.txt
```

This optional file pins the public
[`mini-radas`](https://github.com/YuhangSong/mini-radas) repository. Its Python
distribution and import name is `radas`. Additional installation time is
typically 10–20 minutes.

## 3. Reproduction

Run these commands from `figure_6_stdp/` with the environment activated:

| Panels | Command | Output | Estimated runtime |
| --- | --- | --- | --- |
| 6a–h | `python figure6.py` | `results/figure6.*` and `plots/figure6*` | 5–20 min |
| 6h only | `python analytic.py` | `results/figure6h.*` and `plots/figure6h.*` | under 1 min |
| 6i–j | `python fit_stdp.py` | `generated/fig6_fits/` | 1–5 min |
| 6i fresh search (optional) | `python run_search.py fig6i --seed 0` | `generated/radas/fig6i/<run-id>/` | 1–12 h |
| 6j fresh search (optional) | `python run_search.py fig6j --seed 0` | `generated/radas/fig6j/<run-id>/` | 1–12 h |

The default `fit_stdp.py` command evaluates the best configurations recorded
by the original searches; it does not rerun optimization. Each fresh search
runs 1,000 trials through mini-radas and writes the fitted curves, trial table,
plots, and run metadata. Quick runs are available with `--smoke --no-plots`.

After generating a–h locally once, replot those ignored local result files
without rerunning simulations:

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
