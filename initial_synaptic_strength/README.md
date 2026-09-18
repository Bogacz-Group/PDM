# Figure 7: dependence on initial synaptic strength

This self-contained folder reproduces Figure 7a and 7b. The sources at
predictive_dendrite commit
[`370b8da`](https://github.com/YuhangSong/predictive_dendrite/commit/370b8daebcd6d52e37075f362f29e3a6c59e1ca3)
are `experiments/spiking/stdp_dw.ipynb` (7a) and
`experiments/spiking/fit_stdp_data_initial_w.ipynb` (7b).

## 1. System requirements

Tested on 64-bit Linux with Python 3.11.14. A CPU is sufficient. Minimal Python
dependencies are pinned in [`requirements.txt`](requirements.txt).

## 2. Installation

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Installation typically takes 5–10 minutes.

The reported figures do not require radas. To repeat the optional fresh
parameter search, install [`requirements-search.txt`](requirements-search.txt)
in the same environment:

```bash
python -m pip install -r requirements-search.txt
```

This optional file pins the public
[`mini-radas`](https://github.com/YuhangSong/mini-radas) repository. Its Python
distribution and import name is `radas`; allow another 10–20 minutes to install
the optional search stack.

## 3. Reproduction

```bash
# Figure 7a: initial weight versus the simulated STDP curve
python figure7.py

# Figure 7b: evaluate the best configuration recorded by the source notebook
python fit_initial_strength.py

# Figure 7b: optionally repeat the complete seeded 1,000-trial search
python run_with_radas.py --seed 0
```

The first two commands write CSV/JSON data and PDF/SVG plots under `results/`,
`plots/`, and `generated/fig7b_fit/`; each typically finishes in 1–5 minutes.
The complete search writes its trial table, best-fit outputs, plots, and run
metadata under `generated/radas/<run-id>/` and may take 1–12 hours.

## 4. Command-line options

```bash
# Change the initial weights and delay grid in Figure 7a
python figure7.py --initial-weights 0.1 0.3 0.5 0.7 0.9 \
  --delay-start-ms -60 --delay-stop-ms 60 --delay-step-ms 2

# Run a smaller development search or change the random seed
python run_with_radas.py --quick --seed 7

# Use the model-weight interval printed in Methods Section H
python run_with_radas.py --search-space manuscript --seed 0

# Replot Figure 7a after generating its local CSV once
python figure7.py --action plot --results results/figure7a.csv
```

`--help` lists all options. The default `legacy` Figure 7b search uses the
`[0.5, 2.0]` model-weight range executed by the source notebook. The manuscript
states `[0.0, 0.3]`; this is exposed as `--search-space manuscript`. The
reported legacy best value (`0.651577...`) is outside the manuscript interval,
so the two variants are kept explicit rather than silently conflated.
