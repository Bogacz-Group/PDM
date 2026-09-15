# Spiking experiments

This folder contains script-based, publication-oriented reproductions of the
spiking results in Figures 5–7 and Supplementary Figure 3d. Numerical
generation is separated from plotting where practical: CSV and JSON files
record the values and parameters, while PDF and SVG files contain the plots.

Figure 5a is a conceptual schematic. It has no numerical source experiment and
is therefore not generated here.

## Figure and source map

| Paper panel | Public entry point | Original analysis-workspace source |
|---|---|---|
| Figure 5b–c | `figure5.py` | `experiments/spiking/basics.ipynb` |
| Figure 6a, 6c | `figure6.py` | `experiments/spiking/stdp_dx.ipynb` |
| Figure 6b, 6d–g | `figure6.py` | `experiments/spiking/stdp_dw.ipynb` |
| Figure 6h | `figure6.py` (`analytic.py` is a standalone alternative) | `experiments/spiking/rafal_analytic/` |
| Figure 6i | `fit_stdp.py` | `experiments/spiking/fit_stdp_data_OptunaSearch_Bi2002.ipynb` |
| Figure 6j | `fit_stdp.py` | `experiments/spiking/fit_stdp_data_OptunaSearch_Melanie2003.ipynb` |
| Figure 7a | `figure7.py` | `experiments/spiking/stdp_dw.ipynb` |
| Figure 7b | `fit_initial_strength.py` | `experiments/spiking/fit_stdp_data_initial_w.ipynb` |
| Supplementary Figure 3d | `supplementary_figure3d.py` | `experiments/spiking/simulation_features.ipynb` |

Supplementary Figure 3c is the firing-rate counterpart and is documented in
[`../xor/README.md`](../xor/README.md).

All source paths in this table refer to `YuhangSong/predictive_dendrite` commit
`370b8daebcd6d52e37075f362f29e3a6c59e1ca3`; the repository-level
[`migration_manifest.json`](../migration_manifest.json) is the canonical,
machine-readable mapping.

## 1. System requirements

- **Operating system:** tested on 64-bit Linux. The scripts use standard
  Python packages and should also run on macOS. The commands below use a POSIX
  shell.
- **Python:** Python 3.11.14.
- **Hardware:** CPU execution is supported and no GPU is required. Seeded
  adaptive fits run one trial at a time for deterministic Optuna suggestions;
  independent supplementary-grid trials may run in parallel when Ray has
  multiple CPUs available.
- **Other software:** Git is required because the open-source experiment
  orchestration dependency is installed from GitHub.
- **Python packages:** `numpy==2.2.6`, `pandas==2.3.3`,
  `matplotlib==3.10.7`, `seaborn==0.13.2`, `scipy==1.16.2`,
  `scikit-learn==1.7.2`, `torch==2.9.0`, `ray==2.53.0`,
  `pyarrow==22.0.0`, `optuna==4.5.0`, and `pytest==8.4.2`. The remaining
  direct dependencies and their exact versions are in `requirements.txt`.
- **Experiment orchestration:** the `radas` Python package is installed from
  the public
  [`mini-radas`](https://github.com/YuhangSong/mini-radas) repository at commit
  `8314d91b773db5148b0a90a8442f549e367e9cfb`.
  `run_with_radas.py` verifies the imported public source fingerprint before
  creating or replacing an experiment.

The digitized experimental points required by Figures 6i–j and 7b are included
under `data/`; no dataset download is required.

## 2. Installation

From a fresh clone, create a self-contained environment for this folder:

```bash
cd PDM/spiking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd ..
```

Estimated installation time: **10–20 minutes** on a broadband connection.
This is an estimate rather than a benchmark; downloading the PyTorch and Ray
wheels usually dominates and a warm package cache can be much faster.

Verify the installation from the repository root (estimated runtime: **under
1 minute**):

```bash
python -m pytest spiking/tests -q
```

## 3. Reproduction

Run the following commands from the repository root with the `spiking`
environment activated.

### Figures 5b–c

```bash
python spiking/figure5.py
```

Outputs:

- `spiking/results/figure5.csv` and `figure5.json`: long-form numerical traces
  and complete parameter metadata;
- `spiking/plots/figure5b.{pdf,svg}`: spike, potential, error, energy, and
  weight traces;
- `spiking/plots/figure5c.{pdf,svg}`: parameter-dependence traces.

Estimated runtime: **1–3 minutes** on a current desktop CPU.

### Figures 6a–h

```bash
python spiking/figure6.py
```

Outputs:

- `spiking/results/figure6.csv` and `figure6.json`: traces, STDP curves, and
  parameters;
- `spiking/plots/figure6a.{pdf,svg}` through
  `spiking/plots/figure6h.{pdf,svg}`;
- `spiking/plots/figure6_core.{pdf,svg}`: a combined 6a–h layout.

Estimated runtime: **5–20 minutes** on a current desktop CPU. To reproduce
only the analytical comparison in Figure 6h, use `python spiking/analytic.py`
(estimated **under 2 minutes**).

### Figures 6i–j

The fast reproduction path evaluates the best parameter configurations
recorded by the original searches:

```bash
python spiking/fit_stdp.py
```

This writes one JSON provenance record, observation CSV, curve CSV, PDF, and
SVG per panel under `spiking/generated/fig6_fits/`. Estimated runtime:
**1–5 minutes** for both panels.

A fresh, seeded 1,000-trial search for each panel can be run through the public
mini-radas entry point:

```bash
python spiking/run_with_radas.py fig6i --seed 0
python spiking/run_with_radas.py fig6j --seed 0
```

Each command writes observations, fitted curve, fit metadata, every trial,
PDF, SVG, and run metadata under
`spiking/generated/radas/<target>/<run-id>/`. The run ID contains the profile,
seed, and a short scientific-options hash. Raw Ray storage is kept separately
under `spiking/generated/radas_storage/`.
The `fig6i` file stem is `fig6i_bi2002`; the `fig6j` stem is
`fig6j_woodin2003`.

Estimated runtime: **1–12 hours per panel**, depending strongly on CPU speed.
Fresh stochastic searches need not rediscover the exact recorded optimum.

### Figures 7a–b

```bash
python spiking/figure7.py
python spiking/fit_initial_strength.py
```

`figure7.py` writes `spiking/results/figure7a.{csv,json}` and
`spiking/plots/figure7a.{pdf,svg}`. `fit_initial_strength.py` writes a JSON
provenance record, observation CSV, curve CSV, PDF, and SVG under
`spiking/generated/fig7b_fit/`. Estimated runtime: **1–5 minutes per script**.

For a fresh, seeded 1,000-trial Figure 7b search:

```bash
python spiking/run_with_radas.py fig7b --seed 0
```

This writes the fit JSON, observations, fitted curves, trials, PDF, SVG, and
`fig7b_run.json` under `spiking/generated/radas/fig7b/<run-id>/`.

Estimated runtime: **1–12 hours** on a desktop CPU.

### Supplementary Figure 3d

The paper grid contains 81 initial forward-weight pairs (`-1` to `1` in steps
of `0.25`), each trained for 64 passes through the four XOR examples:

```bash
python spiking/run_with_radas.py supp3d --seed 0
```

The command writes `supplementary_figure3d.csv`, a one-row-per-trial summary,
PDF, SVG, and `supp3d_run.json` under
`spiking/generated/radas/supp3d/<run-id>/`. The direct sequential equivalent is
`python spiking/supplementary_figure3d.py`. Estimated runtime for the full
grid: **15–60 minutes** on a desktop CPU.

### Plot-only paths

Once the corresponding CSV exists, plots can be regenerated without repeating
the numerical simulation:

```bash
python spiking/figure5.py --action plot
python spiking/figure6.py --action plot
python spiking/figure7.py --action plot
python spiking/supplementary_figure3d.py --plot-only
```

The last command uses the sequential script's default CSV. To replot the
mini-radas result in place, pass
`--results-path spiking/generated/radas/supp3d/<run-id>/supplementary_figure3d.csv`
and select the desired `--pdf-output` and `--svg-output` paths.

Estimated runtime: **under 1 minute per command**. Figures 6i–j and 7b do not
have a plot-only flag; their default fixed-configuration paths are the quick
reproduction paths described above.

All runtimes in this section are conservative estimates, not measurements.
They vary with CPU, storage, process-startup overhead, and selected options.

## 4. Command-line options

Every entry point supports `--help`. The following examples show the main
scientific controls.

Figure 5 defaults to the executable historical source (`alpha=0.2`). To use
the value printed in the manuscript caption (`alpha=0.1`), or to override it
explicitly:

```bash
python spiking/figure5.py --parameter-set manuscript
python spiking/figure5.py --learning-rate 0.05 --gamma 0.1
```

Change the Figure 6 delay grid and the parameter values displayed in panels
6d–g:

```bash
python spiking/figure6.py \
  --delay-start-ms -60 --delay-stop-ms 60 --delay-step-ms 2 \
  --membrane-time-constants-ms 5 10 20 \
  --alpha-values 0.005 0.01 0.02 \
  --tau-values-ms 0 3 6 \
  --gamma-values -0.5 0.02 0.1 0.5
```

Figures 6i–j default to the historical observation-index variance weighting.
The manuscript describes a strict `+/-5 ms` time window instead; that mode is
available explicitly and records the numerical variance floor in its JSON:

```bash
python spiking/fit_stdp.py --manuscript-weighting
python spiking/run_with_radas.py fig6i \
  --weighting manuscript --variance-floor 1e-12 \
  --num-samples 2000 --seed 7 --cpus-per-trial 1
```

The legacy rule uses five neighboring observations on either side in sorted
index order, not observations within five milliseconds. This distinction is
documented in [`data/README.md`](data/README.md). The embedded configurations
were selected with `--weighting legacy`.

Change the initial weights or other STDP parameters in Figure 7a:

```bash
python spiking/figure7.py \
  --initial-weights 0.1 0.3 0.5 0.7 0.9 \
  --transmission-delay-ms 3 --learning-rate 0.02
```

For the Figure 7b refit, the migrated notebook sampled the mapped model-weight
range from `[0.5, 2.0]`, while Methods Section H states `[0.0, 0.3]`. The
reported best value, `0.6515770896337575`, lies outside the Methods interval.
The historical search range is therefore the default, and both choices remain
available:

```bash
python spiking/run_with_radas.py fig7b --search-space legacy --seed 0
python spiking/run_with_radas.py fig7b --search-space manuscript --seed 0
```

Use a reduced profile or a single Supplementary Figure 3d trajectory while
checking an installation:

```bash
python spiking/run_with_radas.py supp3d --quick --seed 0
python spiking/supplementary_figure3d.py \
  --initial-wf1 -0.5 --initial-wf2 0.5 \
  --num-dataset-iterations 16 --seed 7
```

The runner also supports `--smoke`, `--storage`, `--output`,
`--experiment-name`, `--user-name`, `--cpus-per-trial`, and `--no-plots`.
See `python spiking/run_with_radas.py --help` for the complete interface.
By default, the materialized directory is isolated by target and run ID, so
full, quick, smoke, seeded, and scientifically distinct runs do not mix. Use
`--output` when a fixed human-readable destination is preferable.

## Reproducibility notes

- Figure 5 retains the notebook's historical `alpha=0.2` as its default and
  exposes the caption's `alpha=0.1` through `--parameter-set manuscript`.
- The old Figure 5 plotting helper displayed the opposite sign for the
  dendritic-error trace. This implementation displays the sign used by the
  manuscript update equation; the weight update itself is unchanged.
- Figure 6 fit curves are normalized so their sum at the observed delays
  matches the sum of the digitized data, reproducing the objective used by the
  migrated notebooks.
- The Supplementary Figure 3d notebook labeled post-update trajectory points
  `0` through `63`. The migrated CSV labels the same numerical states `1`
  through `64` in `dataset_iteration`, explicitly treating the coordinate as
  the number of completed dataset passes, and retains the original `0` through
  `63` labels in `source_iteration_index`; the update sequence is unchanged.
- Fresh searches and trajectory grids use an explicit seed. JSON and CSV
  outputs retain the selected parameter or search-space variant for audit.
