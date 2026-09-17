# Spiking experiments

This folder reproduces Figures 5–7 and Supplementary Figure 3d. Numerical
results are written to CSV/JSON and plots to PDF/SVG. Figure 5a is a conceptual
schematic and has no numerical reproduction script.

| Paper panels | Entry point |
|---|---|
| Figure 5b–c | `figure5.py` |
| Figure 6a–h | `figure6.py` (`analytic.py` for 6h only) |
| Figure 6i–j | `fit_stdp.py` |
| Figure 7a | `figure7.py` |
| Figure 7b | `fit_initial_strength.py` |
| Supplementary Figure 3d | `supplementary_figure3d.py` |

Original notebook paths and the pinned source revision are recorded in
[`../migration_manifest.json`](../migration_manifest.json).

## 1. System requirements

- 64-bit Linux and Python 3.11.14.
- A CPU is sufficient; no GPU is required.
- Exact package versions are pinned in [`requirements.txt`](requirements.txt).
  This includes the public
  [`mini-radas`](https://github.com/YuhangSong/mini-radas) package at commit
  `8314d91b773db5148b0a90a8442f549e367e9cfb`.

## 2. Installation

```bash
cd PDM/spiking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd ..
```

Estimated installation time: **10–20 minutes**. To verify the installation
(typically **under 1 minute**), run from the repository root:

```bash
python -m pytest spiking/tests -q
```

## 3. Reproduction

Run commands from the repository root with the environment activated.

| Panels | Command | Main output | Estimated runtime |
|---|---|---|---|
| Figure 5b–c | `python spiking/figure5.py` | `spiking/results/figure5.*`, `spiking/plots/figure5*` | 1–3 min |
| Figure 6a–h | `python spiking/figure6.py` | `spiking/results/figure6.*`, `spiking/plots/figure6*` | 5–20 min |
| Figure 6i–j | `python spiking/fit_stdp.py` | `spiking/generated/fig6_fits/` | 1–5 min |
| Figure 7a | `python spiking/figure7.py` | `spiking/results/figure7a.*`, `spiking/plots/figure7a.*` | 1–5 min |
| Figure 7b | `python spiking/fit_initial_strength.py` | `spiking/generated/fig7b_fit/` | 1–5 min |
| Supplementary Figure 3d | `python spiking/run_with_radas.py supp3d --seed 0` | CSV, trials, PDF/SVG, and metadata under `spiking/generated/radas/supp3d/<run-id>/` | 15–60 min |

The Figure 6i–j and 7b commands above evaluate the best configurations from
the original searches. Fresh seeded 1,000-trial searches are available with:

```bash
python spiking/run_with_radas.py fig6i --seed 0
python spiking/run_with_radas.py fig6j --seed 0
python spiking/run_with_radas.py fig7b --seed 0
```

Each fresh search writes fitted data, trials, plots, and run metadata under
`spiking/generated/radas/<target>/<run-id>/` and is estimated to take
**1–12 hours**.

Existing numerical results can be replotted without rerunning simulations:

```bash
python spiking/figure5.py --action plot
python spiking/figure6.py --action plot
python spiking/figure7.py --action plot
python spiking/supplementary_figure3d.py --plot-only \
  --results-path spiking/generated/radas/supp3d/RUN_ID/supplementary_figure3d.csv
```

Plot-only commands are estimated to take **under 1 minute** each.
Replace `RUN_ID` with the generated run-directory name.

## 4. Command-line options

Examples of the main scientific controls are:

```bash
# Figure 5: use the parameter value printed in the manuscript caption
python spiking/figure5.py --parameter-set manuscript

# Figure 6: change the STDP grid or use the manuscript weighting rule
python spiking/figure6.py \
  --delay-start-ms -60 --delay-stop-ms 60 --delay-step-ms 2 \
  --alpha-values 0.005 0.01 0.02
python spiking/run_with_radas.py fig6i \
  --weighting manuscript --num-samples 2000 --seed 7

# Figure 7: change initial weights or use the Methods search interval
python spiking/figure7.py --initial-weights 0.1 0.3 0.5 0.7 0.9
python spiking/run_with_radas.py fig7b \
  --search-space manuscript --seed 0

# Reduced Supplementary Figure 3d run
python spiking/run_with_radas.py supp3d --quick --seed 0
```

The source notebooks and manuscript differ in three places that affect these
options: Figure 5 uses `alpha=0.2` in the notebook but `alpha=0.1` in the
caption; the Figure 6 notebook uses observation-index variance weighting
rather than a strict `+/-5 ms` window; and the Figure 7b notebook searches a
weight range of `[0.5, 2.0]` while Methods states `[0.0, 0.3]`. The historical
best value for this range (`0.651577...`) lies outside the Methods interval, so
the historical implementations remain the defaults. Details are in
[`data/README.md`](data/README.md) and
[`configs/README.md`](configs/README.md).

The old Figure 5 plotting helper used the opposite sign for the displayed
dendritic-error trace; this implementation follows the manuscript equation
without changing the weight update. Figure 6 fit curves retain the notebook's
normalization to the digitized observations.

For Supplementary Figure 3d, `dataset_iteration` records completed passes
(`1`–`64`) and `source_iteration_index` retains the notebook labels
(`0`–`63`). Every entry point supports `--help` for the full option list.
