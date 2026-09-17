# Supplementary Figure 3c

This directory reproduces the firing-rate XOR weight trajectories in
Supplementary Figure 3c. Source provenance is recorded in `provenance.json`.

## 1. System requirements

- 64-bit Linux, Python 3.11.14, and a CPU.
- Exact package versions are pinned in `requirements.txt`, including public
  [`mini-radas`](https://github.com/YuhangSong/mini-radas) commit
  `8314d91b773db5148b0a90a8442f549e367e9cfb`.

## 2. Installation

```bash
cd PDM/supplementary_figure_3c
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Estimated installation time: **10–20 minutes**. Verify the environment with
`python -m pytest tests -q` (typically **under 1 minute**).

## 3. Reproduction

Run the complete 81-condition grid with mini-radas:

```bash
python run_with_radas.py --seed 0
```

This writes trajectories, a trial table, PDF/SVG plots, and run metadata under
`generated/radas/<run-id>/`. Estimated runtime: **1–10 minutes**.

The sequential equivalent is:

```bash
python weight_trajectories.py
python plot_weight_trajectories.py
```

It writes `results/supplementary_figure3c.csv` and
`plots/supplementary_figure3c.{pdf,svg}` with similar numerical runtime; plotting
takes **under 1 minute**.

## 4. Command-line options

```bash
python weight_trajectories.py \
  --grid-min -1 --grid-max 1 --grid-step 0.5 \
  --learning-rate 0.05 --num-dataset-iterations 128 --seed 7
python run_with_radas.py --quick --seed 7
```

The source uses 128 whole-dataset optimizer steps, equivalent to 512 sample
contributions; both counts are recorded. All entry points support `--help`.
