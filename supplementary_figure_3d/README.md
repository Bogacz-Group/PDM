# Supplementary Figure 3d

This directory reproduces the spiking XOR weight trajectories in
Supplementary Figure 3d. Source provenance is recorded in `provenance.json`.

## 1. System requirements

- 64-bit Linux, Python 3.11.14, and a CPU.
- Exact package versions are pinned in `requirements.txt`, including public
  [`mini-radas`](https://github.com/YuhangSong/mini-radas) commit
  `8314d91b773db5148b0a90a8442f549e367e9cfb`.

## 2. Installation

```bash
cd PDM/supplementary_figure_3d
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
`generated/radas/<run-id>/`. Estimated runtime: **15–60 minutes**.

The sequential equivalent is:

```bash
python supplementary_figure3d.py
```

It writes `results/supplementary_figure3d.csv` and
`plots/supplementary_figure3d.{pdf,svg}`. Estimated runtime: **15–60 minutes**.

## 4. Command-line options

```bash
python supplementary_figure3d.py \
  --initial-wf1 -0.5 --initial-wf2 0.5 \
  --num-dataset-iterations 16 --seed 7
python run_with_radas.py --quick --seed 7
```

`dataset_iteration` records completed passes (`1`–`64`) and
`source_iteration_index` retains the notebook labels (`0`–`63`). All entry
points support `--help`.
