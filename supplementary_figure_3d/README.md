# Supplementary Figure 3d

This directory reproduces the spiking XOR weight trajectories in
Supplementary Figure 3d. The sources at predictive_dendrite commit
[`370b8da`](https://github.com/YuhangSong/predictive_dendrite/commit/370b8daebcd6d52e37075f362f29e3a6c59e1ca3)
are `simulation_features.ipynb`, `simulation_features.py`, and
`simulation_XOR.py`, all under `experiments/spiking/`.

## 1. System requirements

- 64-bit Linux, Python 3.11.14, and a CPU.
- Minimal package versions are pinned in `requirements.txt`.

## 2. Installation

```bash
cd PDM/supplementary_figure_3d
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Estimated installation time: **5–10 minutes**.

The reported panel does not require radas. The optional parallel grid runner
uses the public [`mini-radas`](https://github.com/YuhangSong/mini-radas)
repository, whose Python distribution and import name is `radas`. Install its
pinned stack only when needed:

```bash
python -m pip install -r requirements-search.txt
```

## 3. Reproduction

Run the complete 81-condition grid sequentially:

```bash
python supplementary_figure3d.py
```

It writes `results/supplementary_figure3d.csv` and
`plots/supplementary_figure3d.{pdf,svg}`. Estimated runtime: **15–60 minutes**.

The optional mini-radas equivalent is:

```bash
python run_with_radas.py --seed 0
```

It writes trajectories, a trial table, PDF/SVG plots, and run metadata under
`generated/radas/<run-id>/` with similar numerical runtime.

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
