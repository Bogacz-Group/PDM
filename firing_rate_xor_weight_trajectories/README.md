# Supplementary Figure 3c

This directory reproduces the firing-rate XOR weight trajectories in
Supplementary Figure 3c. The source is
`experiments/learn/XOR_record_wf_field.ipynb` at predictive_dendrite commit
[`370b8da`](https://github.com/YuhangSong/predictive_dendrite/commit/370b8daebcd6d52e37075f362f29e3a6c59e1ca3).

## 1. System requirements

- 64-bit Linux, Python 3.11.14, and a CPU.
- Minimal package versions are pinned in `requirements.txt`.

## 2. Installation

```bash
cd PDM/firing_rate_xor_weight_trajectories
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
python weight_trajectories.py
python plot_weight_trajectories.py
```

This writes `results/supplementary_figure3c.csv` and
`plots/supplementary_figure3c.{pdf,svg}`. Estimated numerical runtime:
**1–10 minutes**; plotting takes **under 1 minute**.

The optional mini-radas equivalent is:

```bash
python run_with_radas.py --seed 0
```

It writes trajectories, a trial table, PDF/SVG plots, and run metadata under
`generated/radas/<run-id>/` with similar numerical runtime.

## 4. Command-line options

```bash
python weight_trajectories.py \
  --grid-min -1 --grid-max 1 --grid-step 0.5 \
  --learning-rate 0.05 --num-dataset-iterations 128 --seed 7
python run_with_radas.py --quick --seed 7
```

The source uses 128 whole-dataset optimizer steps, equivalent to 512 sample
contributions; both counts are recorded. All entry points support `--help`.
